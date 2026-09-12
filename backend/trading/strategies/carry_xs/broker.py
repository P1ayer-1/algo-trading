"""The `Broker` protocol for a perp-only book, against BloFin's REST API.

Thin on purpose, exactly like `strategies/carry/broker.py`: every method is one
call and no logic, so the ordering, the imbalance repair and the verification
stay in `execute.py` where the tests can reach them without a network.

Narrower than the carry broker, and deliberately so. This book never touches
spot, so there is no `place_spot`, no `spot_balance` and no `transfer` - and a
broker that cannot reach the spot endpoints cannot mis-hedge a leg in base
units when it meant quote, which is the bug that cost the first live carry
$0.265 in step 9h. Capability removed is capability that cannot be misused.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional


def _decimal(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return Decimal(default)


class BlofinPerpBroker:
    """Every exchange call a cross-sectional book needs, and no decisions."""

    def __init__(self, client, trading_api, market_api):
        self.client = client
        self.trading = trading_api
        self.market = market_api

    def margin_mode(self) -> str:
        payload = self.client.get("/api/v1/account/margin-mode", params={},
                                  sign=True)
        return str((payload.get("data") or {}).get("marginMode", "unknown"))

    def set_leverage(self, inst_id: str, leverage: Decimal) -> Dict[str, Any]:
        return self.trading.setLeverage(
            instId=inst_id, leverage=str(int(leverage)), marginMode="cross")

    def place_perp(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str,
                   reduce_only: bool = False) -> Dict[str, Any]:
        return self.trading.placeOrder(
            instId=inst_id, marginMode="cross", positionSide="net",
            side=side, orderType="market", size=str(size),
            reduceOnly="true" if reduce_only else "false",
            clientOrderId=client_order_id)

    def positions(self) -> List[Dict[str, Any]]:
        """Every open perp position. The whole book in one call.

        Per-instrument reads would give a book assembled from a dozen
        timestamps, and the one number this executor most needs to be right
        about - the NET exposure across all legs - would then be a sum of
        readings taken at different moments.
        """
        payload = self.client.get("/api/v1/account/positions", params={},
                                  sign=True)
        return [row for row in (payload.get("data") or []) if isinstance(row, dict)]

    def contract_values(self) -> Dict[str, Decimal]:
        """`{inst_id: base units per contract}` for every listed swap.

        One call, and it covers instruments the plan does not mention - a
        leftover position from some earlier run still has to be priced, and
        pricing it on contract count would be wrong by the contract multiplier,
        which on BloFin ranges from 0.001 (BTC) to 1000 (DOGE).
        """
        payload = self.market.getInstruments()
        out: Dict[str, Decimal] = {}
        for row in payload.get("data") or []:
            inst_id = str(row.get("instId") or "")
            value = _decimal(row.get("contractValue"), "0")
            if inst_id and value > 0:
                out[inst_id] = value
        return out

    def quote(self, inst_id: str) -> Optional[Dict[str, Decimal]]:
        """Best bid and ask, now. Used to re-check a plan's spread before
        sending, because a plan is minutes old by the time it is executed and
        the spread on a thin alt is the largest term in its cost."""
        payload = self.market.getTickers(instId=inst_id)
        for row in payload.get("data") or []:
            if row.get("instId") == inst_id:
                bid, ask = _decimal(row.get("bidPrice")), _decimal(row.get("askPrice"))
                if bid > 0 and ask > bid:
                    return {"bid": bid, "ask": ask, "last": _decimal(row.get("last"))}
        return None
