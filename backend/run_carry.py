"""Open a planned carry on the demo account. Dry run unless --confirm.

    python backend\\run_carry.py --instrument SUI-USDT --notional 200
    python backend\\run_carry.py --instrument SUI-USDT --notional 200 --confirm

Without `--confirm` this rehearses: it builds the plan against live prices,
walks every step the open would take, and prints them without sending
anything. With `--confirm` it sends them.

Demo by default and demo on purpose. `--production` exists so the flag has to
be typed rather than defaulted into, and there is no reason to type it until a
carry has been run end to end here first.

The order, and the failure it is chosen for
-------------------------------------------
Perp leg first, spot leg second. Between the two the position is directional,
so the choice is about which exposure to be holding if the second leg fails.
A failed spot leg leaves a SHORT perp - closeable instantly, on the deeper
book, with a `reduce_only` order - and that is what gets unwound automatically.
A failed perp leg leaves nothing on at all, which is the cheapest outcome
available. The reverse order would strand you long spot.

What it checks that a plan cannot
---------------------------------
The plan assumes a maintenance margin rate. MMR is tiered and
instrument-specific - 0.500% measured on SOL-USDT, 0.300% on BTC-USDT - so it
is a guess until a position exists. After both legs are on, this reads the
position back and compares the exchange's own liquidation price to the planned
one. Closer than planned is reported as a problem, immediately.

Margin mode is checked and never changed: on BloFin it is an account-wide
setting, so flipping it for one carry would re-margin every other open
position.
"""

from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SPOT_TAKER_FEE_BPS, TAKER_FEE_BPS, VIP_TIER  # noqa: E402
from plan_carry import (  # noqa: E402
    DEMO_BASE_URL,
    MEASURED_FEE_BUFFER_BPS,
    MEASURED_MMR,
    PRODUCTION_BASE_URL,
    build_market,
    decimal_of,
    funding_per_day_bps,
    read_wallets,
    report as report_plan,
)
from server import log  # noqa: E402
from trading.carry import plan_carry  # noqa: E402
from trading.carry_executor import CarryExecutor  # noqa: E402
from trading.risk import RiskLimits  # noqa: E402


class BlofinBroker:
    """The `Broker` protocol, against BloFin's REST API.

    Thin on purpose: every method is one call and no logic, so the ordering,
    unwinding and verification stay in `carry_executor` where the tests can
    reach them without a network.
    """

    def __init__(self, client, trading_api):
        self.client = client
        self.trading = trading_api

    def transfer(self, *, currency: str, amount: Decimal,
                 from_account: str, to_account: str) -> Dict[str, Any]:
        return self.trading.transfer(
            currency=currency, amount=str(amount),
            fromAccount=from_account, toAccount=to_account)

    def margin_mode(self) -> str:
        payload = self.client.get("/api/v1/account/margin-mode", params={},
                                  sign=True)
        return str((payload.get("data") or {}).get("marginMode", "unknown"))

    def set_leverage(self, inst_id: str, leverage: Decimal) -> Dict[str, Any]:
        return self.trading.setLeverage(
            instId=inst_id, leverage=str(int(leverage)), marginMode="isolated")

    def place_perp(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str,
                   reduce_only: bool = False) -> Dict[str, Any]:
        return self.trading.placeOrder(
            instId=inst_id, marginMode="isolated", positionSide="net",
            side=side, orderType="market", size=str(size),
            reduceOnly="true" if reduce_only else "false",
            clientOrderId=client_order_id)

    def place_spot(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str) -> Dict[str, Any]:
        return self.client.post("/api/v1/spot/trade/order", {
            "instId": inst_id, "side": side, "orderType": "market",
            "size": str(size), "clientOrderId": client_order_id,
        })

    def perp_position(self, inst_id: str) -> Optional[Dict[str, Any]]:
        payload = self.client.get("/api/v1/account/positions",
                                  params={"instId": inst_id}, sign=True)
        for row in payload.get("data") or []:
            if row.get("instId") == inst_id:
                return row
        return None

    def spot_balance(self, currency: str) -> Decimal:
        payload = self.client.get("/api/v1/asset/balances",
                                  params={"accountType": "spot"}, sign=True)
        for row in payload.get("data") or []:
            if row.get("currency") == currency:
                return decimal_of(row.get("available"))
        return Decimal(0)


def report_execution(result, *, hold_days: Decimal) -> None:
    print("\n" + "=" * 78)
    mode = "DRY RUN - NOTHING SENT" if result.dry_run else "LIVE"
    print(f"EXECUTION  {result.inst_id}  ({mode})")
    print("=" * 78)

    for step in result.steps:
        marker = "ok " if step.ok else "FAIL"
        sent = "" if step.sent else "  (not sent)"
        print(f"  [{marker}] {step.name:<10} {step.detail}{sent}")
        if step.error:
            print(f"         {step.error}")

    if result.actual_liquidation is not None:
        print("\n  VERIFIED AGAINST THE EXCHANGE")
        print(f"    planned liquidation  {result.planned_liquidation}")
        print(f"    actual liquidation   {result.actual_liquidation}")
        if result.actual_mmr is not None:
            print(f"    MMR actually charged {result.actual_mmr:.5f}  "
                  f"(plan assumed {MEASURED_MMR})")

    print("\n" + "=" * 78)
    if result.dry_run:
        print("REHEARSAL COMPLETE")
        print("=" * 78)
        print("  Nothing was sent. Re-run with --confirm to place these "
              "orders on the demo\n  account.")
    elif result.opened and result.ok:
        print("CARRY IS ON")
        print("=" * 78)
        print(f"  Both legs filled and the exchange's liquidation price "
              f"matches the plan.\n  Hold it and let funding accrue; the point "
              f"of the test is comparing realised\n  against the "
              f"{hold_days}-day prediction, which needs days, not minutes.")
    elif result.unwound:
        print("UNWOUND - NOTHING IS OPEN")
        print("=" * 78)
        for problem in result.problems:
            print(f"  - {problem}")
    else:
        print("PROBLEMS")
        print("=" * 78)
        for problem in result.problems:
            print(f"  - {problem}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--notional", type=Decimal, default=Decimal("200"),
                        help="Target size in USD. Start small.")
    parser.add_argument("--leverage", type=Decimal, default=Decimal("3"))
    parser.add_argument("--hold-days", type=Decimal, default=Decimal("30"))
    parser.add_argument("--funding-pages", type=int, default=6)
    parser.add_argument("--confirm", action="store_true",
                        help="Actually send the orders. Without this it is a "
                             "rehearsal.")
    parser.add_argument("--production", action="store_true",
                        help="Against the live account rather than demo. Do "
                             "not use this until a carry has run end to end "
                             "on demo.")
    args = parser.parse_args(argv)

    if SPOT_TAKER_FEE_BPS is None:
        raise SystemExit(
            f"No spot fee schedule for VIP {VIP_TIER} in config.SPOT_VIP_TIERS.")

    from blofin.client import Client
    from blofin.rest_market import MarketAPI
    from blofin.rest_trading import TradingAPI

    base_url = PRODUCTION_BASE_URL if args.production else DEMO_BASE_URL
    environment = "production" if args.production else "demo"
    if args.production and args.confirm:
        raise SystemExit(
            "--production with --confirm sends real orders with real money, "
            "and this tool\nhas never done that. Run the carry on demo first, "
            "hold it, and compare the\nrealised result against the "
            "prediction. Then decide deliberately.")

    api_key = os.environ.get("API_KEY")
    secret = os.environ.get("SECRET")
    passphrase = os.environ.get("PASSPHRASE")
    if not (api_key and secret and passphrase):
        raise SystemExit("API_KEY, SECRET and PASSPHRASE must be set in .env.")

    client = Client(apiKey=api_key, apiSecret=secret, passphrase=passphrase,
                    baseUrl=base_url)
    public = Client()
    api = MarketAPI(public)

    log(f"{environment}: building the plan from live prices...")
    market = build_market(public, api, args.instrument)
    if market is None:
        return 1
    wallets = read_wallets(client)
    daily = funding_per_day_bps(api, args.instrument, args.funding_pages)

    plan = plan_carry(
        market=market, wallets=wallets,
        target_notional_usd=args.notional, leverage=args.leverage,
        funding_per_day_bps=daily, hold_days=args.hold_days,
        spot_taker_bps=Decimal(str(SPOT_TAKER_FEE_BPS)),
        perp_taker_bps=Decimal(str(TAKER_FEE_BPS)),
        limits=RiskLimits(),
        maintenance_margin_rate=MEASURED_MMR,
        fee_buffer_bps=MEASURED_FEE_BUFFER_BPS,
    )
    report_plan(plan, market, wallets, hold_days=args.hold_days,
                environment=environment)
    if not plan.ok:
        return 1

    if args.confirm:
        log("")
        log("--confirm given: these orders WILL be sent.")

    executor = CarryExecutor(
        BlofinBroker(client, TradingAPI(client)),
        dry_run=not args.confirm,
        on_log=log,
    )
    result = executor.open(plan, base_currency=args.instrument.split("-")[0])
    report_execution(result, hold_days=args.hold_days)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
