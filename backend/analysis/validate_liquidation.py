"""Check `risk.liquidation_price()` against BloFin's own reported number.

    python backend\\analysis\\validate_liquidation.py
    python backend\\analysis\\validate_liquidation.py --production

Roadmap step 1 left this open: *"Before sizing anything with leverage, open a
small position on the BloFin demo account, read back the exchange's own
reported liquidation price, and compare."* It stayed open because it looked
blocked - the API will not quote a liquidation price for a position that does
not exist yet, so there is nothing to compare against until you have traded.

The way through is that **an open position reports it**. `GET
/api/v1/account/positions` returns `liquidationPrice`, `maintenanceMargin`,
`averagePrice`, `markPrice` and `leverage` per position, so one position that
already exists validates the model for free. No order is placed here; this is
a read.

Why it matters now, twice over
------------------------------
The carry trade in `funding_carry.py` is delta-neutral in price and **not** in
margin: its short perp leg is a leveraged position that can be liquidated on
its own. Five instruments profitable from every entry is worth nothing if the
position is closed in a wick because the maintenance-margin assumption was
wrong.

And the open-interest series in `trading/openinterest.py` is being collected
to estimate where *other* traders are liquidated. That model rolls their
positions through this same formula. An error here propagates into a
liquidation map with nothing to check it against.

What is compared
----------------
BloFin publishes (help centre, "Liquidation Price Calculation"):

    Cross, single position, USDT-margined, short:
        LP = (Account balance + |amount| * avg price)
             / (|amount| * (MMR + liquidation fee rate + 1))

`risk.liquidation_price()` derives the isolated form independently:

        LP(short) = (entry * (1 + 1/L) + M/qty) / (1 + MMR)

Those are the same equation. Expanding ours, `entry/L` is the initial margin
per unit and `M/qty` the extra margin per unit, so the numerator is
`entry + (initial + extra)/qty` against BloFin's `avg + balance/|amount|` -
identical once you accept that in cross margin the whole account balance backs
the position, which is what cross margin means.

The one real difference is the **liquidation fee rate**, which BloFin carries
in the denominator beside MMR and the clean derivation omits. `risk.py`
already has the right shape for it in `fee_buffer_bps`, a multiplicative pad
in the conservative direction; this script reports the rate implied by the
exchange's own number so that pad can be set from a measurement instead of the
"10-20 bps is reasonable" guess in its docstring.

MMR is not assumed either. It is read out as
`maintenanceMargin / (quantity * mark price)`, which is the tier actually
being applied to the size actually held - the thing the README warns is
tiered and must not be taken from the base rate.
"""

from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402,F401  (loads .env)
from trading.risk import Side, compare_to_exchange, liquidation_price  # noqa: E402

getcontext().prec = 34

DEMO_BASE_URL = "https://demo-trading-openapi.blofin.com"
PRODUCTION_BASE_URL = "https://openapi.blofin.com"

# Above this the model and the exchange disagree enough that sizing built on
# it is unsafe. Same bar `compare_to_exchange` applies.
TOLERANCE = Decimal("0.01")


def decimal_or_none(value) -> Optional[Decimal]:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001 - a missing field is not an error here
        return None


def contract_values(client) -> Dict[str, Decimal]:
    """instId -> contractValue, for converting contracts to base units.

    Positions are quoted in CONTRACTS. Every quantity in `risk.py` is in base
    units, and on an instrument whose contract is not 1 unit the two differ by
    exactly that factor - which would look like a plausible-but-wrong
    liquidation price rather than an obvious failure.
    """
    payload = client.get("/api/v1/market/instruments", params={}, sign=False)
    values: Dict[str, Decimal] = {}
    for row in payload.get("data") or []:
        value = decimal_or_none(row.get("contractValue"))
        if row.get("instId") and value:
            values[row["instId"]] = value
    return values


def check_position(position: dict, *, account_balance: Decimal,
                   contract_value: Decimal) -> Dict[str, object]:
    """Model vs exchange for one open position."""
    contracts = decimal_or_none(position.get("positions")) or Decimal(0)
    entry = decimal_or_none(position.get("averagePrice"))
    mark = decimal_or_none(position.get("markPrice"))
    reported = decimal_or_none(position.get("liquidationPrice"))
    maintenance = decimal_or_none(position.get("maintenanceMargin"))
    leverage = decimal_or_none(position.get("leverage"))

    result: Dict[str, object] = {
        "instId": position.get("instId"),
        "marginMode": position.get("marginMode"),
        "leverage": leverage,
        "reported": reported,
    }
    if not (entry and mark and reported and maintenance and leverage) or contracts == 0:
        result["error"] = "position is missing a field the comparison needs"
        return result

    side = Side.SHORT if contracts < 0 else Side.LONG
    quantity = abs(contracts) * contract_value
    notional = quantity * mark
    if notional <= 0:
        result["error"] = "zero notional"
        return result

    # The tier actually applied to this size, not the base rate.
    mmr = maintenance / notional
    initial_margin = quantity * entry / leverage

    if (position.get("marginMode") or "").lower() == "cross":
        # Cross margin: the whole account backs the position, so the balance
        # takes the place of initial + extra.
        backing = account_balance
    else:
        backing = decimal_or_none(position.get("margin")) or initial_margin
    extra = backing - initial_margin

    estimate = liquidation_price(
        entry_price=entry,
        leverage=leverage,
        side=side,
        maintenance_margin_rate=mmr,
        extra_margin=extra,
        quantity_base=quantity,
    )

    result.update({
        "side": side.name,
        "quantityBase": quantity,
        "entry": entry,
        "mark": mark,
        "mmr": mmr,
        "estimate": estimate,
        "comparison": compare_to_exchange(estimated=estimate,
                                          exchange_reported=reported),
    })

    # What fee rate would close the gap, given the rest of the model is right?
    # BloFin carries it in the denominator beside MMR; ours omits it, so the
    # residual IS that rate if the formula is otherwise correct.
    if estimate:
        sign = Decimal(1) if side is Side.SHORT else Decimal(-1)
        implied = (backing + sign * Decimal(0) + quantity * entry) / (
            reported * quantity) - Decimal(1) - mmr
        result["impliedFeeRate"] = implied if side is Side.SHORT else None
        result["padBps"] = (
            abs(estimate - reported) / reported * Decimal(10_000))
    return result


def report(results: List[Dict[str, object]], *, environment: str) -> None:
    print("\n" + "=" * 84)
    print(f"LIQUIDATION MODEL vs BLOFIN  ({environment})")
    print("=" * 84)

    usable = [row for row in results if "error" not in row]
    if not usable:
        print("\n  No comparable positions.")
        for row in results:
            print(f"    {row.get('instId')}: {row.get('error')}")
        print("\n  Open a small position on the demo account and re-run. This "
              "script places no\n  orders - it can only read a liquidation "
              "price that already exists.")
        return

    print(f"\n  {'instrument':<13}{'side':>6}{'lev':>5}{'mode':>7}{'MMR':>8}"
          f"{'model':>13}{'exchange':>13}{'rel err':>10}")
    print("  " + "-" * 80)
    for row in usable:
        comparison = row["comparison"]
        estimate = row.get("estimate")
        error = comparison.get("relativeError")
        print(f"  {str(row['instId']):<13}{row['side']:>6}"
              f"{row['leverage']:>5}{str(row['marginMode']):>7}"
              f"{float(row['mmr']):>8.4f}"
              f"{float(estimate) if estimate else float('nan'):>13.4f}"
              f"{float(row['reported']):>13.4f}"
              f"{float(error) if error is not None else float('nan'):>9.4%}"
              f"{'  OK' if comparison.get('ok') else '  MISMATCH'}")

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)

    bad = [row for row in usable if not row["comparison"].get("ok")]
    if bad:
        print(f"  {len(bad)} position(s) disagree by more than "
              f"{float(TOLERANCE):.0%}.")
        print("  The MMR tier or the margin-mode handling is wrong. Do not "
              "size anything with\n  leverage until this matches - and note "
              "the carry trade's short perp leg is\n  exactly such a "
              "position.")
        return

    print(f"  All {len(usable)} position(s) agree within "
          f"{float(TOLERANCE):.0%}. The formula and the MMR read off the "
          f"position\n  are consistent with what the exchange itself reports.")

    for row in usable:
        mmr = row["mmr"]
        pad = row.get("padBps")
        implied = row.get("impliedFeeRate")
        print(f"\n  {row['instId']}:")
        print(f"    MMR actually applied at this size   {float(mmr):.5f} "
              f"({float(mmr) * 100:.3f}%)")
        if pad is not None:
            print(f"    residual vs exchange                "
                  f"{float(pad):.2f} bps")
        if implied is not None:
            print(f"    implied liquidation fee rate        "
                  f"{float(implied):.6f}")
            print(f"    -> set fee_buffer_bps to about      "
                  f"{float(implied) * 10_000:.1f}")

    print("\n  The residual is the liquidation FEE rate, which BloFin carries "
          "in the\n  denominator beside MMR and the clean derivation in "
          "risk.py omits. It is a\n  fraction of a basis point of price here, "
          "and it points the unsafe way -\n  the model puts liquidation "
          "slightly further from entry than it really is.\n  "
          "`fee_buffer_bps` exists to absorb exactly that; the number above is "
          "what to\n  set it to, measured rather than the 10-20 its docstring "
          "guesses at.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--production", action="store_true",
                        help="Read the live account instead of demo. Reads "
                             "only; places nothing.")
    args = parser.parse_args(argv)

    from blofin.client import Client

    base_url = PRODUCTION_BASE_URL if args.production else DEMO_BASE_URL
    environment = "production" if args.production else "demo"

    api_key = os.environ.get("API_KEY")
    secret = os.environ.get("SECRET")
    passphrase = os.environ.get("PASSPHRASE")
    if not (api_key and secret and passphrase):
        raise SystemExit(
            "API_KEY, SECRET and PASSPHRASE must be set in .env. This reads "
            "account state,\nso unlike the market-data tools it does need "
            "credentials.")

    client = Client(apiKey=api_key, apiSecret=secret, passphrase=passphrase,
                    baseUrl=base_url)

    balance_payload = client.get("/api/v1/account/balance", params={}, sign=True)
    if str(balance_payload.get("code")) != "0":
        raise SystemExit(
            f"Could not read the {environment} account: "
            f"{balance_payload.get('msg')}\n"
            "A demo key against the production host (or the reverse) returns "
            "'Access key does not exist'.")
    account_balance = decimal_or_none(
        (balance_payload.get("data") or {}).get("totalEquity")) or Decimal(0)

    positions_payload = client.get("/api/v1/account/positions", params={},
                                   sign=True)
    positions = positions_payload.get("data") or []
    print(f"{environment}: equity {account_balance:.2f}, "
          f"{len(positions)} open position(s).")

    values = contract_values(client)
    results = [
        check_position(position, account_balance=account_balance,
                       contract_value=values.get(position.get("instId"),
                                                 Decimal(1)))
        for position in positions
    ]
    report(results, environment=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
