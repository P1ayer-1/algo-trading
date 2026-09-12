"""Print exactly what a cross-sectional carry book would do. Sends nothing.

    python backend\\plan_carry_xs.py --notional 2000
    python backend\\plan_carry_xs.py --notional 5000 --leverage 3 --top-frac 0.2
    python backend\\plan_carry_xs.py --notional 2000 --min-volume 10000000

The strategy is step 9q: long the perps whose recent funding is lowest, short
the ones whose funding is highest, dollar-neutral, rebalanced weekly. It
measured +40.5 bps a week net of twice the taker fee over five years of
Binance, and beat 100%, 99% and 95% of 200 shuffled controls on Binance,
BloFin and Hyperliquid respectively.

This places no orders. There is no code path from here to `placeOrder` and
`trading/strategies/carry_xs` imports no broker, which is checkable with a
grep and is checked that way. Every sizing, rounding and margin question gets
settled while the answer is still text on a screen.

Read the REFUSALS first, then the two standing warnings. Unlike the spot/perp
carry this book has real price exposure - the legs are different coins and
nothing cancels - and the planner states its measured size rather than leaving
it to be found out.

What it reads, and how much that costs
--------------------------------------
`getTickers` and `getInstruments` are one call each and cover quotes, volume
and the size rules. Funding history and daily candles are one call per
instrument that passes the volume filter, which is about fifty on this venue,
so the whole plan is roughly a hundred requests and a few seconds.

The funding score is the trailing 7-day mean, expressed per day and NEGATED,
matching the research: high score means expected to outperform, so the book
goes long it. An instrument with less than `--min-funding-days` of history is
excluded and said so, rather than scored on whatever arrived.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TAKER_FEE_BPS, VIP_TIER  # noqa: E402
from trading.risk import RiskLimits, to_decimal  # noqa: E402
from trading.strategies.carry_xs import (  # noqa: E402
    BookConfig,
    BookPlan,
    Candidate,
    plan_book,
)

DAY_MS = 86_400_000


def market_api():
    from blofin.client import Client
    from blofin.rest_market import MarketAPI
    return MarketAPI(Client())


def _float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def daily_funding(api, inst_id: str) -> Dict[str, float]:
    """`{date: bps accrued that UTC day}` over one page of history.

    One page is 100 settlements, so about 33 days at an 8-hourly cadence and
    fewer on a 4-hourly instrument - enough for both the 7-day score and the
    history check, in a single call per instrument.

    Same bucketing rule as every panel in `analysis/`: a settlement stamped T
    paid for the interval ENDING at T, so it belongs to the day before T, with
    the stamp rounded to its nominal minute first. Getting this wrong here
    while the research gets it right would put the live score a day out of step
    with the one that was measured.

    The newest day is DROPPED. The page always includes the settlement that has
    just printed, and the UTC day it belongs to is usually still in progress,
    so its total is a fraction of a day's carry that would read as unusually
    cheap funding on every instrument at once.
    """
    payload = api.getFundingRateHistory(inst_id, limit="100").get("data") or []
    out: Dict[str, float] = {}
    for row in payload:
        try:
            ts = int(row["fundingTime"])
            rate = float(row["fundingRate"]) * 10_000.0
        except (KeyError, TypeError, ValueError):
            continue
        nominal = int(round(ts / 60_000.0)) * 60_000
        key = datetime.fromtimestamp((nominal - 1) / 1000.0,
                                     timezone.utc).strftime("%Y-%m-%d")
        out[key] = out.get(key, 0.0) + rate
    ordered = sorted(out.items())
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {day: value for day, value in ordered if day != today}


def daily_vol_bps(api, inst_id: str, *, days: int = 31) -> float:
    """Standard deviation of daily log returns over `days`, in bps.

    This is the risk measure the research sized on, computed the same way from
    closes, so the live book and the backtested one weight positions alike.
    """
    import math

    payload = api.getCandlesticks(inst_id, bar="1D",
                                  limit=str(days + 2)).get("data") or []
    closes: List[float] = []
    for row in payload:
        try:
            if str(row[8]) == "0":              # the day still in progress
                continue
            closes.append((int(row[0]), float(row[4])))
        except (IndexError, TypeError, ValueError):
            continue
    closes.sort()
    values = [close for _, close in closes]
    steps = [math.log(b / a) for a, b in zip(values, values[1:])
             if a > 0 and b > 0]
    if len(steps) < 10:
        return 0.0
    mean = sum(steps) / len(steps)
    variance = sum((s - mean) ** 2 for s in steps) / (len(steps) - 1)
    return math.sqrt(variance) * 10_000.0


def gather(api, *, min_volume: float, carry_days: int, min_funding_days: int,
           limit: Optional[int] = None) -> List[Candidate]:
    tickers = api.getTickers().get("data") or []
    instruments = {row.get("instId"): row
                   for row in (api.getInstruments().get("data") or [])}

    shortlist = []
    for row in tickers:
        inst_id = str(row.get("instId", ""))
        if not inst_id.endswith("-USDT") or inst_id not in instruments:
            continue
        if str(instruments[inst_id].get("state", "live")) != "live":
            continue
        last = _float(row.get("last"))
        volume = _float(row.get("volCurrency24h")) * last
        bid, ask = _float(row.get("bidPrice")), _float(row.get("askPrice"))
        if volume < min_volume or bid <= 0 or ask <= bid:
            continue
        shortlist.append((volume, inst_id, bid, ask))
    shortlist.sort(reverse=True)
    if limit:
        shortlist = shortlist[:limit]

    print("Reading funding and candles for {} instruments over ${:,.0f}/24h..."
          .format(len(shortlist), min_volume))
    candidates: List[Candidate] = []
    for index, (volume, inst_id, bid, ask) in enumerate(shortlist, start=1):
        meta = instruments[inst_id]
        try:
            funding = daily_funding(api, inst_id)
            vol = daily_vol_bps(api, inst_id)
        except Exception as exc:                 # noqa: BLE001
            print("\n  " + inst_id + ": skipped (" + str(exc) + ")")
            continue
        # The score is the trailing mean over `carry_days`, negated, so that
        # high means "expected to outperform" - the convention the research
        # fixed before its results were seen.
        recent = list(funding.values())[-carry_days:]
        score = -(sum(recent) / len(recent)) if recent else 0.0
        candidates.append(Candidate(
            inst_id=inst_id,
            carry_bps_per_day=score,
            bid=to_decimal(bid), ask=to_decimal(ask),
            volume_usd_24h=volume,
            vol_30d_bps=vol,
            contract_value=to_decimal(meta.get("contractValue"), "1"),
            lot_size=to_decimal(meta.get("lotSize"), "1"),
            min_size=to_decimal(meta.get("minSize"), "1"),
            max_leverage=to_decimal(meta.get("maxLeverage"), "10"),
            funding_days=len(funding)))
        print("\r  {}/{}  {:<18}".format(index, len(shortlist), inst_id),
              end="", flush=True)
    print()
    return candidates


def show(plan: BookPlan, config: BookConfig) -> None:
    print()
    print("=" * 78)
    print("CROSS-SECTIONAL CARRY BOOK".center(78))
    print("=" * 78)
    print("{} of {} instruments eligible; top/bottom {:.0%}; {}-day hold; "
          "VIP {}".format(plan.eligible, plan.considered, config.top_frac,
                          config.hold_days, VIP_TIER))
    print()

    for label, legs in (("LONG  (funding lowest - these pay you to hold them)",
                         plan.long_legs),
                        ("SHORT (funding highest - their longs pay you)",
                         plan.short_legs)):
        print(label)
        print("  {:<16}{:>12}{:>14}{:>12}{:>11}{:>10}".format(
            "instrument", "contracts", "notional $", "carry/day", "spread bps",
            "liq away"))
        for leg in sorted(legs, key=lambda item: -item.notional_usd):
            if not leg.ok:
                continue
            print("  {:<16}{:>12}{:>14,.2f}{:>12.3f}{:>11.2f}{:>10}".format(
                leg.inst_id, leg.contracts, leg.notional_usd,
                leg.carry_bps_per_day, leg.spread_bps,
                "{:.1%}".format(leg.liquidation_distance)
                if leg.liquidation_distance is not None else "-"))
        print()

    print("BOOK")
    print("  gross notional      ${:,.2f}".format(plan.gross_notional_usd))
    print("  net exposure        ${:,.2f}  ({:.2%} of gross)".format(
        plan.net_notional_usd,
        abs(plan.net_notional_usd) / plan.gross_notional_usd
        if plan.gross_notional_usd else 0))
    print("  margin at {}x       ${:,.2f}".format(plan.leverage, plan.margin_usd))
    print()
    print("ECONOMICS, per {}-day hold, per unit of gross notional".format(
        config.hold_days))
    print("  carry spread        {:+.3f} bps/day between the two baskets".format(
        plan.carry_spread_bps_per_day))
    print("  expected funding    {:+.1f} bps".format(plan.expected_funding_bps))
    print("  round trip          {:+.1f} bps  (taker + half spread, both ways)"
          .format(-plan.round_trip_bps))
    print("  expected net        {:+.1f} bps = ${:+,.2f}".format(
        plan.expected_net_bps, plan.expected_net_usd))
    if plan.breakeven_days:
        print("  break-even          {:.1f} days".format(plan.breakeven_days))
    print()
    print("  The funding line is the one with evidence behind it. The price leg "
          "is\n  not forecast here and is not in the number above: measured at "
          "+33 bps a\n  week on average and -10 to +72 by year, it is the "
          "variance, not the edge.")

    if plan.warnings:
        print()
        print("WARNINGS")
        for warning in plan.warnings[-6:]:
            print("  - " + warning)
        if len(plan.warnings) > 6:
            print("  ({} more, mostly per-instrument exclusions)".format(
                len(plan.warnings) - 6))

    print()
    if plan.ok:
        print("PLAN OK. Nothing has been sent, and there is no code here that "
              "could send it.")
    else:
        print("REFUSED, for all of these reasons:")
        for reason in plan.reasons:
            print("  - " + reason)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notional", type=Decimal, default=Decimal("2000"),
                        help="gross notional: half long, half short")
    parser.add_argument("--leverage", type=Decimal, default=Decimal("2"))
    parser.add_argument("--top-frac", type=float, default=0.3)
    parser.add_argument("--hold-days", type=int, default=7)
    parser.add_argument("--carry-days", type=int, default=7,
                        help="trailing window for the funding score")
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--min-funding-days", type=int, default=30)
    parser.add_argument("--max-instruments", type=int, default=80)
    args = parser.parse_args(argv)

    config = BookConfig(
        top_frac=args.top_frac, hold_days=args.hold_days,
        min_volume_usd=args.min_volume, min_funding_days=args.min_funding_days,
        gross_notional_usd=args.notional, leverage=args.leverage)

    started = time.time()
    candidates = gather(market_api(), min_volume=args.min_volume,
                        carry_days=args.carry_days,
                        min_funding_days=args.min_funding_days,
                        limit=args.max_instruments)
    if not candidates:
        raise SystemExit(
            "No instruments cleared the ${:,.0f}/24h filter on this venue.\n"
            "  Lower --min-volume, or check the venue is reachable."
            .format(args.min_volume))

    limits = RiskLimits(max_notional=max(args.notional, RiskLimits().max_notional))
    plan = plan_book(candidates, config, limits,
                     taker_fee_bps=float(TAKER_FEE_BPS))
    show(plan, config)
    print("\n({} instruments read in {:.1f}s)".format(
        len(candidates), time.time() - started))
    return 0 if plan.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
