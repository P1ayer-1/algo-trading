"""The same book on two venues at once: does stacking BloFin and Hyperliquid pay?

    python backend\\analysis\\venue_stack.py --hold-days 3 --lag 1

Why
---
This account can trade BloFin, and Hyperliquid needs only an address. Each
venue's 3-day carry/momentum book is measured separately (step 9aa); a book
run on both is one portfolio, and whether it is better than either alone
depends on how correlated the two are - they rank overlapping coins on
different funding, different universes and different marks. This aligns the
two period series by date, sums them at half the gross each, and scores the
stack beside its halves. Binance is included as the reference and is not
tradeable from here.

Costs are each venue's own: BloFin fee plus half its measured spread,
Hyperliquid and Binance flat 5 bps (the default; pass `--cost-bps`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.factor_panel import (  # noqa: E402
    REPO_ROOT, block_bootstrap, build_features, load_panel, load_spreads,
    run_factor, tradeable)

PANELS = {
    "binance": (REPO_ROOT / "data" / "panel" / "daily.csv", 5e6, None),
    "blofin": (REPO_ROOT / "data" / "panel" / "blofin-daily.csv", 2e6,
               REPO_ROOT / "data" / "panel" / "blofin-spreads.csv"),
    "hyperliquid": (REPO_ROOT / "data" / "panel" / "hyperliquid-daily.csv", 5e6, None),
}


def _day_number(date: str) -> int:
    from datetime import date as _date
    return _date.fromisoformat(date).toordinal()


def aligned_start(dates, anchor: str, hold_days: int, minimum: int) -> int:
    """First index at or after `minimum` whose date sits on the anchor's
    `hold_days` grid, so every venue rebalances on the same calendar days."""
    base = _day_number(anchor)
    for i in range(minimum, len(dates)):
        if _day_number(dates[i]) >= base and (_day_number(dates[i]) - base) % hold_days == 0:
            return i
    raise SystemExit("no aligned start for anchor " + anchor)


def book_by_exit_date(name: str, *, factor: str, hold_days: int, top_frac: float,
                      cost_bps: float, lag: int, start: int,
                      anchor: Optional[str] = None, band: float = 0.0,
                      min_volume: Optional[float] = None) -> Dict[str, float]:
    path, default_volume, spreads = PANELS[name]
    min_volume = default_volume if min_volume is None else min_volume
    panel = load_panel(path)
    if anchor is not None:
        start = aligned_start(panel.dates, anchor, hold_days, start)
    features = build_features(panel)
    eligible = tradeable(panel, min_history=90, min_volume=min_volume)
    cost_per_symbol = None
    if spreads is not None:
        cost_per_symbol = load_spreads(spreads, panel.symbols, taker_fee_bps=cost_bps,
                                       default_spread_bps=10.0)
    result = run_factor(panel, features[factor], eligible, hold_days=hold_days,
                        top_frac=top_frac, cost_bps=cost_bps, start=start,
                        risk=-features["lowvol_30"], cost_per_symbol=cost_per_symbol,
                        lag=lag, band=band)
    return {panel.dates[p.exit_index]: p.net_bps for p in result.periods}


def describe(label: str, values: np.ndarray, periods_per_year: float) -> None:
    mean = values.mean()
    sd = values.std(ddof=1)
    sharpe = mean / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
    lo, hi = block_bootstrap(values, block=4)
    cumulative = np.cumsum(values)
    drawdown = float(np.max(np.maximum.accumulate(cumulative) - cumulative))
    print("{:<28} {:>5d} {:>+8.1f} [{:+.1f}, {:+.1f}]  Sharpe {:.2f}  sd {:.0f}  worst {:+.0f}  drawdown {:.0f}".format(
        label, len(values), mean, lo, hi, sharpe, sd, values.min(), drawdown))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--factor", default="carry_mom")
    parser.add_argument("--hold-days", type=int, default=3)
    parser.add_argument("--top-frac", type=float, default=0.3)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--venues", default="blofin,hyperliquid,binance")
    parser.add_argument("--band", type=float, default=0.0)
    parser.add_argument("--blofin-volume", type=float, default=2e6)
    parser.add_argument("--blofin-top", type=float, default=None, help="top fraction for BloFin only")
    args = parser.parse_args(argv)
    cost_bps = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps

    venues = args.venues.split(",")
    # One calendar anchor for every venue: the latest of their warm-up dates.
    anchor = max(load_panel(PANELS[v][0]).dates[args.start] for v in venues)
    books = {v: book_by_exit_date(
                 v, factor=args.factor, hold_days=args.hold_days,
                 top_frac=(args.blofin_top if v == "blofin" and args.blofin_top else args.top_frac),
                 cost_bps=cost_bps, lag=args.lag, start=args.start, anchor=anchor,
                 band=args.band, min_volume=(args.blofin_volume if v == "blofin" else None))
             for v in venues}
    # Align on the dates where every venue has a period. The phases line up
    # because every panel closes at 00:00 UTC and the harness steps from the
    # same `start`; dates missing on one venue are dropped from all.
    common = sorted(set.intersection(*(set(b) for b in books.values())))
    if len(common) < 20:
        raise SystemExit("only {} common rebalance dates; the phases do not align".format(len(common)))
    periods_per_year = 365.0 / args.hold_days
    print("{} at a {}-day hold, lag {}, cost {:.0f} (+ half spread on BloFin); bps per period on gross, "
          "{} common periods {} .. {}".format(args.factor, args.hold_days, args.lag, cost_bps,
                                             len(common), common[0], common[-1]))
    series = {v: np.array([books[v][d] for d in common]) for v in venues}
    for v in venues:
        describe(v + " alone", series[v], periods_per_year)
    tradeable_venues = [v for v in venues if v != "binance"]
    if len(tradeable_venues) >= 2:
        stack = sum(series[v] for v in tradeable_venues) / len(tradeable_venues)
        describe("+".join(tradeable_venues) + ", equal gross", stack, periods_per_year)
    names = list(series)
    corr = np.corrcoef(np.vstack([series[v] for v in names]))
    print("correlations: " + ", ".join(
        "{}/{} {:+.2f}".format(names[i], names[j], corr[i, j])
        for i in range(len(names)) for j in range(i + 1, len(names))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
