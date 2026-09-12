"""Three staggered 3-day tranches, and the book's return by market regime.

    python backend\\analysis\\tranche_book.py --hold-days 3 --factor carry_mom --cost-bps 10 --lag 1

Two questions step 9aa left open
--------------------------------
**Does staggering help?** A 3-day book has three rebalance phases, and the
harness scores one at a time (+22.4 / +17.6 / +23.0 on Binance). A live book
can run all three as tranches, a third of the notional each, rebalancing one
tranche a day. The mean is the average of the phases by construction; what
staggering can change is the variance, and by how much depends on how
correlated the phases are - which is measured here rather than assumed.
Each tranche's period return is booked on its exit day, the three daily
series are summed, and the result is scored per week beside the single-phase
book at the same gross.

**Where does the book lose?** The price leg is the larger half at short
holds, and momentum is known to crash when the market turns. Period returns
are split by the trailing 30-day return and trailing 30-day volatility of the
equal-weight cross-section at entry, in terciles, and by the market's own
move DURING the period (which is not knowable at entry and is printed as a
diagnostic, not a filter). A regime filter that improves the table would be
an in-sample result and is not applied here; the table says whether one is
worth pre-registering.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.factor_panel import (  # noqa: E402
    DEFAULT_PANEL, Result, _log_return, _trailing_std, block_bootstrap,
    build_features, load_panel, run_factor, tradeable)


def daily_series(result: Result, n_dates: int) -> np.ndarray:
    """Period net booked on the exit day, zero elsewhere."""
    out = np.zeros(n_dates)
    for period in result.periods:
        out[period.exit_index] += period.net_bps
    return out


def weekly(series: np.ndarray, start: int) -> np.ndarray:
    body = series[start:]
    n = len(body) // 7
    return body[:n * 7].reshape(n, 7).sum(axis=1)


def describe(label: str, weeks: np.ndarray) -> None:
    mean = weeks.mean()
    sd = weeks.std(ddof=1)
    sharpe = mean / sd * np.sqrt(52) if sd > 0 else 0.0
    lo, hi = block_bootstrap(weeks, block=4)
    cumulative = np.cumsum(weeks)
    drawdown = float(np.max(np.maximum.accumulate(cumulative) - cumulative))
    print("{:<34} {:>+8.1f} [{:+.1f}, {:+.1f}]  Sharpe {:.2f}  sd {:.0f}  worst week {:+.0f}  "
          "drawdown {:.0f}".format(label, mean, lo, hi, sharpe, sd, weeks.min(), drawdown))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--factor", default="carry_mom")
    parser.add_argument("--hold-days", type=int, default=3)
    parser.add_argument("--top-frac", type=float, default=0.3)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--band", type=float, default=0.0)
    args = parser.parse_args(argv)

    panel = load_panel(args.panel)
    features = build_features(panel)
    eligible = tradeable(panel, min_history=90, min_volume=args.min_volume)
    risk = -features["lowvol_30"]
    n_dates = panel.shape[0]
    scores = features[args.factor]

    phases = []
    for phase in range(args.hold_days):
        result = run_factor(panel, scores, eligible, hold_days=args.hold_days,
                            top_frac=args.top_frac, cost_bps=args.cost_bps,
                            start=args.start + phase, risk=risk, lag=args.lag, band=args.band)
        phases.append(result)

    print("{} at a {}-day hold, cost {:.0f}, lag {}, band {:g}; bps per WEEK on gross notional".format(
        args.factor, args.hold_days, args.cost_bps, args.lag, args.band))
    print("{:<34} {:>8} {:>16}  {:>11}  {:>6}  {:>15}  {:>13}".format(
        "book", "mean", "95% block", "", "", "", ""))
    first_week = args.start + args.hold_days
    dailies = [daily_series(r, n_dates) for r in phases]
    weeks = [weekly(d, first_week) for d in dailies]
    for phase, w in enumerate(weeks):
        describe("phase {} alone".format(phase), w)
    stacked = sum(dailies) / len(dailies)
    stacked_weeks = weekly(stacked, first_week)
    describe("three tranches, a third each", stacked_weeks)
    corr = np.corrcoef(np.vstack(weeks))
    off = corr[np.triu_indices(len(weeks), k=1)]
    print("correlation between phases' weekly returns: {}".format(
        ", ".join("{:+.2f}".format(c) for c in off)))

    # Regime split, on the single-phase book (phase 0), by conditions at entry.
    close = panel.close
    with np.errstate(invalid="ignore"):
        market_daily = np.nanmean(np.where(eligible, _log_return(close, 1), np.nan), axis=1)
    market_30 = np.convolve(np.nan_to_num(market_daily), np.ones(30), mode="full")[:n_dates]
    vol_30 = np.full(n_dates, np.nan)
    for d in range(30, n_dates):
        vol_30[d] = np.nanstd(market_daily[d - 29:d + 1])
    result = phases[0]
    nets = np.array([p.net_bps for p in result.periods])
    prices = np.array([p.price_bps for p in result.periods])
    entries = np.array([p.entry for p in result.periods])
    during = np.array([p.market_bps for p in result.periods])

    def terciles(values: np.ndarray, label: str) -> None:
        ok = np.isfinite(values)
        edges = np.nanpercentile(values[ok], [100 / 3, 200 / 3])
        print()
        print("{:<28} {:>8} {:>10} {:>10} {:>10} {:>6}".format(label, "periods", "net", "price leg", "se", "hit"))
        for name, pick in (("low", values <= edges[0]), ("mid", (values > edges[0]) & (values <= edges[1])),
                           ("high", values > edges[1])):
            pick = pick & ok
            if pick.sum() < 5:
                continue
            v = nets[pick]
            print("{:<28} {:>8d} {:>+10.1f} {:>+10.1f} {:>10.1f} {:>6.0%}".format(
                name, int(pick.sum()), v.mean(), prices[pick].mean(),
                v.std(ddof=1) / np.sqrt(len(v)), (v > 0).mean()))

    terciles(market_30[entries], "trailing 30d market return")
    terciles(vol_30[entries], "trailing 30d market vol")
    terciles(during, "market move DURING the hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
