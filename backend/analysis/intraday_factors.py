"""Cross-sectional factors at an intraday bar, scored as money by `factor_panel`.

    python backend\\analysis\\intraday_factors.py --bar 480 --hold 1        # 8h bars, one-settlement hold
    python backend\\analysis\\intraday_factors.py --bar 480 --hold 3        # 8h bars, held a day
    python backend\\analysis\\intraday_factors.py --bar 60 --hold 4 --factors rev_1,rev_4

Why
---
Steps 9q-9z settled the weekly cross-section; the daily-hold sweep of 2026-09-12
found the same blend earns MORE per week rebalanced daily (carry_mom +9.3
bps/day, Sharpe 2.36 on Binance). The open question is where that stops: does
the signal keep paying at a settlement-by-settlement hold, and do the intraday
signals the repo found reversal in (step 8, 15-240 minutes, ten majors) become
tradeable across a hundred names once dispersion is that much wider?

Same harness, same rules. `run_factor` is imported unchanged, so the holds do
not overlap, the label is cross-sectionally demeaned and net of funding, the
universe is point in time, cost is charged on turnover, and the control is the
mean of shuffles plus the share the real book beat. `hold_days` there is a
count of BARS here; nothing in it assumed a bar was a day.

Sign convention is `factor_panel.build_features`'s: HIGH means "expected to
outperform", so carry is negated funding and reversal is the negated trailing
return, stated here before any result is seen.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.factor_panel import (  # noqa: E402
    Result, _rank_blend, _trailing_mean, _trailing_std, block_bootstrap,
    print_report, run_factor, tradeable)
from analysis.intraday import build_grid, log_returns  # noqa: E402

MINUTES_PER_DAY = 1440


def _ffill(values: np.ndarray) -> np.ndarray:
    """Carry the last finite value forward down each column."""
    out = values.copy()
    for s in range(out.shape[1]):
        col = out[:, s]
        idx = np.where(np.isfinite(col), np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = col[idx]
        filled[~np.isfinite(col) & (idx == 0)] = np.nan
        out[:, s] = filled
    return out


def _trailing_sum_zero(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum treating NaN as zero; NaN until `window` rows exist."""
    filled = np.nan_to_num(values, nan=0.0)
    csum = np.cumsum(filled, axis=0)
    out = np.full(values.shape, np.nan)
    out[window - 1:] = csum[window - 1:] - np.vstack(
        [np.zeros((1, values.shape[1])), csum[:-window]])
    return out


def build_bar_features(grid, bars_per_day: int) -> Dict[str, np.ndarray]:
    close = grid.close
    bar_return = log_returns(close, 1)
    features: Dict[str, np.ndarray] = {}

    # Carry at settlement resolution. `carry_last` is the single most recent
    # settlement, `carry_<n>d` the mean over n days of them. Funding rows are
    # NaN between settlements on sub-8h bars, so the trailing sums treat a
    # NaN as zero and divide by the number of settlements expected.
    settlements_per_day = 3
    last = _ffill(grid.funding)
    features["carry_last"] = -last
    for days in (1, 3, 7):
        window = days * bars_per_day
        features["carry_" + str(days) + "d"] = (
            -_trailing_sum_zero(grid.funding, window) / (days * settlements_per_day))

    # Momentum over days, reversal over bars.
    for days in (3, 7, 14, 30):
        window = days * bars_per_day
        features["mom_" + str(days) + "d"] = log_returns(close, window)
    for bars in (1, 2, 3):
        features["rev_" + str(bars) + "b"] = -log_returns(close, bars)
    features["rev_1d"] = -log_returns(close, bars_per_day)

    # Flow: taker-buy share, last bar and trailing.
    features["taker_1b"] = grid.taker
    features["taker_1d"] = _trailing_mean(grid.taker, bars_per_day)
    features["taker_7d"] = _trailing_mean(grid.taker, 7 * bars_per_day)

    # Volatility over 30 days of bars, used as the risk measure.
    vol = _trailing_std(bar_return, 30 * bars_per_day)
    features["lowvol_30d"] = -vol

    # Blends, the same 60/40 rank blend the daily harness froze.
    features["carry_mom"] = _rank_blend((features["carry_7d"], 0.6),
                                        (features["mom_14d"], 0.4))
    features["carry1_mom"] = _rank_blend((features["carry_1d"], 0.6),
                                         (features["mom_14d"], 0.4))
    features["carry_mom_rev"] = _rank_blend((features["carry_7d"], 0.5),
                                            (features["mom_14d"], 0.3),
                                            (features["rev_1b"], 0.2))
    return features


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bar", type=int, default=480, help="bar minutes, multiple of 15")
    parser.add_argument("--hold", type=int, default=1, help="hold in bars")
    parser.add_argument("--top-frac", type=float, default=0.3)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--min-volume", type=float, default=5e6, help="median DAILY dollar volume")
    parser.add_argument("--min-history-days", type=int, default=90)
    parser.add_argument("--factors")
    parser.add_argument("--control-seeds", type=int, default=20)
    parser.add_argument("--vol-scale", action="store_true")
    parser.add_argument("--start-days", type=int, default=120)
    parser.add_argument("--years", help="e.g. 2024,2025 to restrict rebalance dates")
    parser.add_argument("--detail", action="store_true", help="net by year for each factor")
    parser.add_argument("--lag", type=int, default=0, help="score with the factor from this many BARS earlier")
    args = parser.parse_args(argv)

    cost_bps = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps
    bars_per_day = MINUTES_PER_DAY // args.bar
    grid = build_grid(args.bar)
    panel = grid.to_panel()
    features = build_bar_features(grid, bars_per_day)
    names = ([n.strip() for n in args.factors.split(",")] if args.factors
             else sorted(features))
    missing = [n for n in names if n not in features]
    if missing:
        raise SystemExit("Unknown factors: " + ", ".join(missing)
                         + "\n  Available: " + ", ".join(sorted(features)))

    eligible = tradeable(panel, min_history=args.min_history_days * bars_per_day,
                         min_volume=args.min_volume / bars_per_day,
                         volume_window=30 * bars_per_day)
    if args.years:
        keep = np.array([d[:4] in args.years.split(",") for d in panel.dates])
        eligible = eligible & keep[:, None]
    start = args.start_days * bars_per_day
    risk = -features["lowvol_30d"] if args.vol_scale else None
    periods_per_year = 365.0 * bars_per_day / args.hold

    counts = eligible[start:].sum(axis=1)
    universe = ("{}-minute bars, universe {} symbols, {} .. {}; eligible per rebalance "
                "min {} median {} max {}".format(
                    args.bar, len(panel.symbols), panel.dates[start], panel.dates[-1],
                    int(counts.min()), int(np.median(counts)), int(counts.max())))

    rows = []
    results: Dict[str, Result] = {}
    for name in names:
        real = run_factor(panel, features[name], eligible, hold_days=args.hold,
                          top_frac=args.top_frac, cost_bps=cost_bps, start=start,
                          risk=risk, lag=args.lag)
        real.name = name
        results[name] = real
        controls = [run_factor(panel, features[name], eligible, hold_days=args.hold,
                               top_frac=args.top_frac, cost_bps=cost_bps, start=start,
                               shuffle_seed=seed, risk=risk, lag=args.lag)
                    for seed in range(args.control_seeds)]
        summaries = [c.summary(periods_per_year) for c in controls]
        control_nets = np.array([s.get("net_bps", np.nan) for s in summaries])
        summary = real.summary(periods_per_year)
        control = {"net_bps": float(np.nanmean(control_nets)),
                   "percentile": float(np.mean(summary["net_bps"] > control_nets))
                   if summary else float("nan")}
        interval = block_bootstrap(real.net, block=max(4, 4 * bars_per_day // args.hold))
        rows.append((name, summary, control, interval))
        print("  scored " + name, flush=True)

    print_report(rows, hold_days=args.hold, cost_bps=cost_bps, top_frac=args.top_frac,
                 universe=universe)
    print("(`hold` is in BARS of {} minutes; {:.0f} periods a year)".format(
        args.bar, periods_per_year))

    # Break-even cost: the cost per unit traded at which net would be zero.
    print()
    print("{:<14} {:>9} {:>8} {:>10} {:>12}".format(
        "factor", "gross", "turn", "cost/per", "break-even"))
    for name, summary, _, _ in rows:
        if not summary:
            continue
        gross = summary["gross_bps"]
        turnover = summary["turnover"]
        charged = turnover * cost_bps / 2.0
        breakeven = (gross / (turnover / 2.0)) if turnover > 0 else float("nan")
        print("{:<14} {:>9.1f} {:>8.2f} {:>10.1f} {:>12.1f}".format(
            name, gross, turnover, charged, breakeven))
    print("`break-even` is the all-in cost per unit of notional traded (fee + half "
          "spread + impact) at which the factor nets zero; the harness charged "
          "{:.1f}.".format(cost_bps))

    if args.detail:
        print()
        for name in names:
            legs = results[name].legs_by_year()
            print(name + "  year: periods / net / price / funding")
            for year, (n, net, price, fund) in legs.items():
                print("   {} {:5d} {:8.1f} {:8.1f} {:8.1f}".format(year, n, net, price, fund))
    return 0


if __name__ == "__main__":
    sys.exit(main())
