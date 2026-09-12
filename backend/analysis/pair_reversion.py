"""Two coins that move together, one of which just did not: does the gap close?

    python backend\\analysis\\pair_reversion.py
    python backend\\analysis\\pair_reversion.py --z 3 --hold-hours 8 --window-hours 24

Where it differs from the reversal already measured
---------------------------------------------------
Step 8 and 9ab measured cross-sectional reversal as a decile ladder over the
whole universe: real, worth under a basis point. That averages every coin
against every other. A pair trade conditions on two things the ladder does
not: that the two names are known to co-move (the top trailing-30-day
correlation partner, re-chosen daily), and that their gap is EXTREME for
that pair (a z-score of the trailing `--window-hours` return difference
against its own 30-day distribution). If the ladder's basis point is the
average of a few large convergences and a lot of nothing, this is where the
large ones would be.

Trade: long the laggard, short the leader, half the gross each, from the
close of the hour the gap exceeds `--z`, held `--hold-hours`, one event per
pair per hold. Cost is two taker legs in and two out, so 2 x cost per unit
of gross. Placebo: the same pairs one day later. The 9n lesson applies: the
entry is a CLOSE, so bid-ask bounce is already paid rather than harvested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.intraday import build_grid, log_returns  # noqa: E402
from analysis.lead_lag import daily_eligibility  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--hold-hours", type=int, default=4)
    parser.add_argument("--z", type=float, default=2.5)
    parser.add_argument("--min-corr", type=float, default=0.6)
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--cost-bps", type=float, default=None, help="per leg")
    args = parser.parse_args(argv)
    cost_leg = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps
    step = 60
    w, h = args.window_hours, args.hold_hours

    grid = build_grid(step, fields=("close", "volume"))
    close = grid.close
    n_t, n_s = close.shape
    day_bars = 24
    # daily_eligibility assumes 15m bars; redo it at the hourly step.
    n_days = n_t // day_bars + 1
    daily_volume = np.full((n_days, n_s), np.nan)
    for k in range(n_days):
        block = grid.volume[k * day_bars:(k + 1) * day_bars]
        if len(block):
            daily_volume[k] = np.nansum(block, axis=0)
    elig_day = np.zeros_like(daily_volume, dtype=bool)
    for k in range(30, n_days):
        with np.errstate(invalid="ignore"):
            elig_day[k] = np.nanmedian(daily_volume[k - 30:k], axis=0) >= args.min_volume
    eligible = elig_day[np.arange(n_t) // day_bars] & grid.complete

    r1 = log_returns(close, 1)
    back = log_returns(close, w)
    fwd = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd[:-h] = np.log(close[h:] / close[:-h]) * 1e4
    years = np.array([d[:4] for d in grid.dates])

    # Daily: partner of each coin = highest trailing-30d correlation of hourly
    # returns among eligible names; sd of the w-hour return DIFFERENCE for
    # that pair from the same 30 days (non-overlapping samples).
    partner = np.full((n_days, n_s), -1, dtype=int)
    pair_sd = np.full((n_days, n_s), np.nan)
    pair_corr = np.full((n_days, n_s), np.nan)
    for k in range(30, n_days):
        lo, hi = (k - 30) * day_bars, k * day_bars
        block = r1[lo:hi]
        ok_cols = elig_day[k] & (np.isfinite(block).mean(axis=0) > 0.9)
        idx = np.flatnonzero(ok_cols)
        if len(idx) < 4:
            continue
        sub = np.nan_to_num(block[:, idx], nan=0.0)
        c = np.corrcoef(sub, rowvar=False)
        np.fill_diagonal(c, -np.inf)
        best = np.argmax(c, axis=1)
        diff = back[lo:hi:w][:, idx]
        for a, i in enumerate(idx):
            j = idx[best[a]]
            if c[a, best[a]] < args.min_corr:
                continue
            partner[k, i] = j
            d = diff[:, a] - diff[:, best[a]]
            d = d[np.isfinite(d)]
            if len(d) >= 15:
                pair_sd[k, i] = d.std()
            pair_corr[k, i] = c[a, best[a]]

    events = []
    last_exit = {}
    for t in range(30 * day_bars, n_t - h):
        k = t // day_bars
        for i in range(n_s):
            j = partner[k, i]
            if j < 0 or not eligible[t, i] or not eligible[t, j]:
                continue
            sd = pair_sd[k, i]
            if not np.isfinite(sd) or sd <= 0:
                continue
            d = back[t, i] - back[t, j]
            if not np.isfinite(d):
                continue
            z = d / sd
            if abs(z) < args.z:
                continue
            key = (min(i, j), max(i, j))
            if last_exit.get(key, -1) >= t:
                continue
            # i is the leader if z > 0: short i, long j.
            lead, lag = (i, j) if z > 0 else (j, i)
            f_lead, f_lag = fwd[t, lead], fwd[t, lag]
            if not (np.isfinite(f_lead) and np.isfinite(f_lag)):
                continue
            pf = np.nan
            if t + day_bars < n_t - h:
                a, b = fwd[t + day_bars, lead], fwd[t + day_bars, lag]
                if np.isfinite(a) and np.isfinite(b):
                    pf = 0.5 * (b - a) - 2 * cost_leg
            events.append((t, abs(z), 0.5 * (f_lag - f_lead) - 2 * cost_leg, pf, pair_corr[k, i], abs(d)))
            last_exit[key] = t + h
    if not events:
        print("no events")
        return 0
    arr = np.array(events, dtype=float)
    net, pl, zs, gaps = arr[:, 2], arr[:, 3], arr[:, 1], arr[:, 5]
    rows = arr[:, 0].astype(int)

    def cse(v, r):
        groups = {}
        for x, rr in zip(v, r):
            if np.isfinite(x):
                groups.setdefault(rr // h, []).append(x)
        means = np.array([np.mean(g) for g in groups.values()])
        return means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else float("nan")

    print("pairs by top trailing-30d hourly correlation (>= {:.2f}), gap over {}h beyond {:g} sd, hold {}h, "
          "cost {:.0f}/leg -> {:.0f} per unit of gross".format(args.min_corr, w, args.z, h, cost_leg, 2 * cost_leg))
    print("{} events; mean partner corr {:.2f}; mean gap at entry {:.0f} bps".format(
        len(net), np.nanmean(arr[:, 4]), gaps.mean()))
    print("net {:+.1f} ±{:.1f} per unit of gross   placebo {:+.1f} ±{:.1f}   hit {:.0%}   gross convergence {:+.1f} bps".format(
        net.mean(), cse(net, rows), np.nanmean(pl), cse(pl, rows), (net > 0).mean(), net.mean() + 2 * cost_leg))
    yrs = years[rows]
    for y in sorted(set(yrs)):
        pick = yrs == y
        print("  {}  n {:>5}  net {:+.1f} ±{:.1f}  placebo {:+.1f}".format(
            y, int(pick.sum()), net[pick].mean(), cse(net[pick], rows[pick]), np.nanmean(pl[pick])))
    print("  by z: ", end="")
    for lo, hi in ((args.z, args.z + 1), (args.z + 1, args.z + 2), (args.z + 2, 1e9)):
        pick = (zs >= lo) & (zs < hi)
        if pick.sum() >= 30:
            print("[{:g},{:g}) n={} net {:+.1f}  ".format(lo, hi, int(pick.sum()), net[pick].mean()), end="")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
