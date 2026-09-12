"""Does an extreme short-window move across a hundred perps revert or continue?

    python backend\\analysis\\extreme_move.py
    python backend\\analysis\\extreme_move.py --window-min 60 --z 4 --hold-min 60

The hypothesis
--------------
A liquidation cascade overshoots: forced sellers take whatever is bid, the
price prints far below where voluntary sellers would have sold, and it comes
back once the forced flow stops. If so, the largest few-minute moves in a
coin's own recent history should be followed by a partial reversal, and the
size of the reversal should grow with how extreme the move was. Step 9n
showed that a variance ratio at short intervals is bid-ask bounce rather
than reversion, so this does NOT use a variance ratio: it conditions on
discrete extreme events, measures the forward return from the CLOSE after
the move (so the bounce is already paid), and subtracts the cross-section's
own move over the same forward window so a market-wide crash and rebound is
not read as a coin-specific one.

Events are a coin's `--window-min` return exceeding `--z` times its own
trailing 30-day standard deviation of returns at that window, on the DOWN
side and the UP side separately. The forward return over `--hold-min` is
reported in excess of the cross-section, by year, with a standard error
clustered by event time (a cascade hits many coins in the same minutes).
A trade fading the move pays a taker round trip; a placebo at the same
coins one day later says what an unconditional window gives.
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

STEP = 15


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window-min", type=int, default=60)
    parser.add_argument("--hold-min", type=int, default=60)
    parser.add_argument("--z", type=float, default=4.0)
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--cost-bps", type=float, default=None, help="per leg")
    parser.add_argument("--placebo-days", type=int, default=1)
    args = parser.parse_args(argv)
    cost_leg = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps

    grid = build_grid(STEP, fields=("close", "volume", "funding"))
    close = grid.close
    n_t, n_s = close.shape
    w = args.window_min // STEP
    h = args.hold_min // STEP
    back = log_returns(close, w)                    # move ending at row t
    fwd = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd[:-h] = np.log(close[h:] / close[:-h]) * 10_000.0

    day_bars = 1440 // STEP
    daily_volume = np.full((n_t // day_bars + 1, n_s), np.nan)
    for k in range(daily_volume.shape[0]):
        block = grid.volume[k * day_bars:(k + 1) * day_bars]
        if len(block):
            daily_volume[k] = np.nansum(block, axis=0)
    eligible_day = np.zeros_like(daily_volume, dtype=bool)
    for k in range(30, daily_volume.shape[0]):
        with np.errstate(invalid="ignore"):
            eligible_day[k] = np.nanmedian(daily_volume[k - 30:k], axis=0) >= args.min_volume
    eligible = eligible_day[np.arange(n_t) // day_bars]

    # Trailing 30-day sd of the w-bar return, on non-overlapping samples so the
    # sd is of the thing being thresholded. Computed once a day and held.
    sd = np.full(close.shape, np.nan)
    stride = w
    for k in range(30, daily_volume.shape[0]):
        lo, hi = (k - 30) * day_bars, k * day_bars
        sample = back[lo:hi:stride]
        with np.errstate(invalid="ignore"):
            sd[hi:hi + day_bars] = np.nanstd(sample, axis=0)

    market_fwd = np.nanmean(np.where(eligible & np.isfinite(fwd), fwd, np.nan), axis=1)
    years = np.array([d[:4] for d in grid.dates])

    def collect(mask, shift_rows=0):
        rows, cols = np.nonzero(mask)
        rows = rows + shift_rows
        keep = (rows >= 0) & (rows < n_t - h)
        rows, cols = rows[keep], cols[keep]
        f = fwd[rows, cols]
        ex = f - market_fwd[rows]
        ok = np.isfinite(f) & np.isfinite(ex) & grid.complete[rows, cols]
        return rows[ok], cols[ok], f[ok], ex[ok]

    def cluster_se(values, rows):
        groups = {}
        for v, r in zip(values, rows):
            groups.setdefault(r // (h if h else 1), []).append(v)
        means = np.array([np.mean(g) for g in groups.values()])
        return means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else float("nan")

    print("window {}m, z>{:g} of trailing 30d sd, hold {}m, cost {:.0f}/leg".format(
        args.window_min, args.z, args.hold_min, cost_leg))
    base = eligible & np.isfinite(back) & np.isfinite(sd) & (sd > 0) & grid.complete
    for side, sign in (("DOWN moves (fade = long)", -1.0), ("UP moves (fade = short)", 1.0)):
        mask = base & (sign * back > args.z * sd)
        rows, cols, f, ex = collect(mask)
        prow, pcol, pf, pex = collect(mask, shift_rows=args.placebo_days * day_bars)
        print()
        print(side + ": {} events, {} symbols, {} distinct hours".format(
            len(rows), len(np.unique(cols)), len(np.unique(rows // 4))))
        fade = -sign * f - 2 * cost_leg
        fade_ex = -sign * ex - 2 * cost_leg
        print("  fade net of round trip: raw {:+.1f} ±{:.1f}   excess {:+.1f} ±{:.1f}   "
              "placebo raw {:+.1f} ±{:.1f}".format(
                  fade.mean(), cluster_se(fade, rows), fade_ex.mean(), cluster_se(fade_ex, rows),
                  (-sign * pf - 2 * cost_leg).mean(), cluster_se(-sign * pf, prow)))
        print("  mean move at entry {:+.0f} bps; hit rate of fade {:.0%}".format(
            back[rows, cols].mean(), (fade > 0).mean()))
        print("  {:>6} {:>7} {:>10} {:>8} {:>10} {:>8}".format("year", "events", "raw fade", "se", "excess", "se"))
        for year in sorted(set(years[rows])):
            pick = years[rows] == year
            print("  {:>6} {:>7d} {:>+10.1f} {:>8.1f} {:>+10.1f} {:>8.1f}".format(
                year, int(pick.sum()), fade[pick].mean(), cluster_se(fade[pick], rows[pick]),
                fade_ex[pick].mean(), cluster_se(fade_ex[pick], rows[pick])))
        # Does the reversal scale with the size of the move?
        z = sign * back[rows, cols] / sd[rows, cols]
        print("  by z: ", end="")
        for lo, hi in ((args.z, args.z + 1), (args.z + 1, args.z + 2), (args.z + 2, args.z + 4), (args.z + 4, 1e9)):
            pick = (z >= lo) & (z < hi)
            if pick.sum() >= 30:
                print("[{:g},{:g}) n={} {:+.1f}  ".format(lo, hi, int(pick.sum()), fade_ex[pick].mean()), end="")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
