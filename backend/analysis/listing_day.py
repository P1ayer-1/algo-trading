"""The first hours after a Binance perpetual lists: is there a path to trade?

    python backend\\analysis\\listing_day.py
    python backend\\analysis\\listing_day.py --entry-hours 2 --hold-hours 6

Why
---
A new perpetual is the one intraday event where the crowd's position is
known in advance: nobody is short yet, leverage arrives all at once, and the
folklore is a pump into the listing and a bleed afterwards. The 15-minute
panel carries every bar from each symbol's first Binance-futures candle, so
the path of the first two days can be read for every listing since 2021.

What it measures, event by event
--------------------------------
For each symbol whose first bar falls inside the sample (not on its first
day), the return from `--entry-hours` after the first bar to `--hold-hours`
later, raw and in excess of the cross-section, plus the whole first-48h path
in 4-hour steps as a mean and a median. The count is the honest N: ~70
listings in five years, so this is a table to read rather than a t-stat to
trust.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.intraday import build_grid  # noqa: E402

STEP = 15


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--entry-hours", type=float, default=1.0)
    parser.add_argument("--hold-hours", type=float, default=8.0)
    parser.add_argument("--cost-bps", type=float, default=None, help="per leg")
    args = parser.parse_args(argv)
    cost_leg = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps

    grid = build_grid(STEP, fields=("close", "volume"))
    close = grid.close
    n_t, n_s = close.shape
    day_bars = 1440 // STEP
    market = np.nanmean(np.log(close[1:] / close[:-1]), axis=1) * 10_000.0
    market = np.concatenate([[np.nan], market])
    market_cum = np.nancumsum(market)

    e = int(args.entry_hours * 60 // STEP)
    h = int(args.hold_hours * 60 // STEP)
    steps = list(range(0, 48 * 4 + 1, 16))                 # every 4 hours to 48h
    paths, events = [], []
    for s, symbol in enumerate(grid.symbols):
        first = np.flatnonzero(np.isfinite(close[:, s]))
        if not len(first) or first[0] < day_bars:
            continue                                        # listed before the sample
        t0 = first[0]
        if t0 + 48 * 4 >= n_t:
            continue
        p0 = close[t0, s]
        path = [np.log(close[t0 + k, s] / p0) * 10_000.0 for k in steps]
        paths.append(path)
        entry, exit_ = t0 + e, t0 + e + h
        raw = np.log(close[exit_, s] / close[entry, s]) * 10_000.0
        ex = raw - (market_cum[exit_] - market_cum[entry])
        vol = float(np.nansum(grid.volume[t0:t0 + day_bars, s]))
        events.append((grid.dates[t0], symbol, raw, ex, vol, path[-1]))

    if not events:
        print("no listings inside the sample")
        return 0
    paths = np.array(paths)
    print("{} listings with a first bar inside the sample".format(len(events)))
    print("\nmean / median path from the first 15m close, bps, every 4 hours:")
    print("  " + " ".join("{:>7}".format(str(k * STEP // 60) + "h") for k in steps))
    print("  " + " ".join("{:>+7.0f}".format(v) for v in np.nanmean(paths, axis=0)) + "   mean")
    print("  " + " ".join("{:>+7.0f}".format(v) for v in np.nanmedian(paths, axis=0)) + "   median")
    raw = np.array([ev[2] for ev in events])
    ex = np.array([ev[3] for ev in events])
    ok = np.isfinite(raw) & np.isfinite(ex)
    raw, ex = raw[ok], ex[ok]
    se = lambda v: v.std(ddof=1) / np.sqrt(len(v))       # noqa: E731
    print("\nshort from {:g}h after listing, cover {:g}h later, cost {:.0f}/leg:".format(
        args.entry_hours, args.hold_hours, cost_leg))
    print("  net raw {:+.0f} ±{:.0f}   net excess {:+.0f} ±{:.0f}   hit {:.0%}   n {}".format(
        -raw.mean() - 2 * cost_leg, se(raw), -ex.mean() - 2 * cost_leg, se(ex),
        (-ex - 2 * cost_leg > 0).mean(), len(raw)))
    years = np.array([ev[0][:4] for ev in events])[ok]
    for year in sorted(set(years)):
        pick = years == year
        print("  {}  n {:>2}  short net excess {:+.0f} ±{:.0f}".format(
            year, int(pick.sum()), -ex[pick].mean() - 2 * cost_leg, se(ex[pick]) if pick.sum() > 1 else float("nan")))
    print("\n  {:<16} {:<14} {:>8} {:>8} {:>10} {:>8}".format("first bar", "symbol", "raw", "excess", "day-1 $vol", "48h"))
    for ev in sorted(events, key=lambda x: x[0]):
        print("  {:<16} {:<14} {:>+8.0f} {:>+8.0f} {:>10.1e} {:>+8.0f}".format(*ev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
