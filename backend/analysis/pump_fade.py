"""Short the top of a pump: the far tail of `extreme_move.py`, priced honestly.

    python backend\\analysis\\pump_fade.py
    python backend\\analysis\\pump_fade.py --window-min 60 --z 8 --hold-min 240 --delay-min 15

Where it came from
------------------
`extreme_move.py` (2026-09-12) reported the fade of UP moves as a wash on
average (excess -5 bps at z > 4) but its last bucket pointed the other way:
at z >= 8 the fade earned +11.6 bps excess over the next hour and +22.9 over
four, on ~3,700 events. That bucket is the pump-and-dump tail, and it was a
by-product of a tool built for the DOWN side. This prices it on its own:
finer z buckets, an entry DELAYED past the bar that printed the top (a
pump's last print is not a price anyone sold at), one event per coin per
hold so a five-bar pump is one trade rather than five, a year table, and a
concentration check, because step 9ab's settlement effect turned out to be
two memecoins in one year.

Cost is a taker round trip per leg by default; `--cost-bps 10` is the stress
for a pumping name whose spread has blown out, which is the state these
events are found in.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.intraday import build_grid, log_returns  # noqa: E402
from analysis.lead_lag import daily_eligibility, STEP  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window-min", type=int, default=60)
    parser.add_argument("--hold-min", type=int, default=240)
    parser.add_argument("--delay-min", type=int, default=15)
    parser.add_argument("--z", type=float, default=8.0)
    parser.add_argument("--min-move-bps", type=float, default=0.0,
                        help="also require the window return to exceed this")
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--cost-bps", type=float, default=None, help="per leg")
    parser.add_argument("--skip-listing-days", type=int, default=7)
    args = parser.parse_args(argv)
    cost_leg = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps
    w, h, d = args.window_min // STEP, args.hold_min // STEP, args.delay_min // STEP

    grid = build_grid(STEP, fields=("close", "volume"))
    close = grid.close
    n_t, n_s = close.shape
    day_bars = 1440 // STEP
    eligible = daily_eligibility(grid, args.min_volume) & grid.complete
    # A listing pump is a different animal; skip the first days of each name.
    for s in range(n_s):
        first = np.flatnonzero(np.isfinite(close[:, s]))
        if len(first):
            eligible[:first[0] + args.skip_listing_days * day_bars, s] = False

    back = log_returns(close, w)
    fwd = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd[:-(h + d)] = np.log(close[h + d:] / close[d:n_t - h]) * 10_000.0
    market_fwd = np.nanmean(np.where(eligible & np.isfinite(fwd), fwd, np.nan), axis=1)
    ex = fwd - market_fwd[:, None]

    sd = np.full(close.shape, np.nan)
    for k in range(30, n_t // day_bars + 1):
        lo, hi = (k - 30) * day_bars, k * day_bars
        with np.errstate(invalid="ignore"):
            sd[hi:hi + day_bars] = np.nanstd(back[lo:hi:w], axis=0)
    z = back / sd
    candidate = eligible & np.isfinite(z) & (z > args.z) & (back > args.min_move_bps) \
        & np.isfinite(ex)

    # One event per coin per hold: the first bar that qualifies opens the
    # trade and nothing in that coin qualifies again until it is closed.
    events = []
    for s in range(n_s):
        rows = np.flatnonzero(candidate[:, s])
        last_exit = -1
        for t in rows:
            if t <= last_exit:
                continue
            events.append((t, s))
            last_exit = t + h + d
    if not events:
        print("no events")
        return 0
    rows = np.array([e[0] for e in events])
    cols = np.array([e[1] for e in events])
    fade_ex = -ex[rows, cols] - 2 * cost_leg
    fade_raw = -fwd[rows, cols] - 2 * cost_leg
    years = np.array([dd[:4] for dd in grid.dates])[rows]
    zs = z[rows, cols]

    def cluster_se(values, r):
        groups = {}
        for v, rr in zip(values, r):
            groups.setdefault(rr // max(h, 1), []).append(v)
        means = np.array([np.mean(g) for g in groups.values()])
        return means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else float("nan")

    print("UP moves over {}m, z > {:g} (and > {:.0f} bps), short {}m after the bar, hold {}m, "
          "cost {:.0f}/leg".format(args.window_min, args.z, args.min_move_bps, args.delay_min,
                                   args.hold_min, cost_leg))
    print("{} events, {} coins, {} distinct hours; mean move at entry {:+.0f} bps".format(
        len(rows), len(np.unique(cols)), len(np.unique(rows // 4)), back[rows, cols].mean()))
    print("fade net: excess {:+.1f} ±{:.1f}   raw {:+.1f} ±{:.1f}   hit rate {:.0%}   median excess {:+.1f}".format(
        fade_ex.mean(), cluster_se(fade_ex, rows), fade_raw.mean(), cluster_se(fade_raw, rows),
        (fade_ex > 0).mean(), np.median(fade_ex)))
    print("  {:>6} {:>7} {:>9} {:>7} {:>9} {:>6}".format("year", "events", "excess", "se", "raw", "hit"))
    for year in sorted(set(years)):
        pick = years == year
        print("  {:>6} {:>7d} {:>+9.1f} {:>7.1f} {:>+9.1f} {:>5.0%}".format(
            year, int(pick.sum()), fade_ex[pick].mean(), cluster_se(fade_ex[pick], rows[pick]),
            fade_raw[pick].mean(), (fade_ex[pick] > 0).mean()))
    print("  by z: ", end="")
    edges = [args.z, args.z + 2, args.z + 4, args.z + 8, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        pick = (zs >= lo) & (zs < hi)
        if pick.sum() >= 20:
            print("[{:g},{:g}) n={} {:+.1f}  ".format(lo, hi, int(pick.sum()), fade_ex[pick].mean()), end="")
    print()
    # Concentration: how much of the total comes from the top names.
    total = fade_ex.sum()
    by_coin = Counter()
    for c, v in zip(cols, fade_ex):
        by_coin[grid.symbols[c]] += v
    top = by_coin.most_common(5)
    print("  top-5 coins' share of total: {:.0%}   {}".format(
        sum(v for _, v in top) / total if total else float("nan"),
        ", ".join("{} {:+.0f}".format(k, v) for k, v in top)))
    # Positive years and the worst event.
    print("  worst event {:+.0f} bps, best {:+.0f}; total events per week {:.2f}".format(
        fade_ex.min(), fade_ex.max(), len(rows) / (n_t * STEP / 60 / 24 / 7)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
