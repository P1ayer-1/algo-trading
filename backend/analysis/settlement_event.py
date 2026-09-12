"""What price does around a funding settlement, conditioned on the rate.

    python backend\\analysis\\settlement_event.py
    python backend\\analysis\\settlement_event.py --min-abs-bps 5 --years 2024,2025,2026

The hypothesis
--------------
A perp's funding is a transfer at the settlement instant to whoever holds a
position then. If longs are about to pay a large rate, some of them close
before the instant and re-open after; if the effect is real the price dips
into a positive settlement and recovers after it, and the reverse for a
negative one. That is a trade held for an hour or two, not a week, and it is
the shortest-horizon place a funding signal could pay.

What is measured
----------------
For every (symbol, settlement) the 15-minute log returns from 4 hours before
to 4 hours after, both RAW and in EXCESS of the cross-section's mean over the
same 15 minutes (so a market-wide move into the hour cancels). Events are
bucketed by the settlement's own rate - which is known to within a few percent
an hour ahead, because it is the time-average of the premium over the interval
just ending - and the mean path per bucket is printed with a standard error on
each window. The two candidate trades are priced explicitly:

  capture   short from T-1h through T to T+1h (collect the rate, pay the path)
  rebound   long from T to T+2h (no funding, just the recovery)

both against a taker round trip, and the same for negative rates with the
signs flipped. A number here is bps per event; the trade count says how often.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.intraday import build_grid, excess, log_returns  # noqa: E402

STEP = 15
BEFORE, AFTER = 16, 16          # 4h either side in 15m steps


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-volume", type=float, default=5e6, help="median daily $ volume, trailing 30d")
    parser.add_argument("--years", help="restrict settlements to these years")
    parser.add_argument("--buckets", default="-1e9,-5,-2,0,2,5,10,1e9",
                        help="edges in bps per settlement")
    parser.add_argument("--cost-bps", type=float, default=None)
    args = parser.parse_args(argv)
    cost = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps

    grid = build_grid(STEP, fields=("close", "volume", "funding"))
    close = grid.close
    n_t, n_s = close.shape
    r15 = log_returns(close, 1)                                   # bps, ends at row t
    complete = grid.complete & np.isfinite(r15)

    # Eligibility from 30-day trailing median daily volume, computed on the
    # 8h-summed volume to keep it cheap: a symbol is in if it clears the bar.
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

    r15_excess = excess(np.where(complete, r15, np.nan), eligible & complete)

    # Events: every row with a settlement.
    settle_rows = np.flatnonzero(grid.funding_periods.max(axis=1) > 0)
    settle_rows = settle_rows[(settle_rows >= BEFORE) & (settle_rows < n_t - AFTER)]
    if args.years:
        years = set(args.years.split(","))
        settle_rows = np.array([t for t in settle_rows if grid.dates[t][:4] in years])
    offsets = np.arange(-BEFORE + 1, AFTER + 1)     # return ending at t+o

    rates, raw_paths, ex_paths, when = [], [], [], []
    for t in settle_rows:
        has = (grid.funding_periods[t] > 0) & eligible[t] & np.isfinite(grid.funding[t])
        if not has.any():
            continue
        idx = np.flatnonzero(has)
        window = slice(t - BEFORE + 1, t + AFTER + 1)
        raw = r15[window][:, idx].T
        ex = r15_excess[window][:, idx].T
        ok = np.all(np.isfinite(raw), axis=1) & np.all(np.isfinite(ex), axis=1)
        rates.append(grid.funding[t, idx[ok]])
        raw_paths.append(raw[ok])
        ex_paths.append(ex[ok])
        when.append(np.full(ok.sum(), t))
    rate = np.concatenate(rates)
    raw_path = np.concatenate(raw_paths)
    ex_path = np.concatenate(ex_paths)
    when = np.concatenate(when)
    print("{} settlement events on {} symbols, {} settlements".format(
        len(rate), n_s, len(np.unique(when))))

    def window_sum(paths, lo_h, hi_h):
        """Sum of 15m returns for (T+lo_h, T+hi_h] in hours."""
        cols = (offsets > lo_h * 4) & (offsets <= hi_h * 4)
        return paths[:, cols].sum(axis=1)

    windows = [(-4, -1), (-1, 0), (0, 1), (1, 4)]
    edges = [float(e) for e in args.buckets.split(",")]

    def se_by_event(values, events):
        """Standard error clustering by settlement instant (events share a market)."""
        groups = {}
        for v, e in zip(values, events):
            groups.setdefault(e, []).append(v)
        means = np.array([np.mean(g) for g in groups.values()])
        return means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else float("nan")

    for label, paths in (("EXCESS of cross-section", ex_path), ("RAW", raw_path)):
        print()
        print(label + " mean 15m-summed return in bps, by settlement rate bucket "
              "(bps/8h); windows are hours relative to the settlement T")
        print("{:>14} {:>7} {:>13} {:>13} {:>13} {:>13}".format(
            "rate bucket", "n", "(-4h,-1h]", "(-1h,0]", "(0,+1h]", "(+1h,+4h]"))
        for lo, hi in zip(edges[:-1], edges[1:]):
            pick = (rate >= lo) & (rate < hi)
            if pick.sum() < 50:
                continue
            cells = []
            for a, b in windows:
                v = window_sum(paths[pick], a, b)
                cells.append("{:+6.1f}±{:<4.1f}".format(v.mean(), se_by_event(v, when[pick])))
            name = "[{:g},{:g})".format(lo, hi).replace("1e+09", "inf").replace("-1e+09", "-inf")
            print("{:>14} {:>7d} {:>13} {:>13} {:>13} {:>13}".format(name, int(pick.sum()), *cells))

    # Price the two trades, per bucket, on RAW returns (each is one leg).
    print()
    print("Trades priced on raw returns, bps per event, taker round trip {:.0f}:".format(cost))
    print("  capture = -(path from T-1h to T+1h) + rate - cost     (short through the settlement)")
    print("  rebound = +(path from T to T+2h) - cost                (long after it)")
    print("  for negative-rate buckets both trades are mirrored.")
    print("{:>14} {:>7} {:>10} {:>10} {:>10} {:>10}".format(
        "rate bucket", "n", "capture", "±se", "rebound", "±se"))
    for lo, hi in zip(edges[:-1], edges[1:]):
        pick = (rate >= lo) & (rate < hi)
        if pick.sum() < 50:
            continue
        sign = 1.0 if (lo + hi) / 2 >= 0 else -1.0
        path_c = window_sum(raw_path[pick], -1, 1)
        capture = -sign * path_c + sign * rate[pick] - cost
        path_r = window_sum(raw_path[pick], 0, 2)
        rebound = sign * path_r - cost
        name = "[{:g},{:g})".format(lo, hi).replace("1e+09", "inf").replace("-1e+09", "-inf")
        print("{:>14} {:>7d} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}".format(
            name, int(pick.sum()), capture.mean(), se_by_event(capture, when[pick]),
            rebound.mean(), se_by_event(rebound, when[pick])))

    # Rank correlation of the rate with the pre- and post-settlement windows.
    from analysis.stats import spearman
    print()
    for a, b in windows:
        v = window_sum(ex_path, a, b)
        print("spearman(rate, excess return ({:+d}h,{:+d}h]) = {:+.4f}".format(
            a, b, spearman(rate, v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
