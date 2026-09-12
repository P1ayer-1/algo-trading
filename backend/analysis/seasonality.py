"""Hour-of-day and weekday returns of an equal-weight perp index, by year.

    python backend\\analysis\\seasonality.py

Why
---
The cheapest possible short-horizon edge is a clock: if the equal-weight
cross-section reliably moves one way in a given UTC hour or on a given
weekday, a position held for that hour needs no forecast at all. Most such
patterns are small next to a taker round trip, and the ones that are not
tend to belong to one year. So the table is per year, in bps per hour, with
the sign agreement across years printed beside the pooled mean - a pattern
that holds in five of five years is a different object from one that holds
on average.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.intraday import build_grid, log_returns  # noqa: E402


def main() -> int:
    grid = build_grid(60)
    r = log_returns(grid.close, 1)
    index = np.nanmean(np.where(grid.complete, r, np.nan), axis=1)
    hours = grid.hours()                         # hour the bar ENDS
    weekday = grid.weekday()
    years = np.array([d[:4] for d in grid.dates])
    year_list = sorted(set(years))

    print("Equal-weight index return in bps for the hour ENDING at each UTC hour, by year")
    print("{:>5} ".format("hour") + " ".join("{:>7}".format(y) for y in year_list) + "   pooled   agree")
    for hr in range(24):
        cells, signs = [], []
        for y in year_list:
            pick = (hours == hr) & (years == y) & np.isfinite(index)
            m = index[pick].mean() if pick.any() else float("nan")
            cells.append(m)
            signs.append(np.sign(m))
        pick = (hours == hr) & np.isfinite(index)
        pooled = index[pick].mean()
        se = index[pick].std(ddof=1) / np.sqrt(pick.sum())
        agree = int(np.sum(np.array(signs) == np.sign(pooled)))
        print("{:>5d} ".format(hr) + " ".join("{:>+7.1f}".format(c) for c in cells)
              + "   {:+6.1f}±{:<4.1f} {}/{}".format(pooled, se, agree, len(year_list)))

    print()
    print("By weekday (Mon=0), bps per hour, by year")
    print("{:>5} ".format("day") + " ".join("{:>7}".format(y) for y in year_list) + "   pooled   agree")
    for d in range(7):
        cells, signs = [], []
        for y in year_list:
            pick = (weekday == d) & (years == y) & np.isfinite(index)
            m = index[pick].mean() if pick.any() else float("nan")
            cells.append(m)
            signs.append(np.sign(m))
        pick = (weekday == d) & np.isfinite(index)
        pooled = index[pick].mean()
        se = index[pick].std(ddof=1) / np.sqrt(pick.sum())
        agree = int(np.sum(np.array(signs) == np.sign(pooled)))
        print("{:>5d} ".format(d) + " ".join("{:>+7.1f}".format(c) for c in cells)
              + "   {:+6.1f}±{:<4.1f} {}/{}".format(pooled, se, agree, len(year_list)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
