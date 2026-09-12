"""Deleveraging events read off open interest: does a forced flush rebound?

    python backend\\analysis\\oi_cascade.py
    python backend\\analysis\\oi_cascade.py --oi-drop-z 3 --ret-z 2 --hold 4

Why open interest and not the price move alone
----------------------------------------------
`extreme_move.py` conditioned on a coin's own price move and found a
rebound that belonged to the market, not the coin, and was over within the
bar. Price alone cannot tell a cascade (positions being closed by the
engine, which is forced flow and should overshoot) from a repricing on news
(positions being opened, which should not). Open interest can: a cascade
takes OI DOWN with price, a repricing takes it UP or leaves it. Bybit's
hourly OI history (`fetch_bybit_oi.py`) makes that distinction measurable
on ~4 years x ~80 names.

Events, stated before the run
-----------------------------
An hour in which OI fell by more than `--oi-drop-z` trailing-30-day standard
deviations of hourly OI changes AND the return was below `--ret-z` standard
deviations (long flush), or above (short squeeze). The forward return over
`--hold` hours, raw and in excess of the eligible cross-section, entered at
the close of the event hour (the honest entry: the flush's low is not a
price anyone bought) and net of a taker round trip. Placebo: the same coins
one day later. Reported by year, with the same for "OI ROSE with the move",
because the difference between those two rows is the whole hypothesis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.oi_factors import build, log_change, trailing_mean, trailing_std  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--oi-drop-z", type=float, default=3.0)
    parser.add_argument("--ret-z", type=float, default=2.0)
    parser.add_argument("--hold", type=int, default=4, help="hours")
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--cost-bps", type=float, default=None, help="per leg")
    args = parser.parse_args(argv)
    cost = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps
    h = args.hold

    ts, dates, symbols, close, volume, oi, funding = build()
    n_t, n_s = close.shape
    daily_vol = trailing_mean(volume, 24) * 24.0
    med30 = np.full(close.shape, np.nan)
    for t in range(720, n_t, 24):
        with np.errstate(invalid="ignore"):
            med30[t:t + 24] = np.nanmedian(daily_vol[t - 720:t:24], axis=0)
    eligible = np.isfinite(close) & np.isfinite(oi) & (med30 >= args.min_volume)

    ret1 = log_change(close, 1) * 1e4
    doi1 = log_change(oi, 1)
    sd_ret = trailing_std(ret1, 720)
    sd_doi = trailing_std(doi1, 720)
    fwd = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd[:-h] = np.log(close[h:] / close[:-h]) * 1e4
    market = np.nanmean(np.where(eligible & np.isfinite(fwd), fwd, np.nan), axis=1)
    ex = fwd - market[:, None]
    years = np.array([d[:4] for d in dates])

    base = eligible & np.isfinite(ret1) & np.isfinite(doi1) & np.isfinite(sd_ret) & np.isfinite(sd_doi) & np.isfinite(ex)
    with np.errstate(invalid="ignore"):
        z_ret = ret1 / sd_ret
        z_oi = doi1 / sd_doi

    def report(label, mask, sign):
        """`sign` is the direction of the position that bets on a rebound."""
        rows, cols = np.nonzero(mask)
        # One event per coin per hold.
        keep = np.ones(len(rows), dtype=bool)
        last = {}
        for i, (r, c) in enumerate(zip(rows, cols)):
            if c in last and r <= last[c]:
                keep[i] = False
            else:
                last[c] = r + h
        rows, cols = rows[keep], cols[keep]
        if len(rows) < 20:
            print("  {:<44} n {:>5}  (too few)".format(label, len(rows)))
            return
        net_ex = sign * ex[rows, cols] - 2 * cost
        net_raw = sign * fwd[rows, cols] - 2 * cost
        prow = rows + 24
        pok = prow < n_t - h
        pl = sign * ex[prow[pok], cols[pok]] - 2 * cost
        pl = pl[np.isfinite(pl)]
        groups = {}
        for v, r in zip(net_ex, rows):
            groups.setdefault(r // h, []).append(v)
        means = np.array([np.mean(g) for g in groups.values()])
        se = means.std(ddof=1) / np.sqrt(len(means))
        by_year = " ".join("{}:{:+.0f}".format(y[2:], net_ex[years[rows] == y].mean())
                           for y in sorted(set(years[rows])))
        print("  {:<44} n {:>5}  excess {:+6.1f} ±{:4.1f}  raw {:+6.1f}  placebo {:+6.1f}  hit {:.0%}  {}".format(
            label, len(rows), net_ex.mean(), se, net_raw.mean(), pl.mean(), (net_ex > 0).mean(), by_year))

    print("Bybit hourly, {} symbols; events: |return| > {:g} sd and |OI change| > {:g} sd of their "
          "trailing 30 days; hold {}h from the event close; cost {:.0f}/leg".format(
              n_s, args.ret_z, args.oi_drop_z, h, cost))
    down = base & (z_ret < -args.ret_z)
    up = base & (z_ret > args.ret_z)
    oi_down = z_oi < -args.oi_drop_z
    oi_up = z_oi > args.oi_drop_z
    print("\nlong after a drop (bet on rebound):")
    report("price down, OI DOWN  (long flush)", down & oi_down, +1.0)
    report("price down, OI UP    (new shorts)", down & oi_up, +1.0)
    report("price down, OI flat", down & ~oi_down & ~oi_up, +1.0)
    print("\nshort after a jump (bet on giveback):")
    report("price up, OI DOWN    (short squeeze)", up & oi_down, -1.0)
    report("price up, OI UP      (new longs)", up & oi_up, -1.0)
    report("price up, OI flat", up & ~oi_down & ~oi_up, -1.0)
    print("\nthe other direction, continuation:")
    report("price up, OI UP -> long", up & oi_up, +1.0)
    report("price down, OI UP -> short", down & oi_up, -1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
