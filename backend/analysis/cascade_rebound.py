"""Buy the market after a liquidation cascade: the basket version of the fade.

    python backend\\analysis\\cascade_rebound.py
    python backend\\analysis\\cascade_rebound.py --z 4 --hold-min 120 --delay-min 15

Where it came from
------------------
`extreme_move.py` (2026-09-12) found that a coin which just fell more than
six of its own hourly standard deviations rebounds +73 bps net over the next
hour, in every one of six years - and that its EXCESS over the cross-section
is zero. The rebound is not the coin's; it is the market's, because the
events are cascades that hit dozens of names in the same minutes. So the
trade is a basket, and the event is the index.

What this prices
----------------
The equal-weight index of eligible perps is formed from 15m closes. An event
is an index return over `--window-min` below `-z` times the index's own
trailing 30-day standard deviation at that window. The position is long the
equal-weight basket (or BTC alone, with `--btc`, as the instrument that is
cheapest to actually buy) from the close `--delay-min` after the event bar
to `--hold-min` later, paying a taker round trip. Events inside an open
position are skipped, so the count is of trades that could have been placed.

Delay matters more here than anywhere else in this repo: a cascade's last
print is a price nobody can buy at, and the honest entry is a bar or two
later. The result is reported by year, with a placebo at the same hours one
day later, and against the unconditional index return at the same hold.
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
    parser.add_argument("--delay-min", type=int, default=0)
    parser.add_argument("--z", type=float, default=4.0)
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--cost-bps", type=float, default=None, help="per leg")
    parser.add_argument("--btc", action="store_true", help="trade BTC instead of the basket")
    parser.add_argument("--side", choices=("down", "up"), default="down")
    args = parser.parse_args(argv)
    cost_leg = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps

    grid = build_grid(STEP, fields=("close", "volume", "funding"))
    close = grid.close
    n_t, n_s = close.shape
    r15 = log_returns(close, 1)
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
    eligible = eligible_day[np.arange(n_t) // day_bars] & grid.complete & np.isfinite(r15)

    # Equal-weight index of 15m log returns, and its cumulative level.
    with np.errstate(invalid="ignore"):
        index_r = np.nanmean(np.where(eligible, r15, np.nan), axis=1)
    index_r = np.nan_to_num(index_r, nan=0.0)
    level = np.cumsum(index_r)
    w, h, d = args.window_min // STEP, args.hold_min // STEP, args.delay_min // STEP
    back = np.full(n_t, np.nan)
    back[w:] = level[w:] - level[:-w]

    sd = np.full(n_t, np.nan)
    for k in range(30, daily_volume.shape[0]):
        lo, hi = (k - 30) * day_bars, k * day_bars
        sample = back[lo:hi:w]
        sample = sample[np.isfinite(sample)]
        if len(sample) > 20:
            sd[hi:hi + day_bars] = sample.std()

    sign = -1.0 if args.side == "down" else 1.0
    events = np.flatnonzero(np.isfinite(back) & np.isfinite(sd) & (sign * back > args.z * sd))

    # Instrument return over the hold, from entry row to exit row.
    if args.btc:
        i = grid.symbols.index("BTCUSDT")
        def hold_return(entry):
            with np.errstate(divide="ignore", invalid="ignore"):
                return float(np.log(close[entry + h, i] / close[entry, i]) * 10_000.0)
    else:
        def hold_return(entry):
            return float(level[entry + h] - level[entry])

    trades, busy_until = [], -1
    for t in events:
        entry = t + d
        exit_row = entry + h
        if entry <= busy_until or exit_row >= n_t:
            continue
        pnl = sign * -1.0 * hold_return(entry) - 2 * cost_leg     # fade: long after a fall
        if np.isfinite(pnl):
            trades.append((entry, pnl, sign * back[t] / sd[t]))
            busy_until = exit_row
    if not trades:
        print("no events")
        return 0
    rows = np.array([tr[0] for tr in trades])
    pnl = np.array([tr[1] for tr in trades])
    zs = np.array([tr[2] for tr in trades])
    stamps = grid.dates
    years = np.array([stamps[r][:4] for r in rows])

    # Placebo: same clock one day later; unconditional: every non-overlapping hold.
    placebo = np.array([sign * -1.0 * hold_return(r + day_bars) - 2 * cost_leg
                        for r in rows if r + day_bars + h < n_t])
    uncond = np.array([sign * -1.0 * hold_return(r) for r in range(0, n_t - h, h)])

    instrument = "BTC" if args.btc else "equal-weight basket"
    print("index {}m move beyond {:g} sd, {} {} {}m after the bar, hold {}m, cost {:.0f}/leg".format(
        args.window_min, args.z, "long" if args.side == "down" else "short", instrument,
        args.delay_min, args.hold_min, cost_leg))
    se = pnl.std(ddof=1) / np.sqrt(len(pnl))
    print("{} trades: mean {:+.1f} bps net  se {:.1f}  t {:.2f}  hit {:.0%}  worst {:+.0f}  best {:+.0f}".format(
        len(pnl), pnl.mean(), se, pnl.mean() / se, (pnl > 0).mean(), pnl.min(), pnl.max()))
    print("placebo one day later: {:+.1f} ±{:.1f};  unconditional {}m hold: {:+.1f}".format(
        placebo.mean(), placebo.std(ddof=1) / np.sqrt(len(placebo)), args.hold_min, uncond.mean()))
    print("mean z at entry {:.1f}; mean index move {:+.0f} bps".format(
        zs.mean(), (sign * zs * np.array([sd[r - d] for r in rows])).mean()))
    print("{:>6} {:>7} {:>10} {:>7} {:>6} {:>9}".format("year", "trades", "mean bps", "se", "hit", "sum bps"))
    for y in sorted(set(years)):
        pick = years == y
        v = pnl[pick]
        print("{:>6} {:>7d} {:>+10.1f} {:>7.1f} {:>6.0%} {:>+9.0f}".format(
            y, int(pick.sum()), v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan"),
            (v > 0).mean(), v.sum()))
    print("by z:", end="")
    for lo, hi in ((args.z, args.z + 1), (args.z + 1, args.z + 2), (args.z + 2, args.z + 4), (args.z + 4, 99)):
        pick = (zs >= lo) & (zs < hi)
        if pick.sum() >= 10:
            print("  [{:g},{:g}) n={} {:+.1f}".format(lo, hi, int(pick.sum()), pnl[pick].mean()), end="")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
