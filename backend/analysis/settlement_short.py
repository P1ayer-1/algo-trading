"""Short the coins whose shorts just paid: the negative-funding settlement trade.

    python backend\\analysis\\settlement_short.py
    python backend\\analysis\\settlement_short.py --threshold -5 --hold-hours 2 --signal previous

Where it came from
------------------
`settlement_event.py` (2026-09-12) bucketed 376k settlement events by the
rate and looked 4 hours either side. Positive-rate buckets showed price
continuing in the funding's direction after the settlement - and showed the
SAME thing at placebo instants 2 and 4 hours earlier, so that is a pump
continuing, not a settlement. The one bucket that was locked to the
settlement instant was rates below -5 bps: those coins RISE in the two hours
before the settlement and FALL in the two hours after, and the placebos
point the other way. The story is a short squeeze into the instant the
shorts pay, unwinding once they have.

What this prices
----------------
At every settlement T, every eligible coin whose signal rate is below
`--threshold` is shorted at the 15m close stamped T and covered `--hold-hours`
later, paying a taker round trip. Two signals:

  settle    the rate paid AT T. Binance publishes it as the time-average of
            the premium over (T-8h, T], so an hour before T it is ~90% known;
            this is the optimistic information set.
  previous  the rate paid at T-8h, known for eight hours. Pessimistic and
            unambiguous. If the effect needs the settle-time rate, this is
            where it shows.

Returns are bps per trade, aggregated to one number per settlement (equal
weight across that settlement's trades) and then to a daily series, because
trades at one settlement share a market and the daily series is what a
Sharpe can honestly be computed on. Each year, and each symbol's share of the
total, is reported so a single coin or a single quarter cannot pass as five
years of a hundred names.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.intraday import build_grid, log_returns  # noqa: E402

STEP = 15


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--threshold", type=float, default=-5.0, help="bps per settlement, strictly below")
    parser.add_argument("--hold-hours", type=float, default=2.0)
    parser.add_argument("--entry-delay-min", type=int, default=0, help="enter this many minutes after T")
    parser.add_argument("--signal", choices=("settle", "previous"), default="settle")
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--cost-bps", type=float, default=None, help="per leg; default taker fee")
    parser.add_argument("--placebo-hours", type=float, default=0.0,
                        help="shift the entry by this many hours (negative = before T)")
    parser.add_argument("--top-symbols", type=int, default=8)
    args = parser.parse_args(argv)
    cost_leg = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps
    round_trip = 2 * cost_leg

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
    eligible = eligible_day[np.arange(n_t) // day_bars]

    hold_bars = int(round(args.hold_hours * 60 / STEP))
    delay_bars = args.entry_delay_min // STEP
    shift_bars = int(round(args.placebo_hours * 60 / STEP))
    settle_bars = 8 * 60 // STEP

    complete = grid.complete                         # a property: compute it once
    settle_rows = np.flatnonzero(grid.funding_periods.max(axis=1) > 0)
    per_settlement = []          # (row, mean bps net, n trades)
    per_symbol = defaultdict(list)
    for t in settle_rows:
        signal_row = t if args.signal == "settle" else t - settle_bars
        if signal_row < 0:
            continue
        rate = grid.funding[signal_row]
        has = (grid.funding_periods[signal_row] > 0) & np.isfinite(rate) & eligible[t]
        pick = has & (rate < args.threshold)
        if not pick.any():
            continue
        entry = t + delay_bars + shift_bars
        exit_row = entry + hold_bars
        if entry < 1 or exit_row >= n_t:
            continue
        idx = np.flatnonzero(pick)
        with np.errstate(divide="ignore", invalid="ignore"):
            move = np.log(close[exit_row, idx] / close[entry, idx]) * 10_000.0
        ok = np.isfinite(move) & complete[entry, idx] & complete[exit_row, idx]
        if not ok.any():
            continue
        pnl = -move[ok] - round_trip                    # short: profit when price falls
        per_settlement.append((t, float(pnl.mean()), int(ok.sum())))
        for s, v in zip(idx[ok], pnl):
            per_symbol[grid.symbols[s]].append(v)

    if not per_settlement:
        print("no trades")
        return 0
    rows = np.array([p[0] for p in per_settlement])
    nets = np.array([p[1] for p in per_settlement])
    counts = np.array([p[2] for p in per_settlement])
    total_trades = int(counts.sum())
    stamps = grid.dates                                # built once: 175k strings
    dates = np.array([stamps[r][:10] for r in rows])
    years = np.array([d[:4] for d in dates])

    print("signal={} threshold<{:g} hold={:g}h delay={}m placebo={:+g}h cost {:.0f}/leg".format(
        args.signal, args.threshold, args.hold_hours, args.entry_delay_min,
        args.placebo_hours, cost_leg))
    print("{} trades at {} settlements over {} days; {:.2f} trades per settlement with a trade".format(
        total_trades, len(nets), len(np.unique(dates)), total_trades / len(nets)))

    # Daily series: mean over that day's settlements, zero on days without one.
    all_days = sorted(set(d[:10] for d in stamps[settle_rows[0]:]))
    day_net = {}
    for d, v in zip(dates, nets):
        day_net.setdefault(d, []).append(v)
    daily = np.array([np.mean(day_net[d]) if d in day_net else 0.0 for d in all_days])
    active = np.array([d in day_net for d in all_days])
    sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(365) if daily.std() > 0 else 0.0
    print()
    print("per trade (pooled)          {:+8.1f} bps   sd {:6.0f}   hit {:.0%}".format(
        np.concatenate(list(per_symbol.values())).mean(),
        np.concatenate(list(per_symbol.values())).std(),
        (np.concatenate(list(per_symbol.values())) > 0).mean()))
    print("per settlement (equal wt)   {:+8.1f} bps   se {:6.1f}   t {:5.2f}".format(
        nets.mean(), nets.std(ddof=1) / np.sqrt(len(nets)),
        nets.mean() / (nets.std(ddof=1) / np.sqrt(len(nets)))))
    print("daily series, active {:.0%} of days: mean {:+.1f} bps/day, Sharpe {:.2f}, worst day {:+.0f}, "
          "worst settlement {:+.0f}".format(active.mean(), daily.mean(), sharpe, daily.min(), nets.min()))

    print()
    print("{:>6} {:>7} {:>8} {:>9} {:>7} {:>9}".format("year", "trades", "settles", "mean bps", "se", "hit"))
    for year in sorted(set(years)):
        pick = years == year
        v = nets[pick]
        print("{:>6} {:>7d} {:>8d} {:>+9.1f} {:>7.1f} {:>9.0%}".format(
            year, int(counts[pick].sum()), int(pick.sum()), v.mean(),
            v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan"), (v > 0).mean()))

    print()
    share = Counter({s: len(v) for s, v in per_symbol.items()})
    print("symbols traded: {}; top by trade count (trades, mean bps, share of total P&L)".format(len(share)))
    total_pnl = sum(sum(v) for v in per_symbol.values())
    for s, n in share.most_common(args.top_symbols):
        v = np.array(per_symbol[s])
        print("  {:<14} {:>6d} {:>+8.1f} {:>6.0%}".format(s, n, v.mean(), v.sum() / total_pnl if total_pnl else 0))
    positive = sum(1 for v in per_symbol.values() if np.mean(v) > 0)
    print("symbols with positive mean: {}/{}".format(positive, len(per_symbol)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
