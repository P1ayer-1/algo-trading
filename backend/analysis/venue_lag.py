"""Does BloFin's book lag Binance's price by enough to take the stale side?

    python backend\\analysis\\venue_lag.py --date 2026-09-11 --instruments ADA-USDT,DOGE-USDT
    python backend\\analysis\\venue_lag.py --date 2026-09-11 --instruments BTC-USDT --binance-latency-ms 500

The hypothesis
--------------
Binance is where price is discovered; BloFin's makers quote off it. If they
re-quote slower than a taker order can arrive, then whenever Binance has
just moved, BloFin's best ask (or bid) is briefly a price from the past, and
buying it earns the part of Binance's move that BloFin has not yet priced.
This is the one HFT idea with data on disk: the recorder's own BloFin book
archive, and Binance's public aggTrades for the same day.

What is measured, on a one-second grid over the whole day
----------------------------------------------------------
* `gap` = Binance mid (last trade, nudged half a tick toward the side it
  did not hit) minus BloFin mid, in bps of BloFin mid. Its distribution
  next to BloFin's own half spread says how often the gap even reaches the
  touch.
* The lead: BloFin's mid change over the next 1/5/30 seconds regressed on
  `gap`. A slope near +1 at short horizons is a book that is catching up; a
  slope near zero is a book that already agrees and a gap that is noise.
* The trade: whenever Binance mid exceeds BloFin's ask by more than the
  taker fee plus `--edge-bps`, buy the ask; mirror for the bid. Marked out at
  BloFin's mid (the optimistic exit) and at the far touch (the taker exit)
  after 5, 30 and 60 seconds, net of fees, one position at a time.

Clocks: BloFin events are stamped at RECEIVE time by the recorder, which is
the time an order could have been decided on. Binance trades carry exchange
time, so `--binance-latency-ms` is added to them before the join to stand in
for the feed latency a machine here would have (default 150ms, roughly the
BloFin feed's own `t - ts`, measured 2026-09-11). Lower it to flatter the
strategy, raise it to be honest about a home connection.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.replay import merged_events  # noqa: E402
from trading.orderbook import OrderBook  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "data"


def binance_symbol(instrument: str) -> str:
    return instrument.replace("-", "")


def load_binance(path: Path, latency_ms: int):
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name).decode("utf-8")
    ts, px, maker = [], [], []
    for row in csv.DictReader(io.StringIO(raw)):
        ts.append(int(row["transact_time"]))
        px.append(float(row["price"]))
        maker.append(row["is_buyer_maker"] == "true")
    ts = np.array(ts, dtype=np.int64) + latency_ms
    px = np.array(px)
    maker = np.array(maker)
    diffs = np.diff(np.unique(px))
    tick = float(diffs[diffs > 0].min()) if len(diffs) else 0.0
    # A trade with the buyer as maker printed at the bid: mid is half a tick up.
    mid = px + np.where(maker, tick / 2.0, -tick / 2.0)
    return ts, mid, tick


def load_blofin(instrument: str, date: str):
    book = OrderBook()
    ts, bid, ask = [], [], []
    for t, _, message in merged_events(DATA / instrument / "raw", date):
        if not isinstance(message, dict):
            continue
        if message.get("arg", {}).get("channel") not in ("books", "books5"):
            continue
        book.apply(message)
        if not book.ready or book.is_crossed():
            continue
        b, a = book.best_bid_ask()
        if b is None:
            continue
        ts.append(t)
        bid.append(b)
        ask.append(a)
    return np.array(ts, dtype=np.int64), np.array(bid), np.array(ask)


def asof(ts_src: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ts_src, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", required=True)
    parser.add_argument("--instruments", required=True)
    parser.add_argument("--binance-dir", type=Path, required=True,
                        help="directory holding <SYMBOL>-<date>.zip aggTrades files")
    parser.add_argument("--binance-latency-ms", type=int, default=150)
    parser.add_argument("--edge-bps", type=float, default=2.0)
    parser.add_argument("--max-age-ms", type=int, default=3000)
    parser.add_argument("--cost-bps", type=float, default=None, help="taker, per leg")
    args = parser.parse_args(argv)
    fee = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps

    for instrument in args.instruments.split(","):
        symbol = binance_symbol(instrument)
        zpath = args.binance_dir / (symbol + "-" + args.date + ".zip")
        if not zpath.exists():
            print(instrument + ": no Binance file " + str(zpath))
            continue
        bts, bmid, tick = load_binance(zpath, args.binance_latency_ms)
        fts, fbid, fask = load_blofin(instrument, args.date)
        if len(fts) < 1000:
            print(instrument + ": too few BloFin book states")
            continue
        lo = max(bts[0], fts[0])
        hi = min(bts[-1], fts[-1])
        grid = np.arange(lo, hi, 1000, dtype=np.int64)
        gm = asof(bts, bmid, grid)
        gbid = asof(fts, fbid, grid)
        gask = asof(fts, fask, grid)
        fmid = (gbid + gask) / 2.0
        # Freshness: a book state or a trade older than `--max-age-ms` is an
        # archive gap (feed drop, restart) rather than a quote anyone could
        # have hit, and a stale BloFin book against a live Binance tape reads
        # as a huge gap that never closes.
        age_f = grid - asof(fts, fts.astype(float), grid)
        age_b = grid - asof(bts, bts.astype(float), grid)
        fresh = (age_f <= args.max_age_ms) & (age_b <= args.max_age_ms)
        ok = np.isfinite(gm) & np.isfinite(fmid) & (fmid > 0) & fresh
        print("  fresh seconds (both venues updated within {}ms): {:.1%}".format(
            args.max_age_ms, np.mean(fresh)))
        gap = np.where(ok, (gm - fmid) / fmid * 1e4, np.nan)
        half = np.where(ok, (gask - gbid) / 2.0 / fmid * 1e4, np.nan)
        print("\n{} on {}: {:.1f}h, {} BloFin book states, {} Binance trades, Binance tick {:.2f} bps".format(
            instrument, args.date, (hi - lo) / 3.6e6, len(fts), len(bts), tick / np.nanmedian(gm) * 1e4))
        print("  BloFin half spread: median {:.2f} bps  p90 {:.2f}".format(
            np.nanmedian(half), np.nanpercentile(half, 90)))
        q = np.nanpercentile(gap, [1, 5, 25, 50, 75, 95, 99])
        print("  gap Binance-BloFin (bps): p1 {:+.1f} p5 {:+.1f} p25 {:+.1f} p50 {:+.1f} p75 {:+.1f} p95 {:+.1f} p99 {:+.1f}".format(*q))
        beyond = np.nanmean(np.abs(gap) > half)
        beyond_fee = np.nanmean(np.abs(gap) > half + fee + args.edge_bps)
        print("  share of seconds |gap| beyond BloFin touch {:.2%}; beyond touch + fee + edge {:.3%}".format(
            beyond, beyond_fee))
        print("  lead: slope of BloFin mid change (bps) on gap, and correlation")
        for horizon in (1, 5, 30):
            fut = np.full(len(grid), np.nan)
            fut[:-horizon] = (fmid[horizon:] - fmid[:-horizon]) / fmid[:-horizon] * 1e4
            m = np.isfinite(fut) & np.isfinite(gap)
            x, y = gap[m], fut[m]
            slope = np.polyfit(x, y, 1)[0]
            corr = np.corrcoef(x, y)[0, 1]
            # And the reverse: does BloFin's gap predict BINANCE's next move?
            bfut = np.full(len(grid), np.nan)
            bfut[:-horizon] = (gm[horizon:] - gm[:-horizon]) / gm[:-horizon] * 1e4
            mb = np.isfinite(bfut) & np.isfinite(gap)
            bslope = np.polyfit(gap[mb], bfut[mb], 1)[0]
            print("    {:>2}s  BloFin catches up: slope {:+.3f} corr {:+.3f}   Binance moves back: slope {:+.3f}".format(
                horizon, slope, corr, bslope))

        # The trade, one position at a time.
        for horizon in (5, 30, 60):
            n = 0
            pnl_mid, pnl_touch = [], []
            t = 0
            while t < len(grid) - horizon:
                if not ok[t]:
                    t += 1
                    continue
                buy = gm[t] > gask[t] * (1 + (fee + args.edge_bps) / 1e4)
                sell = gm[t] < gbid[t] * (1 - (fee + args.edge_bps) / 1e4)
                if not (buy or sell):
                    t += 1
                    continue
                if buy:
                    entry = gask[t]
                    exit_mid, exit_touch = fmid[t + horizon], gbid[t + horizon]
                    sign = 1.0
                else:
                    entry = gbid[t]
                    exit_mid, exit_touch = fmid[t + horizon], gask[t + horizon]
                    sign = -1.0
                if not (np.isfinite(exit_mid) and np.isfinite(exit_touch)):
                    t += 1
                    continue
                pnl_mid.append(sign * (exit_mid - entry) / entry * 1e4 - 2 * fee)
                pnl_touch.append(sign * (exit_touch - entry) / entry * 1e4 - 2 * fee)
                n += 1
                t += horizon
            if n:
                pm, pt = np.array(pnl_mid), np.array(pnl_touch)
                print("  trade, exit after {:>2}s: {:>5} fills/day   net at mid {:+.2f} ±{:.2f}   net at touch {:+.2f} ±{:.2f}   hit(touch) {:.0%}".format(
                    horizon, n, pm.mean(), pm.std(ddof=1) / np.sqrt(n), pt.mean(), pt.std(ddof=1) / np.sqrt(n), (pt > 0).mean()))
            else:
                print("  trade, exit after {:>2}s: no fills".format(horizon))
    return 0


if __name__ == "__main__":
    sys.exit(main())
