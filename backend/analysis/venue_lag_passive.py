"""Post at BloFin's stale price when Binance has already moved: the maker version of `venue_lag.py`.

    python backend\\analysis\\venue_lag_passive.py --date 2026-09-11 --instruments SUI-USDT --binance-dir <dir>
    python backend\\analysis\\venue_lag_passive.py --date 2026-09-11 --instruments SUI-USDT,DOGE-USDT --edge-bps 5 --binance-dir <dir>

Where it came from
------------------
`venue_lag.py` (step 9ad) found the lead is real - BloFin's mid follows a
Binance-BloFin gap with slope 0.79 inside 30 seconds on SUI - and that a
TAKER who lifts the stale ask nets exactly nothing, because the fee is the
edge. The taker pays 5 bps to cross a spread the stale side is about to
re-price anyway. The alternative is to be the stale side's counterparty
from inside the book: when Binance says fair value is above BloFin's ask,
post a bid one tick under that ask - a price nobody on BloFin is quoting
yet, so the order is alone and first at its level - and wait for a seller
who has not seen the Binance print. That fill costs the maker fee, and it is
struck at a price already known to be below fair. Step 9ac showed a touch
quote on BloFin fills 3-11% of the time and is run over when it does; the
question here is whether a quote posted only when the leader has already
moved is filled by the flow it wants and not the flow it fears.

What is simulated, event by event on the archive
------------------------------------------------
The merged BloFin book and trade stream for the day, with Binance aggTrades
(plus `--binance-latency-ms`) as the leader's mid. The up side: whenever
Binance mid is at least `--edge-bps` above `ask - tick`, and `ask - tick` is
above the best bid (so the order is alone at its level), post a bid there.
The order is filled by the first BloFin SELL-aggressor print at or below its
price, or by the ask moving down through it; it is cancelled if the best bid
moves above it (it is no longer first), if Binance falls back below it, or
after `--order-ttl-s` seconds. A fill is marked out at BloFin's mid after
5/30/60 s and priced with a taker exit at the far touch after
`--hold-s` seconds; one order or position at a time per side. The down side
mirrors it. `--unconditional` posts the same orders on a fixed clock with no
Binance signal, as the control: that is the plain passive quote of 9ac, and
the difference between the two is the value of the lead.

Fees come from `config` (the account's tier): maker to get in, taker to get
out in the conservative row, maker both ways in the optimistic one.

What it found (2026-09-12, three days, README step 9ae)
--------------------------------------------------------
The signalled fill marks out +2 to +3.5 bps at mid after 30 s against -2.1
to -2.6 for the clock control, on twice the fill rate. The exit decides the
sign: a taker exit loses, a one-tick maker exit gives the drift away, a
maker exit at the Binance-implied fair (`--exit-at fair`) keeps it on the
fills that go right, and a leader stop (`--stop-bps`) cuts the ones that go
wrong. At `--edge-bps 7 --stop-bps 3` it is positive in 12 of 12
instrument-days on SUI, DOGE, AVAX and BTC (+0.8 to +7.4 a fill), rising
monotonically with the edge. It fails on ADA (a 4.8 bps tick) and at 500 ms
of feed latency on the alts. Size is not modelled; every print at the
order's price fills it in full.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.replay import merged_events  # noqa: E402
from analysis.venue_lag import binance_symbol, load_binance  # noqa: E402
from trading.orderbook import OrderBook  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "data"


def load_stream(instrument: str, date: str) -> Tuple[List[tuple], float]:
    """Chronological (t, kind, ...) tuples: ('book', bid, ask) and ('trade', price, size, side)."""
    book = OrderBook()
    out: List[tuple] = []
    prices = set()
    for t, _, message in merged_events(DATA / instrument / "raw", date):
        if not isinstance(message, dict):
            continue
        channel = message.get("arg", {}).get("channel")
        if channel in ("books", "books5"):
            book.apply(message)
            if not book.ready or book.is_crossed():
                continue
            b, a = book.best_bid_ask()
            if b is None:
                continue
            out.append((t, "book", b, a))
            if len(prices) < 5000:
                prices.add(b)
                prices.add(a)
        elif channel == "trades":
            for row in message.get("data") or []:
                try:
                    out.append((t, "trade", float(row["price"]), float(row["size"]), row["side"]))
                except (KeyError, ValueError, TypeError):
                    continue
    diffs = np.diff(np.array(sorted(prices)))
    tick = float(diffs[diffs > 0].min()) if len(diffs) else 0.0
    return out, tick


def simulate(stream, tick, bts, bmid, *, edge_bps, ttl_ms, hold_ms, maker, taker,
             unconditional_every_ms=0, passive_exit=False, queue="pessimistic", exit_at="tick",
             stop_bps=0.0):
    """Per-fill records: (side, wait_ms, mark5, mark30, mark60, net, passive_exit_flag, hold_ms).

    The exit order sits at max(ask - tick, entry + tick) for a long. Inside
    the spread it is first at its level and any opposing print at its price
    fills it. When that price IS the best ask it joins a queue whose position
    is unobservable, so it is bracketed as `passive_sim.py` does: `optimistic`
    fills on any print at the price, `pessimistic` only when the far touch
    moves through it.
    """
    fills = []
    posts = {+1: 0, -1: 0}
    triggers = {+1: 0, -1: 0}
    order = {+1: None, -1: None}            # (price, t_post)
    position = {+1: None, -1: None}         # [entry, t_fill, t_post, exit_price or None, joined]
    bid = ask = None
    j = 0
    n_b = len(bts)
    last_clock = None
    mids: List[Tuple[int, float]] = []
    pending = []

    def close(side, t, exit_price, passive):
        entry, t_fill, t_post, _, _ = position[side]
        fee = maker + (maker if passive else taker)
        net = side * (exit_price - entry) / entry * 1e4 - fee
        fills.append([side, t_fill - t_post, entry, t_fill, net, 1.0 if passive else 0.0, t - t_fill])
        position[side] = None

    for ev in stream:
        t = ev[0]
        while j < n_b and bts[j] <= t:
            j += 1
        b_mid = bmid[j - 1] if j > 0 else None
        if ev[1] == "book":
            bid, ask = ev[2], ev[3]
            mids.append((t, (bid + ask) / 2.0))
            for side in (+1, -1):
                o = order[side]
                if o is not None:
                    price, t_post = o
                    if side == +1 and bid > price + 1e-12:
                        order[side] = None
                    elif side == -1 and ask < price - 1e-12:
                        order[side] = None
                    elif side == +1 and ask <= price + 1e-12:
                        pending.append((side, t_post, t, price)); order[side] = None
                    elif side == -1 and bid >= price - 1e-12:
                        pending.append((side, t_post, t, price)); order[side] = None
                p = position[side]
                if p is not None and p[3] is not None:
                    xp = p[3]
                    if side == +1 and bid >= xp - 1e-12:
                        close(side, t, xp, True)        # the far touch came through the price
                    elif side == -1 and ask <= xp + 1e-12:
                        close(side, t, xp, True)
                    elif exit_at == "tick" and side == +1 and ask < xp - 1e-12:
                        p[3] = None                     # the book moved away: re-post
                    elif exit_at == "tick" and side == -1 and bid > xp + 1e-12:
                        p[3] = None
        elif ev[1] == "trade":
            price, size, taker_side = ev[2], ev[3], ev[4]
            for side in (+1, -1):
                o = order[side]
                if o is not None:
                    oprice, t_post = o
                    if side == +1 and taker_side == "sell" and price <= oprice + 1e-12:
                        pending.append((side, t_post, t, oprice)); order[side] = None
                    elif side == -1 and taker_side == "buy" and price >= oprice - 1e-12:
                        pending.append((side, t_post, t, oprice)); order[side] = None
                p = position[side]
                if p is not None and p[3] is not None and bid is not None:
                    xp = p[3]
                    inside = (ask > xp + 1e-12) if side == +1 else (bid < xp - 1e-12)
                    if inside or queue == "optimistic":
                        if side == +1 and taker_side == "buy" and price >= xp - 1e-12:
                            close(side, t, xp, True)
                        elif side == -1 and taker_side == "sell" and price <= xp + 1e-12:
                            close(side, t, xp, True)
        if bid is None or b_mid is None:
            continue
        for side in (+1, -1):
            o = order[side]
            if o is None:
                continue
            price, t_post = o
            if t - t_post > ttl_ms or (side == +1 and b_mid < price) or (side == -1 and b_mid > price):
                order[side] = None
        clock_ok = True
        if unconditional_every_ms:
            clock_ok = last_clock is None or t - last_clock >= unconditional_every_ms
            if clock_ok:
                last_clock = t
        for side in (+1, -1):
            if order[side] is not None or position[side] is not None:
                continue
            price = ask - tick if side == +1 else bid + tick
            alone = (price > bid + 1e-12) if side == +1 else (price < ask - 1e-12)
            if not alone:
                continue
            if unconditional_every_ms:
                signal = clock_ok
            else:
                gap = (b_mid - price) / price * 1e4 * side
                signal = gap >= edge_bps
                if signal:
                    triggers[side] += 1
            if signal:
                order[side] = (price, t)
                posts[side] += 1
        while pending:
            side, t_post, t_fill, entry = pending.pop()
            position[side] = [entry, t_fill, t_post, None, False]
        for side in (+1, -1):
            p = position[side]
            if p is None:
                continue
            entry, t_fill, t_post, xp, _ = p
            stopped = stop_bps > 0 and side * (b_mid - entry) / entry * 1e4 < -stop_bps
            if t - t_fill >= hold_ms or stopped:
                # The leader has gone through the entry: the fill was the informed
                # kind after all, and waiting only lets the far touch move further.
                close(side, t, bid if side == +1 else ask, False)
                continue
            if passive_exit and xp is None:
                if exit_at == "fair":
                    # Sell at what the leader says the coin is worth, never below entry + tick.
                    if side == +1:
                        price = max(entry + tick, np.ceil(b_mid / tick - 1e-9) * tick)
                    else:
                        price = min(entry - tick, np.floor(b_mid / tick + 1e-9) * tick)
                    p[3], p[4] = price, False
                elif side == +1:
                    price = max(ask - tick, entry + tick)
                    if price <= ask + 1e-12:
                        p[3], p[4] = price, price >= ask - 1e-12
                else:
                    price = min(bid + tick, entry - tick)
                    if price >= bid - 1e-12:
                        p[3], p[4] = price, price <= bid + 1e-12
    mt = np.array([m[0] for m in mids], dtype=np.int64)
    mm = np.array([m[1] for m in mids])
    out = []
    for side, wait, entry, t_fill, net, passive, held in fills:
        marks = []
        for h in (5000, 30000, 60000):
            k = np.searchsorted(mt, t_fill + h, side="left")
            marks.append(mm[k] if k < len(mt) else np.nan)
        mark_bps = [side * (m - entry) / entry * 1e4 - maker for m in marks]
        out.append((side, wait, *mark_bps, net, passive, held))
    return out, posts, triggers


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", required=True)
    parser.add_argument("--instruments", required=True)
    parser.add_argument("--binance-dir", type=Path, required=True)
    parser.add_argument("--binance-latency-ms", type=int, default=150)
    parser.add_argument("--edge-bps", type=float, default=3.0,
                        help="Binance mid must be this far past the posted price")
    parser.add_argument("--order-ttl-s", type=float, default=10.0)
    parser.add_argument("--hold-s", type=float, default=30.0)
    parser.add_argument("--unconditional", type=float, default=0.0,
                        help="control: post every N seconds with no signal")
    parser.add_argument("--passive-exit", action="store_true",
                        help="exit with a maker order inside the spread; cross only at --hold-s")
    parser.add_argument("--queue", choices=("pessimistic", "optimistic"), default="pessimistic",
                        help="fill assumption for an exit that has to join the far touch")
    parser.add_argument("--exit-at", choices=("tick", "fair"), default="fair",
                        help="tick: one tick past entry or inside the spread; fair: the Binance mid, rounded")
    parser.add_argument("--stop-bps", type=float, default=0.0,
                        help="cross out at once if Binance mid moves this far through the entry")
    parser.add_argument("--maker-bps", type=float, default=None)
    parser.add_argument("--taker-bps", type=float, default=None)
    args = parser.parse_args(argv)
    maker = float(config.MAKER_FEE_BPS) if args.maker_bps is None else args.maker_bps
    taker = float(config.TAKER_FEE_BPS) if args.taker_bps is None else args.taker_bps

    for instrument in args.instruments.split(","):
        zpath = args.binance_dir / (binance_symbol(instrument) + "-" + args.date + ".zip")
        if not zpath.exists():
            print(instrument + ": no Binance file " + str(zpath))
            continue
        bts, bmid, _ = load_binance(zpath, args.binance_latency_ms)
        stream, tick = load_stream(instrument, args.date)
        hours = (stream[-1][0] - stream[0][0]) / 3.6e6 if stream else 0.0
        fills, posts, triggers = simulate(
            stream, tick, bts, bmid, edge_bps=args.edge_bps, ttl_ms=int(args.order_ttl_s * 1000),
            hold_ms=int(args.hold_s * 1000), maker=maker, taker=taker,
            unconditional_every_ms=int(args.unconditional * 1000), passive_exit=args.passive_exit,
            queue=args.queue, exit_at=args.exit_at, stop_bps=args.stop_bps)
        n_trades = sum(1 for e in stream if e[1] == "trade")
        label = "unconditional every {:g}s".format(args.unconditional) if args.unconditional else \
            "edge {:g} bps, latency {}ms".format(args.edge_bps, args.binance_latency_ms)
        if args.passive_exit:
            label += ", maker exit at {} ({} queue), stop {:g}".format(args.exit_at, args.queue, args.stop_bps)
        print("\n{} {}: {:.1f}h, tick {:.2f} bps, {} BloFin trades; {}; fees maker {:.1f} / taker {:.1f}".format(
            instrument, args.date, hours, tick / np.median(bmid) * 1e4, n_trades, label, maker, taker))
        print("  posted {} bids / {} asks; filled {} / {}".format(
            posts[+1], posts[-1], sum(1 for f in fills if f[0] == 1), sum(1 for f in fills if f[0] == -1)))
        if not fills:
            print("  no fills")
            continue
        arr = np.array([f[1:] for f in fills], dtype=float)
        n = len(arr)
        per_day = n / max(hours, 1e-9) * 24
        se = lambda v: np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())  # noqa: E731
        print("  fill rate {:.1%}, median wait {:.1f}s, {:.0f} fills/day".format(
            n / max(posts[+1] + posts[-1], 1), np.median(arr[:, 0]) / 1000.0, per_day))
        print("  markout at mid net of maker fee:  5s {:+.2f} ±{:.2f}   30s {:+.2f} ±{:.2f}   60s {:+.2f} ±{:.2f}".format(
            np.nanmean(arr[:, 1]), se(arr[:, 1]), np.nanmean(arr[:, 2]), se(arr[:, 2]),
            np.nanmean(arr[:, 3]), se(arr[:, 3])))
        if args.passive_exit:
            passive = arr[:, 5] > 0.5
            print("  passive exit: {:.0%} of fills filled the exit inside {:g}s, median hold {:.1f}s; net {:+.2f} ±{:.2f} "
                  "(passive {:+.2f}, forced {:+.2f})   hit {:.0%}   -> {:+.0f} bps/day".format(
                      passive.mean(), args.hold_s, np.median(arr[:, 6]) / 1000.0, arr[:, 4].mean(), se(arr[:, 4]),
                      arr[passive, 4].mean() if passive.any() else float("nan"),
                      arr[~passive, 4].mean() if (~passive).any() else float("nan"),
                      (arr[:, 4] > 0).mean(), arr[:, 4].mean() * per_day))
        else:
            print("  net with taker exit at the touch after {:g}s: {:+.2f} ±{:.2f}   hit {:.0%}   -> {:+.0f} bps/day".format(
                args.hold_s, arr[:, 4].mean(), se(arr[:, 4]), (arr[:, 4] > 0).mean(), arr[:, 4].mean() * per_day))
        for side, name in ((1, "bids"), (-1, "asks")):
            pick = np.array([f[0] == side for f in fills])
            if pick.sum() > 5:
                print("    {}: n {}  30s mid {:+.2f}  net {:+.2f}".format(
                    name, int(pick.sum()), np.nanmean(arr[pick, 2]), arr[pick, 4].mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
