"""The lead-quote state machine: what to post, cancel and exit, and nothing else.

Where it came from
------------------
README step 9ae. When Binance's mid is `edge` bps past BloFin's `ask - tick`,
post a bid at `ask - tick` - alone at its level, so the next uninformed
seller on BloFin fills it at the maker fee at a price already known to be
below fair. Exit with a maker order at the Binance-implied fair, never below
entry + tick; cross out only if the leader moves `stop` bps back through the
entry, or at the hold limit. Mirror for the ask side. On three archived
days that was +0.8 to +7.4 bps a fill on SUI, DOGE, AVAX and BTC at edge 7 /
stop 3, monotone in the edge, and it died at 500 ms of feed latency.

What this file is, and is not
-----------------------------
This is the backtest's rule set (`analysis/venue_lag_passive.py`) rewritten
as an event-driven object with no clock of its own and no I/O: feed it
leader mids, follower book states and follower prints, and it returns
`Intent`s - post, cancel, exit_post, exit_cross - and records the fills it
BELIEVES happened from the production tape ("paper" fills, the same rule as
the study). It imports no broker and has no path to the order endpoint; the
runner in `execute.py` decides whether an intent becomes a demo order, and
`tests/test_lead_quote.py` greps that this file never could.

Paper fills drive the state, real ones do not, deliberately: the demo book is
not the market, so a demo order's fate says nothing about the strategy. What
a demo order measures is latency, and that is logged beside the paper fill it
mirrors so the two can be compared after the fact.

Floats throughout, like the feature engine: this is a description of a
quote, not the margin arithmetic `risk.py` guards with Decimal.

Two live refinements, off by default
------------------------------------
With `QuoteConfig` at its defaults this is exactly the backtest's rule set.
Two switches, added 2026-09-13 and turned on by `run_lead_quote.py`, are not
yet in `venue_lag_passive.py`, so a run with them on is a different strategy
from the one 9ae scored and its start row says so:

  flicker_ms   the leader must stay `edge` past the level this long before a
               post. 17 of the first 137 "leader came back" cancels came
               within 2 ms of their post: Binance quote flickers, not moves.
  trade_watch  BloFin's book arrives in 100 ms batches but its trades push per
               print. A sell print above a resting bid proves a better bid
               stood there, so the bid is no longer first and is cancelled
               without waiting for the batch; a print at or through the level
               since the last batch also withholds a post there. Exits are
               unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class QuoteConfig:
    tick: float
    edge_bps: float = 7.0
    stop_bps: float = 3.0
    order_ttl_ms: int = 10_000
    hold_ms: int = 120_000
    maker_bps: float = 0.6
    taker_bps: float = 5.0
    flicker_ms: int = 0            # flicker filter; 0 = post on the first leader tick past the edge
    trade_watch: bool = False      # let prints cancel and withhold entries before the next book batch


@dataclass(frozen=True)
class Intent:
    """One thing the runner may do. `side` is +1 for the bid side, -1 for the ask."""
    kind: str                      # post | cancel | exit_post | exit_cross
    side: int
    price: Optional[float]
    reason: str


@dataclass
class Fill:
    side: int
    t_post: int
    t_fill: int
    entry: float
    exit_price: float = float("nan")
    t_exit: int = 0
    passive_exit: bool = False
    net_bps: float = float("nan")

    @property
    def wait_ms(self) -> int:
        return self.t_fill - self.t_post

    @property
    def hold_ms(self) -> int:
        return self.t_exit - self.t_fill


@dataclass
class _Order:
    price: float
    t_post: int


@dataclass
class _Position:
    entry: float
    t_fill: int
    t_post: int
    exit_price: Optional[float] = None


@dataclass
class Quoter:
    """One instrument, two sides, one order or position per side."""

    config: QuoteConfig
    bid: Optional[float] = None
    ask: Optional[float] = None
    leader_mid: Optional[float] = None
    orders: dict = field(default_factory=lambda: {+1: None, -1: None})
    positions: dict = field(default_factory=lambda: {+1: None, -1: None})
    fills: List[Fill] = field(default_factory=list)
    posts: int = 0
    cancels: int = 0
    # Why a pair posts nothing: XRP, the biggest BloFin tape that passes the
    # screen, posted 0 times in its first 1.55 h (2026-09-13). Either the
    # leader never got `edge` past the post price (the BloFin book keeps up),
    # or it did and the level was occupied. These tell the two apart.
    gap_episodes: int = 0          # times the leader moved >= edge past the post price, either side
    gap_blocked: int = 0           # ...with size already at that level, so nothing was posted
    max_gap_bps: float = 0.0
    flickers: int = 0              # edge episodes over within flicker_ms, so never posted
    trade_cancels: int = 0         # resting entries a print showed were no longer first
    trade_blocked: int = 0         # edge episodes whose post a print withheld at least once
    wake_at: Optional[int] = None  # when a pending confirmation needs a decision with no event due
    _in_gap: dict = field(default_factory=lambda: {+1: False, -1: False})
    _gap_start: dict = field(default_factory=lambda: {+1: 0, -1: 0})
    _blocked_in_gap: dict = field(default_factory=lambda: {+1: False, -1: False})
    _print_bid: Optional[float] = None   # highest sell-aggressor print since the last book: a bid stood there
    _print_ask: Optional[float] = None   # lowest buy-aggressor print since the last book

    # ---- events ---------------------------------------------------------

    def on_leader(self, t: int, mid: float) -> List[Intent]:
        self.leader_mid = mid
        return self._decide(t)

    def on_book(self, t: int, bid: float, ask: float) -> List[Intent]:
        self.bid, self.ask = bid, ask
        self._print_bid = self._print_ask = None       # the book is authoritative again
        out: List[Intent] = []
        for side in (+1, -1):
            order = self.orders[side]
            if order is not None:
                if side == +1 and bid > order.price + 1e-12 or side == -1 and ask < order.price - 1e-12:
                    self.orders[side] = None
                    self.cancels += 1
                    out.append(Intent("cancel", side, order.price, "no longer first at level"))
                elif side == +1 and ask <= order.price + 1e-12 or side == -1 and bid >= order.price - 1e-12:
                    self._fill(side, t, order.price)
            pos = self.positions[side]
            if pos is not None and pos.exit_price is not None:
                if side == +1 and bid >= pos.exit_price - 1e-12 or side == -1 and ask <= pos.exit_price + 1e-12:
                    self._close(side, t, pos.exit_price, passive=True)
        out.extend(self._decide(t))
        return out

    def on_trade(self, t: int, price: float, taker_side: str) -> List[Intent]:
        """`taker_side` is the aggressor, as BloFin's trades channel reports it."""
        out: List[Intent] = []
        watch = self.config.trade_watch
        if watch and taker_side == "sell":
            self._print_bid = price if self._print_bid is None else max(self._print_bid, price)
        elif watch and taker_side == "buy":
            self._print_ask = price if self._print_ask is None else min(self._print_ask, price)
        for side in (+1, -1):
            order = self.orders[side]
            if order is not None:
                if side == +1 and taker_side == "sell" and price <= order.price + 1e-12:
                    self._fill(side, t, order.price)
                elif side == -1 and taker_side == "buy" and price >= order.price - 1e-12:
                    self._fill(side, t, order.price)
                elif watch and (side == +1 and taker_side == "sell" or side == -1 and taker_side == "buy"):
                    # Not a fill, so the print was beyond the order: someone rested a better price.
                    self.orders[side] = None
                    self.cancels += 1
                    self.trade_cancels += 1
                    out.append(Intent("cancel", side, order.price, "print beyond the level"))
            pos = self.positions[side]
            if pos is not None and pos.exit_price is not None and self.bid is not None:
                # Pessimistic queue: a print only fills an exit that is strictly
                # inside the spread, i.e. first at its level.
                inside = (self.ask > pos.exit_price + 1e-12) if side == +1 else (self.bid < pos.exit_price - 1e-12)
                if inside:
                    if side == +1 and taker_side == "buy" and price >= pos.exit_price - 1e-12:
                        self._close(side, t, pos.exit_price, passive=True)
                    elif side == -1 and taker_side == "sell" and price <= pos.exit_price + 1e-12:
                        self._close(side, t, pos.exit_price, passive=True)
        out.extend(self._decide(t))
        return out

    def on_clock(self, t: int) -> List[Intent]:
        return self._decide(t)

    # ---- decisions ------------------------------------------------------

    def _decide(self, t: int) -> List[Intent]:
        out: List[Intent] = []
        self.wake_at = None
        if self.bid is None or self.ask is None or self.leader_mid is None:
            return out
        cfg, mid = self.config, self.leader_mid
        for side in (+1, -1):
            order = self.orders[side]
            if order is not None:
                stale = t - order.t_post > cfg.order_ttl_ms
                reversed_ = (mid < order.price) if side == +1 else (mid > order.price)
                if stale or reversed_:
                    self.orders[side] = None
                    self.cancels += 1
                    out.append(Intent("cancel", side, order.price, "ttl" if stale else "leader came back"))
            pos = self.positions[side]
            if pos is not None:
                adverse = side * (mid - pos.entry) / pos.entry * 1e4 < -cfg.stop_bps
                if t - pos.t_fill >= cfg.hold_ms or adverse:
                    touch = self.bid if side == +1 else self.ask
                    self._close(side, t, touch, passive=False)
                    out.append(Intent("exit_cross", side, touch, "leader stop" if adverse else "hold limit"))
                elif pos.exit_price is None:
                    if side == +1:
                        price = max(pos.entry + cfg.tick, math.ceil(mid / cfg.tick - 1e-9) * cfg.tick)
                    else:
                        price = min(pos.entry - cfg.tick, math.floor(mid / cfg.tick + 1e-9) * cfg.tick)
                    pos.exit_price = price
                    out.append(Intent("exit_post", side, price, "fair"))
                continue
            if self.orders[side] is not None:
                continue
            price = self.ask - cfg.tick if side == +1 else self.bid + cfg.tick
            alone = (price > self.bid + 1e-12) if side == +1 else (price < self.ask - 1e-12)
            gap = side * (mid - price) / price * 1e4
            if gap > self.max_gap_bps:
                self.max_gap_bps = gap
            in_gap = gap >= cfg.edge_bps
            if in_gap and not self._in_gap[side]:
                self.gap_episodes += 1
                self._gap_start[side] = t
                self._blocked_in_gap[side] = False
                if not alone:
                    self.gap_blocked += 1
            elif not in_gap and self._in_gap[side] and t - self._gap_start[side] < cfg.flicker_ms:
                self.flickers += 1
            self._in_gap[side] = in_gap
            if not alone or not in_gap:
                continue
            printed = self._print_bid if side == +1 else self._print_ask
            if printed is not None and side * (printed - price) > -1e-12:
                # A print at or through the level since the last book: it is taken.
                if not self._blocked_in_gap[side]:
                    self._blocked_in_gap[side] = True
                    self.trade_blocked += 1
                continue
            if t - self._gap_start[side] < cfg.flicker_ms:
                due = self._gap_start[side] + cfg.flicker_ms
                self.wake_at = due if self.wake_at is None else min(self.wake_at, due)
                continue
            self.orders[side] = _Order(price, t)
            self.posts += 1
            out.append(Intent("post", side, price, "gap {:.1f} bps".format(gap)))
        return out

    # ---- bookkeeping ----------------------------------------------------

    def _fill(self, side: int, t: int, price: float) -> None:
        order = self.orders[side]
        self.orders[side] = None
        self.positions[side] = _Position(price, t, order.t_post)
        self.fills.append(Fill(side, order.t_post, t, price))

    def _close(self, side: int, t: int, price: float, *, passive: bool) -> None:
        pos = self.positions[side]
        self.positions[side] = None
        cfg = self.config
        fee = cfg.maker_bps + (cfg.maker_bps if passive else cfg.taker_bps)
        for fill in reversed(self.fills):
            if fill.side == side and fill.t_fill == pos.t_fill and math.isnan(fill.net_bps):
                fill.exit_price = price
                fill.t_exit = t
                fill.passive_exit = passive
                fill.net_bps = side * (price - pos.entry) / pos.entry * 1e4 - fee
                break

    # ---- reading --------------------------------------------------------

    def closed_fills(self) -> List[Fill]:
        return [f for f in self.fills if not math.isnan(f.net_bps)]
