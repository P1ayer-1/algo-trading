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

    # ---- events ---------------------------------------------------------

    def on_leader(self, t: int, mid: float) -> List[Intent]:
        self.leader_mid = mid
        return self._decide(t)

    def on_book(self, t: int, bid: float, ask: float) -> List[Intent]:
        self.bid, self.ask = bid, ask
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
        for side in (+1, -1):
            order = self.orders[side]
            if order is not None:
                if side == +1 and taker_side == "sell" and price <= order.price + 1e-12:
                    self._fill(side, t, order.price)
                elif side == -1 and taker_side == "buy" and price >= order.price - 1e-12:
                    self._fill(side, t, order.price)
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
        return self._decide(t)

    def on_clock(self, t: int) -> List[Intent]:
        return self._decide(t)

    # ---- decisions ------------------------------------------------------

    def _decide(self, t: int) -> List[Intent]:
        out: List[Intent] = []
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
            if not alone:
                continue
            gap = side * (mid - price) / price * 1e4
            if gap >= cfg.edge_bps:
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
