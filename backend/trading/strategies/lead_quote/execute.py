"""Run the lead quote live: production feeds, paper fills, demo orders only when told.

    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=0.0001), log=RunLog(path))
    asyncio.run(runner.run(minutes=60))                    # dry: feeds + paper fills + latency
    runner = LeadQuoteRunner(..., broker=BlofinQuoteBroker(...))   # demo orders mirrored

What runs where, and why
------------------------
Both feeds are PRODUCTION: Binance's public `bookTicker` stream for the
leader (no account, no key - Binance is a data source here and nothing
else) and BloFin's public books + trades for the follower. The `Quoter`
decides from those alone, and its fills are paper fills read off the
production tape by the same rule as the backtest. That is the reference
result, because demo's book is not the market and a demo order's fate says
nothing about whether the strategy works.

What a demo order DOES measure is the part the backtest assumed: how long a
post, a cancel and a cross take to be acknowledged from this host, whether
`post_only` at `ask - tick` is accepted, and whether the venue's own order
stream reports what the tape implied. So with a broker every intent is
mirrored to the demo account as a real order, its acknowledgement round
trip is logged beside the paper event it came from, and the private orders
channel is logged as it arrives. Without a broker (the default) nothing is
sent and the same log is written minus those rows.

Latency is the point. Every leader message carries Binance's event time and
every follower message BloFin's `ts`; receive time minus those, sampled once
a second, is the lag `plan_quote` gates on. The first `warmup_seconds` only
measure; quoting starts after the gate passes.

The SDK is imported lazily inside the things that need it, so this module
imports - and its state machine tests run - without it.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from .plan import QuotePlan, plan_quote
from .quoter import Intent, QuoteConfig, Quoter

BINANCE_WS = "wss://fstream.binance.com/ws/"


def now_ms() -> int:
    return int(time.time() * 1000)


class Broker(Protocol):
    """The four calls a quoter needs on the account. Thin, like the others."""

    def place_limit(self, *, inst_id: str, side: str, size: Decimal, price: str,
                    client_order_id: str, reduce_only: bool) -> Dict[str, Any]: ...

    def place_market(self, *, inst_id: str, side: str, size: Decimal,
                     client_order_id: str, reduce_only: bool) -> Dict[str, Any]: ...

    def cancel(self, *, inst_id: str, order_id: str) -> Dict[str, Any]: ...

    def positions(self) -> List[Dict[str, Any]]: ...


class BlofinQuoteBroker:
    """`post_only` limit orders and reduce-only exits against BloFin's REST API."""

    def __init__(self, client, trading_api):
        self.client = client
        self.trading = trading_api

    def place_limit(self, *, inst_id, side, size, price, client_order_id, reduce_only):
        return self.trading.placeOrder(
            instId=inst_id, marginMode="cross", positionSide="net", side=side,
            orderType="post_only", size=str(size), price=price,
            reduceOnly="true" if reduce_only else "false", clientOrderId=client_order_id)

    def place_market(self, *, inst_id, side, size, client_order_id, reduce_only):
        return self.trading.placeOrder(
            instId=inst_id, marginMode="cross", positionSide="net", side=side,
            orderType="market", size=str(size),
            reduceOnly="true" if reduce_only else "false", clientOrderId=client_order_id)

    def cancel(self, *, inst_id, order_id):
        return self.trading.cancelOrder(orderId=order_id, instId=inst_id)

    def positions(self):
        payload = self.client.get("/api/v1/account/positions", params={}, sign=True)
        return [row for row in (payload.get("data") or []) if isinstance(row, dict)]

    async def aclose(self) -> None:
        pass


class AsyncBlofinQuoteBroker(BlofinQuoteBroker):
    """The same calls on the SDK's aiohttp `AsyncClient`: each returns a coroutine.

    `LeadQuoteRunner` awaits these on the event loop instead of handing a
    blocking `requests` call to a worker thread.
    """

    is_async = True

    async def positions(self):
        payload = await self.client.get("/api/v1/account/positions", params={}, sign=True)
        return [row for row in (payload.get("data") or []) if isinstance(row, dict)]

    async def aclose(self) -> None:
        await self.client.close()


class RunLog:
    """One JSON object per line, receive-time stamped. Flushed per row: a run
    that dies mid-fill must still leave the fill on disk."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, **row: Any) -> None:
        row.setdefault("t", now_ms())
        self._handle.write(json.dumps(row, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


@dataclass
class DemoOrder:
    cid: str
    kind: str
    side: int
    order_id: str = ""
    state: str = "sent"
    filled: bool = False


@dataclass
class LeadQuoteRunner:
    inst_id: str
    config: QuoteConfig
    log: RunLog
    broker: Optional[Broker] = None
    size: Decimal = Decimal("1")
    warmup_seconds: float = 20.0
    stall_timeout_s: float = 30.0
    on_log: Callable[[str], None] = print
    private_feed: Optional[Callable[[], Any]] = None    # factory for the demo orders channel
    quoter: Quoter = field(init=False)
    plan: Optional[QuotePlan] = None
    leader_lags: List[float] = field(default_factory=list)
    follower_lags: List[float] = field(default_factory=list)
    demo_orders: Dict[str, DemoOrder] = field(default_factory=dict)
    entry_order: Dict[int, Optional[DemoOrder]] = field(default_factory=lambda: {+1: None, -1: None})
    quoting: bool = False
    demo_net: Decimal = Decimal("0")        # contracts this run's demo fills have left open, signed
    _seen_fills: int = 0
    _closed: int = 0
    _bid: float = 0.0
    _ask: float = 0.0

    def __post_init__(self) -> None:
        self.quoter = Quoter(self.config)

    @property
    def dry_run(self) -> bool:
        return self.broker is None

    # ---- feeds ------------------------------------------------------------

    async def leader_feed(self) -> None:
        import websockets                                  # noqa: WPS433 - lazy, optional
        symbol = self.inst_id.replace("-", "").lower()
        backoff = 1.0
        last_sample = 0
        while True:
            try:
                async with websockets.connect(BINANCE_WS + symbol + "@bookTicker",
                                              ping_interval=20, ping_timeout=20) as ws:
                    self.on_log("leader feed connected: Binance " + symbol + "@bookTicker")
                    backoff = 1.0
                    while True:
                        # asyncio.timeout, not wait_for: on 3.11 wait_for can swallow a
                        # cancel that lands as recv() completes. The leader feed ticks many
                        # times a second, and on 2026-09-12 it kept running after cancel
                        # and hung shutdown with seven probe orders left resting.
                        async with asyncio.timeout(self.stall_timeout_s):
                            raw = await ws.recv()
                        t = now_ms()
                        msg = json.loads(raw)
                        bid, ask = float(msg["b"]), float(msg["a"])
                        if bid <= 0 or ask <= 0:
                            continue
                        event_ms = int(msg.get("E") or 0)
                        if event_ms and t - last_sample >= 1000:
                            lag = float(t - event_ms)
                            self.leader_lags.append(lag)
                            self.log.write(event="lag", feed="leader", lag_ms=lag)
                            last_sample = t
                        if self.quoting:
                            await self.handle(self.quoter.on_leader(t, (bid + ask) / 2.0))
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # noqa: BLE001
                self.on_log("leader feed error: {}".format(exc))
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def follower_feed(self) -> None:
        from blofin.websocket_client import BlofinWsPublicClient   # noqa: WPS433 - lazy
        from ...orderbook import OrderBook
        backoff = 1.0
        last_sample = 0
        while True:
            client = BlofinWsPublicClient(isDemo=False)
            book = OrderBook()
            try:
                await client.connect()
                await client.subscribeOrderBook(self.inst_id, depth="books")
                await client.subscribeTrades(self.inst_id)
                self.on_log("follower feed connected: BloFin production " + self.inst_id)
                backoff = 1.0
                messages = client.listen().__aiter__()
                while True:
                    async with asyncio.timeout(self.stall_timeout_s):    # see leader_feed
                        message = await messages.__anext__()
                    t = now_ms()
                    if not isinstance(message, dict):
                        continue
                    channel = (message.get("arg") or {}).get("channel")
                    data = message.get("data")
                    if channel == "books":
                        book.apply(message)
                        if not book.ready:
                            self.on_log("follower book desync ({}) - resyncing".format(book.last_gap_reason))
                            break
                        if book.is_crossed():
                            continue
                        bid, ask = book.best_bid_ask()
                        if bid is None:
                            continue
                        self._bid, self._ask = bid, ask
                        row = data[0] if isinstance(data, list) and data else data
                        ts = int((row or {}).get("ts") or 0) if isinstance(row, dict) else 0
                        if ts and t - last_sample >= 1000:
                            lag = float(t - ts)
                            self.follower_lags.append(lag)
                            self.log.write(event="lag", feed="follower", lag_ms=lag)
                            last_sample = t
                        if self.quoting:
                            await self.handle(self.quoter.on_book(t, bid, ask))
                    elif channel == "trades" and self.quoting:
                        for row in data or []:
                            try:
                                price, side = float(row["price"]), str(row["side"])
                            except (KeyError, TypeError, ValueError):
                                continue
                            await self.handle(self.quoter.on_trade(t, price, side))
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # noqa: BLE001
                self.on_log("follower feed error: {}".format(exc))
            finally:
                try:
                    await client.close()
                except Exception:                          # noqa: BLE001
                    pass
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def demo_orders_feed(self) -> None:
        """The demo account's own order stream, logged as it arrives."""
        if self.private_feed is None:
            return
        backoff = 1.0
        while True:
            client = self.private_feed()
            try:
                await client.connect()
                await client.subscribeOrders(self.inst_id)
                self.on_log("demo orders channel connected")
                backoff = 1.0
                async for message in client.listen():
                    for row in (message.get("data") or []) if isinstance(message, dict) else []:
                        cid = str(row.get("clientOrderId") or "")
                        order = self.demo_orders.get(cid)
                        if order is None:
                            continue
                        order.state = str(row.get("state") or "")
                        order.order_id = str(row.get("orderId") or order.order_id)
                        if order.state == "filled" and not order.filled:
                            order.filled = True
                            try:
                                filled = Decimal(str(row.get("filledSize") or "0"))
                            except Exception:          # noqa: BLE001
                                filled = Decimal("0")
                            sign = order.side if order.kind == "post" else -order.side
                            self.demo_net += sign * filled
                        self.log.write(event="demo_order", cid=cid, kind=order.kind, side=order.side,
                                       state=order.state, filled_size=row.get("filledSize"),
                                       avg_price=row.get("averagePrice"), fee=row.get("fee"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:                       # noqa: BLE001
                self.on_log("demo orders channel error: {}".format(exc))
            finally:
                try:
                    await client.close()
                except Exception:                          # noqa: BLE001
                    pass
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def clock(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            if self.quoting:
                await self.handle(self.quoter.on_clock(now_ms()))

    # ---- intents ----------------------------------------------------------

    async def handle(self, intents: List[Intent]) -> None:
        for intent in intents:
            self.log.write(event="intent", kind=intent.kind, side=intent.side,
                           price=intent.price, reason=intent.reason, inst_id=self.inst_id)
            if self.broker is not None:
                await self.mirror(intent)
        # Paper fills the quoter recorded since the last look.
        fills = self.quoter.fills
        while self._seen_fills < len(fills):
            fill = fills[self._seen_fills]
            self._seen_fills += 1
            self.log.write(event="paper_fill", side=fill.side, entry=fill.entry, wait_ms=fill.wait_ms)
            self.on_log("paper fill: {} {} after {:.1f}s".format(
                "bid" if fill.side == 1 else "ask", fill.entry, fill.wait_ms / 1000.0))
        closed = self.quoter.closed_fills()
        while self._closed < len(closed):
            fill = closed[self._closed]
            self._closed += 1
            self.log.write(event="paper_fill_closed", side=fill.side, entry=fill.entry,
                           exit=fill.exit_price, hold_ms=fill.hold_ms,
                           passive_exit=fill.passive_exit, net_bps=fill.net_bps)
            self.on_log("paper exit: {} {:+.2f} bps ({}, held {:.1f}s)".format(
                "bid" if fill.side == 1 else "ask", fill.net_bps,
                "maker" if fill.passive_exit else "crossed", fill.hold_ms / 1000.0))

    async def mirror(self, intent: Intent) -> None:
        """One demo order per intent, acknowledged and timed. Never blocks the feeds."""
        side = intent.side
        if intent.kind == "post":
            cid = "lq" + secrets.token_hex(6)
            order = DemoOrder(cid, "post", side)
            self.demo_orders[cid] = order
            self.entry_order[side] = order
            await self._send("post", order, lambda: self.broker.place_limit(
                inst_id=self.inst_id, side="buy" if side == 1 else "sell", size=self.size,
                price=self._fmt(intent.price), client_order_id=cid, reduce_only=False))
        elif intent.kind == "cancel":
            order = self.entry_order.get(side)
            if order is not None and order.order_id and not order.filled:
                await self._send("cancel", order, lambda: self.broker.cancel(
                    inst_id=self.inst_id, order_id=order.order_id))
            self.entry_order[side] = None
        elif intent.kind in ("exit_post", "exit_cross"):
            entry = self.entry_order.get(side)
            self.entry_order[side] = None
            if entry is None or not entry.filled:
                # The tape filled the paper order; the demo book did not fill
                # the real one. Nothing to exit on demo, and the entry must go.
                if entry is not None and entry.order_id and entry.state not in ("canceled", "filled"):
                    await self._send("cancel", entry, lambda: self.broker.cancel(
                        inst_id=self.inst_id, order_id=entry.order_id))
                self.log.write(event="demo_skip", kind=intent.kind, side=side,
                               reason="demo entry not filled")
                return
            cid = "lq" + secrets.token_hex(6)
            order = DemoOrder(cid, intent.kind, side)
            self.demo_orders[cid] = order
            exit_side = "sell" if side == 1 else "buy"
            if intent.kind == "exit_post":
                await self._send("exit_post", order, lambda: self.broker.place_limit(
                    inst_id=self.inst_id, side=exit_side, size=self.size,
                    price=self._fmt(intent.price), client_order_id=cid, reduce_only=True))
            else:
                await self._send("exit_cross", order, lambda: self.broker.place_market(
                    inst_id=self.inst_id, side=exit_side, size=self.size,
                    client_order_id=cid, reduce_only=True))

    async def _send(self, kind: str, order: DemoOrder, call: Callable[[], Dict[str, Any]]) -> None:
        started = now_ms()
        try:
            response = await self._call(call)
        except Exception as exc:                            # noqa: BLE001
            self.log.write(event="demo_ack", kind=kind, cid=order.cid, ok=False, msg=str(exc),
                           rtt_ms=now_ms() - started)
            self.on_log("demo {} failed: {}".format(kind, exc))
            return
        rtt = now_ms() - started
        rows = response.get("data") if isinstance(response, dict) else None
        row = rows[0] if isinstance(rows, list) and rows else (rows if isinstance(rows, dict) else {})
        code = str((row or {}).get("code", response.get("code", "0") if isinstance(response, dict) else "0"))
        ok = code in ("0", "")
        # probe_post was missing here until 2026-09-12: a probe then cancelled only
        # when the WS order stream beat the REST ack, so the probe mostly timed posts.
        if kind in ("post", "probe_post", "exit_post", "exit_cross", "flatten") and ok:
            order.order_id = str((row or {}).get("orderId") or "")
        self.log.write(event="demo_ack", kind=kind, cid=order.cid, order_id=order.order_id,
                       ok=ok, msg=str((row or {}).get("msg", "")), rtt_ms=rtt)

    async def _call(self, call: Callable[[], Any]) -> Any:
        """An async broker's call is awaited on the loop; a blocking one runs in a thread."""
        if getattr(self.broker, "is_async", False):
            return await call()
        return await asyncio.to_thread(call)

    def _fmt(self, price: Optional[float]) -> str:
        tick = Decimal(str(self.config.tick))
        return str((Decimal(str(price)) / tick).quantize(Decimal(1)) * tick)

    # ---- lifecycle --------------------------------------------------------

    def measured_plan(self) -> QuotePlan:
        def median(values: List[float]) -> Optional[float]:
            if not values:
                return None
            ordered = sorted(values)
            return ordered[len(ordered) // 2]
        return plan_quote(self.inst_id, tick=self.config.tick, bid=self._bid, ask=self._ask,
                          leader_lag_ms=median(self.leader_lags),
                          follower_lag_ms=median(self.follower_lags))

    async def run(self, *, minutes: float, measure_only: bool = False, probe_cycles: int = 0) -> QuotePlan:
        self.log.write(event="start", inst_id=self.inst_id, dry_run=self.dry_run,
                       config=self.config.__dict__, size=str(self.size))
        tasks = [asyncio.create_task(self.leader_feed()), asyncio.create_task(self.follower_feed()),
                 asyncio.create_task(self.clock())]
        if self.broker is not None:
            tasks.append(asyncio.create_task(self.demo_orders_feed()))
        try:
            await asyncio.sleep(self.warmup_seconds)
            self.plan = self.measured_plan()
            self.log.write(event="plan", ok=self.plan.ok, reasons=self.plan.reasons,
                           warnings=self.plan.warnings, tick_bps=self.plan.tick_bps,
                           spread_bps=self.plan.spread_bps, leader_lag_ms=self.plan.leader_lag_ms,
                           follower_lag_ms=self.plan.follower_lag_ms)
            for line in plan_lines(self.plan):
                self.on_log(line)
            if probe_cycles and self.broker is not None:
                await self.probe(probe_cycles)
                return self.plan
            if measure_only or not self.plan.ok:
                return self.plan
            self.quoting = True
            self.on_log("quoting {} for {:g} minutes ({})".format(
                self.inst_id, minutes, "DRY: paper fills only" if self.dry_run else "demo orders mirrored"))
            await asyncio.sleep(minutes * 60.0)
        finally:
            self.quoting = False
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.broker is not None:
                try:
                    await self.shutdown_demo()
                finally:
                    aclose = getattr(self.broker, "aclose", None)
                    if aclose is not None:
                        await aclose()
            self.log.write(event="stop", posts=self.quoter.posts, cancels=self.quoter.cancels,
                           paper_fills=len(self.quoter.fills), closed=len(self.quoter.closed_fills()))
        return self.plan

    async def shutdown_demo(self) -> None:
        """Leave the demo account the way THIS RUN found it: cancel its resting
        orders and close what its own fills left open - and nothing else.

        The first live run (2026-09-12) read the account's positions at
        shutdown and closed a SUI short of 1,677 contracts that it had never
        opened: a leftover from earlier demo work on the same account. Demo
        money, no harm, and precisely the behaviour `strategies/__init__.py`
        exists to forbid - an executor acting on risk it did not create. So
        the only quantity closed here is `demo_net`, accumulated from this
        run's own order stream, and a position the account holds beyond that
        is reported and left alone.
        """
        for order in list(self.demo_orders.values()):
            if order.order_id and order.state not in ("filled", "canceled"):
                await self._send("cancel", order, lambda o=order: self.broker.cancel(
                    inst_id=self.inst_id, order_id=o.order_id))
        try:
            rows = await self._call(self.broker.positions)
        except Exception as exc:                            # noqa: BLE001
            self.on_log("could not read demo positions at shutdown: {}".format(exc))
            rows = []
        held = Decimal("0")
        for row in rows:
            if str(row.get("instId")) == self.inst_id:
                held += Decimal(str(row.get("positions") or "0"))
        foreign = held - self.demo_net
        if foreign != 0:
            self.on_log("demo account holds {} {} contracts this run did not open; left alone".format(
                foreign, self.inst_id))
            self.log.write(event="foreign_position", inst_id=self.inst_id, contracts=str(foreign))
        if self.demo_net == 0:
            return
        size = self.demo_net
        cid = "lq" + secrets.token_hex(6)
        order = DemoOrder(cid, "flatten", 1 if size > 0 else -1)
        self.demo_orders[cid] = order
        self.on_log("closing this run's demo position: {} contracts, reduce_only".format(size))
        await self._send("flatten", order, lambda: self.broker.place_market(
            inst_id=self.inst_id, side="sell" if size > 0 else "buy", size=abs(size),
            client_order_id=cid, reduce_only=True))
        self.demo_net = Decimal("0")

    async def probe(self, cycles: int, pause_s: float = 1.0) -> None:
        """Order round-trip latency without a fill: post far below the bid, cancel.

        This is the number the backtest could not measure and the reason to
        compare hosts: a `post_only` bid 5% under the touch cannot fill, so
        the ack and the cancel are pure venue latency from this machine.
        Needs the follower feed for a price; nothing else.
        """
        for _ in range(600):
            if self._bid > 0:
                break
            await asyncio.sleep(0.1)
        if self._bid <= 0:
            self.on_log("probe: no follower quote yet")
            return
        for _ in range(cycles):
            price = self._fmt(self._bid * 0.95)
            cid = "lq" + secrets.token_hex(6)
            order = DemoOrder(cid, "probe", 1)
            self.demo_orders[cid] = order
            await self._send("probe_post", order, lambda: self.broker.place_limit(
                inst_id=self.inst_id, side="buy", size=self.size, price=price,
                client_order_id=cid, reduce_only=False))
            if order.order_id:
                await self._send("probe_cancel", order, lambda: self.broker.cancel(
                    inst_id=self.inst_id, order_id=order.order_id))
                order.state = "canceled"
            await asyncio.sleep(pause_s)


def plan_lines(plan: QuotePlan) -> List[str]:
    out = ["plan {}: tick {:.2f} bps, spread {:.2f} bps, leader lag {}, follower lag {}".format(
        plan.inst_id, plan.tick_bps, plan.spread_bps,
        "-" if plan.leader_lag_ms is None else "{:.0f} ms".format(plan.leader_lag_ms),
        "-" if plan.follower_lag_ms is None else "{:.0f} ms".format(plan.follower_lag_ms))]
    for warning in plan.warnings:
        out.append("  warning: " + warning)
    for reason in plan.reasons:
        out.append("  REFUSED: " + reason)
    out.append("  PLAN OK" if plan.ok else "  plan refused; nothing will be quoted")
    return out
