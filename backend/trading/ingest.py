"""The live microstructure feed: book + trades + funding -> features -> disk.

This runs as its own websocket connection, separate from the chart's existing
ticker/candle stream. That separation is deliberate: the chart is a tool you
want to keep working, and the microstructure feed is the one that will be
churning, resyncing, and getting restarted while the strategy is developed.
Neither should be able to take the other down.

Recovery behaviour: if the order book detects a sequence gap it marks itself
stale, and this loop tears the connection down and reconnects to force a fresh
snapshot. Reconnecting is cheap; trading on a silently-wrong book is not.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from blofin.websocket_client import BlofinWsPublicClient

from .features import FeatureEngine, FeatureSnapshot
from .orderbook import OrderBook
from .rawlog import RawEventLog
from .recorder import FeatureRecorder, LabelConfig
from .tape import TradeTape


class MicrostructureFeed:
    """Owns the book, the tape, the feature engine and the recorder.

    `latest` always holds the most recent computed snapshot, so any consumer
    (the chart, a future signal engine) can read current features without
    subscribing to anything.
    """

    def __init__(
        self,
        inst_id: str,
        *,
        use_demo: bool = False,
        book_depth: str = "books",
        data_dir: Optional[Path] = None,
        record: bool = True,
        sample_interval_ms: int = 250,
        label_config: Optional[LabelConfig] = None,
        tape_window_seconds: float = 60.0,
        record_raw: bool = True,
        on_snapshot: Optional[Callable[[FeatureSnapshot], Any]] = None,
    ):
        self.inst_id = inst_id
        self.use_demo = use_demo
        self.book_depth = book_depth
        self.on_snapshot = on_snapshot

        self.book = OrderBook()
        self.tape = TradeTape(window_seconds=tape_window_seconds)
        self.engine = FeatureEngine()
        self.recorder = FeatureRecorder(
            data_dir or Path("data"),
            label_config=label_config,
            sample_interval_ms=sample_interval_ms,
            enabled=record,
        )
        # The raw archive. Written before any parsing, so a bug in the feature
        # code can never corrupt or lose the source data — the archive can
        # always be replayed once the bug is fixed.
        self.raw_log = RawEventLog(data_dir or Path("data"), enabled=record_raw)

        self.latest: FeatureSnapshot = FeatureSnapshot()
        self.connected = False
        self.reconnects = 0
        self.messages = 0
        self.started_at = time.time()

        # A single crossed update can just be a mid-update race, so tolerate a
        # few in a row before forcing an expensive resubscribe.
        self.crossed_events = 0
        self.crossed_tolerance = 3

    # ---- message handling ------------------------------------------------

    def handle_message(self, message: Dict[str, Any]) -> bool:
        """Route one websocket message. Returns True if a resync is needed."""
        channel = message.get("arg", {}).get("channel", "")
        data = message.get("data")
        if data is None:
            return False

        self.messages += 1
        # Archive first, parse second. If anything below throws, the event is
        # already safely on disk and can be replayed after the fix.
        self.raw_log.write(channel, message)

        if channel in ("books", "books5"):
            self.book.apply(message)
            if not self.book.ready:
                return True  # sequence gap — caller must reconnect
            # A crossed book (bid >= ask) means our mirror has drifted from the
            # exchange's: we have kept a level that was actually removed, or
            # missed one. Features computed from it are invalid, and unlike a
            # sequence gap nothing will repair it on its own — the stale level
            # simply sits there. So treat a persistently crossed book as a
            # desync and force a fresh snapshot.
            if self.book.is_crossed():
                self.crossed_events += 1
                if self.crossed_events >= self.crossed_tolerance:
                    self.book.mark_stale(
                        f"book crossed for {self.crossed_events} consecutive updates"
                    )
                    self.crossed_events = 0
                    return True
                return False
            self.crossed_events = 0
            self.engine.on_book_event(self.book)
            self._emit()
        elif channel == "trades":
            if self.tape.add_message(data):
                self._emit()
        elif channel == "funding-rate":
            self.engine.on_funding(data)

        return False

    def _emit(self) -> None:
        snapshot = self.engine.compute(self.book, self.tape)
        self.latest = snapshot
        self.recorder.observe(snapshot)
        if self.on_snapshot is not None:
            try:
                self.on_snapshot(snapshot)
            except Exception as exc:
                print(f"Feature consumer error: {exc}")

    # ---- the loop --------------------------------------------------------

    async def run(self) -> None:
        """Connect, subscribe, and stream until cancelled. Reconnects forever."""
        backoff = 1.0
        while True:
            client = BlofinWsPublicClient(isDemo=self.use_demo)
            try:
                await client.connect()
                await client.subscribeOrderBook(self.inst_id, depth=self.book_depth)
                await client.subscribeTrades(self.inst_id)
                await client.subscribeFundingRate(self.inst_id)

                self.connected = True
                backoff = 1.0
                mode = "demo" if self.use_demo else "production"
                print(
                    f"Microstructure feed connected ({mode}): "
                    f"{self.inst_id} [{self.book_depth}, trades, funding-rate]"
                )

                async for message in client.listen():
                    if self.handle_message(message):
                        print(
                            "Order book desync "
                            f"({self.book.last_gap_reason}) — resyncing."
                        )
                        break

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"Microstructure feed error: {exc}")
            finally:
                self.connected = False
                self.book.reset()
                self.reconnects += 1
                try:
                    await client.close()
                except Exception:
                    pass

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def close(self) -> None:
        self.recorder.close()
        self.raw_log.close()

    # ---- reporting -------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        bid, ask = self.book.best_bid_ask()
        return {
            "connected": self.connected,
            "instId": self.inst_id,
            "bookReady": self.book.is_ready,
            "bookLevels": {"bids": len(self.book.bids), "asks": len(self.book.asks)},
            "bestBid": bid,
            "bestAsk": ask,
            "seqId": self.book.seq_id,
            "resyncs": self.book.resync_count,
            "reconnects": self.reconnects,
            "messages": self.messages,
            "uptimeSeconds": round(time.time() - self.started_at, 1),
            "tape": self.tape.snapshot(),
            "recorder": self.recorder.stats(),
            "rawLog": self.raw_log.stats(),
        }
