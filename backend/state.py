"""The single shared, in-memory snapshot of "what the chart should show".

LiveChartState holds the current candles/price/levels and is safe to read
and update from multiple asyncio tasks (REST refresh loop, ticker poll loop,
websocket stream loop) thanks to its internal lock. `broadcast()` pushes the
current snapshot out to every connected browser tab.

Nothing here is persisted to disk — restart the process and history resets
to whatever the next REST fetch returns.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from config import (
    BAR,
    CANDLE_LIMIT,
    INST_ID,
    RESISTANCE_DEFAULT_ABOVE,
    RESISTANCE_LEVEL_COUNT,
    SUPPORT_DEFAULT_BELOW,
    SUPPORT_LEVEL_COUNT,
)
from market_data import decimal_from
from support_resistance import find_support_levels, find_resistance_levels


def get_range(current_price: Decimal) -> "tuple[Decimal, Decimal]":
    """The price window (low, high) to search for support/resistance in,
    centered on the current price unless overridden via env vars."""
    configured_low = os.getenv("BLOFIN_SUPPORT_LOW")
    configured_high = os.getenv("BLOFIN_RESISTANCE_HIGH") or os.getenv("BLOFIN_SUPPORT_HIGH")
    low_range = decimal_from(configured_low, str(current_price - SUPPORT_DEFAULT_BELOW))
    high_range = decimal_from(configured_high, str(current_price + RESISTANCE_DEFAULT_ABOVE))
    if low_range > high_range:
        low_range, high_range = high_range, low_range
    return low_range, high_range


def public_candle(candle: Dict[str, Any]) -> Dict[str, float]:
    """Convert a Decimal-based internal candle to plain floats for JSON."""
    return {
        "ts": int(candle["ts"]),
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
    }


def public_level(level: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "level": float(level["level"]),
        "touches": int(level["touches"]),
        "lastTs": int(level["last_ts"]),
    }


@dataclass
class LiveChartState:
    candles: List[Dict[str, Any]] = field(default_factory=list)
    current_price: Optional[Decimal] = None
    tick_size: Decimal = Decimal("0.1")
    supports: List[Dict[str, Any]] = field(default_factory=list)
    resistances: List[Dict[str, Any]] = field(default_factory=list)
    clients: Set[Any] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Latest microstructure features + feed health, published by the trading
    # stack. Plain JSON-ready dicts so the browser can render them directly.
    # None until the microstructure feed produces its first valid snapshot,
    # which lets the UI distinguish "not running" from "running, all zeros".
    features: Optional[Dict[str, Any]] = None
    feed_status: Optional[Dict[str, Any]] = None

    async def snapshot(self) -> Dict[str, Any]:
        async with self.lock:
            return {
                "type": "snapshot",
                "instId": INST_ID,
                "bar": BAR,
                "currentPrice": float(self.current_price) if self.current_price is not None else None,
                "candles": [public_candle(candle) for candle in self.candles],
                "supports": [public_level(level) for level in self.supports[:SUPPORT_LEVEL_COUNT]],
                "resistances": [public_level(level) for level in self.resistances[:RESISTANCE_LEVEL_COUNT]],
                "features": self.features,
                "feedStatus": self.feed_status,
                "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

    async def set_features(
        self, features: Dict[str, Any], feed_status: Dict[str, Any]
    ) -> None:
        async with self.lock:
            self.features = features
            self.feed_status = feed_status

    async def set_candles(self, candles: List[Dict[str, Any]]) -> None:
        async with self.lock:
            self.candles = sorted(candles, key=lambda item: item["ts"])
            if self.candles and self.current_price is None:
                self.current_price = self.candles[-1]["close"]
            elif self.candles and self.current_price is not None:
                candle = self.candles[-1]
                candle["close"] = self.current_price
                candle["high"] = max(candle["high"], self.current_price)
                candle["low"] = min(candle["low"], self.current_price)
            self._recalculate_locked()

    async def upsert_candle(self, candle: Dict[str, Any]) -> None:
        async with self.lock:
            by_ts = {item["ts"]: item for item in self.candles}
            by_ts[candle["ts"]] = candle
            self.candles = sorted(by_ts.values(), key=lambda item: item["ts"])[-int(CANDLE_LIMIT):]
            self.current_price = candle["close"]
            self._recalculate_locked()

    async def set_price(self, price: Decimal) -> bool:
        async with self.lock:
            changed = self.current_price != price
            self.current_price = price
            if self.candles:
                candle = self.candles[-1]
                candle["close"] = price
                candle["high"] = max(candle["high"], price)
                candle["low"] = min(candle["low"], price)
            return changed

    async def recalculate(self) -> None:
        async with self.lock:
            self._recalculate_locked()

    def _recalculate_locked(self) -> None:
        if self.current_price is None:
            return
        low_range, high_range = get_range(self.current_price)
        self.supports = find_support_levels(
            self.candles,
            low_range=low_range,
            high_range=self.current_price,
            tick_size=self.tick_size,
        )
        self.resistances = find_resistance_levels(
            self.candles,
            low_range=self.current_price,
            high_range=high_range,
            tick_size=self.tick_size,
        )


async def broadcast(state: LiveChartState) -> None:
    """Send the current snapshot to every connected websocket client,
    dropping any that have gone stale (closed/broken connections)."""
    message = json.dumps(await state.snapshot())
    stale_clients = []
    for client in list(state.clients):
        try:
            await client.send(message)
        except Exception:
            stale_clients.append(client)
    for client in stale_clients:
        state.clients.discard(client)
