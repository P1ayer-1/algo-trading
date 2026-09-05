import argparse
import asyncio
import builtins
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

SDK_SRC = Path(__file__).resolve().parent / "blofin-sdk-python" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

try:
    import websockets
    from blofin.client import DemoClient
    from blofin.rest_market import MarketAPI
    from blofin.websocket_client import BlofinWsPublicClient
except ModuleNotFoundError as exc:
    missing = exc.name or "a dependency"
    raise SystemExit(
        f"Missing Python package '{missing}'. Use the crypto env or install dependencies with: "
        "python -m pip install -r blofin-sdk-python\\requirements.txt"
    ) from exc

print = partial(builtins.print, flush=True)


def load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env(Path(__file__).resolve().parent / ".env")

INST_ID = os.getenv("BLOFIN_INST_ID", "BTC-USDT")
BAR = os.getenv("BLOFIN_SUPPORT_BAR", "15m")
CANDLE_LIMIT = os.getenv("BLOFIN_SUPPORT_CANDLE_LIMIT", "500")
LOOKBACK = int(os.getenv("BLOFIN_SUPPORT_LOOKBACK", "3"))
CLUSTER_PCT = Decimal(os.getenv("BLOFIN_SUPPORT_CLUSTER_PCT", "0.001"))
SUPPORT_DEFAULT_BELOW = Decimal(os.getenv("BLOFIN_SUPPORT_DEFAULT_BELOW", "5000"))
RESISTANCE_DEFAULT_ABOVE = Decimal(os.getenv("BLOFIN_RESISTANCE_DEFAULT_ABOVE", "5000"))
SUPPORT_LEVEL_COUNT = int(os.getenv("BLOFIN_SUPPORT_LEVEL_COUNT", "8"))
RESISTANCE_LEVEL_COUNT = int(os.getenv("BLOFIN_RESISTANCE_LEVEL_COUNT", str(SUPPORT_LEVEL_COUNT)))
RECALCULATE_SECONDS = int(os.getenv("BLOFIN_LEVEL_RECALCULATE_SECONDS", "30"))
HTTP_PORT = int(os.getenv("BLOFIN_CHART_HTTP_PORT", "8765"))
WS_PORT = int(os.getenv("BLOFIN_CHART_WS_PORT", "8766"))
HOST = os.getenv("BLOFIN_CHART_HOST", "127.0.0.1")


def decimal_from(value: Any, fallback: str = "0") -> Decimal:
    if value is None or value == "":
        value = fallback
    return Decimal(str(value))


def quantize_down(value: Decimal, increment: Decimal) -> Decimal:
    if increment <= 0:
        return value
    units = (value / increment).to_integral_value(rounding=ROUND_DOWN)
    return units * increment


def decimal_to_str(value: Decimal) -> str:
    return format(value.normalize(), "f")


def candle_to_record(candle: Any) -> Optional[Dict[str, Any]]:
    if isinstance(candle, dict):
        timestamp = candle.get("ts") or candle.get("time")
        open_price = candle.get("open")
        high = candle.get("high")
        low = candle.get("low")
        close = candle.get("close")
    elif isinstance(candle, list) and len(candle) >= 5:
        timestamp = candle[0]
        open_price = candle[1]
        high = candle[2]
        low = candle[3]
        close = candle[4]
    else:
        return None

    try:
        return {
            "ts": int(timestamp),
            "open": decimal_from(open_price),
            "high": decimal_from(high),
            "low": decimal_from(low),
            "close": decimal_from(close),
        }
    except Exception:
        return None


def get_range(current_price: Decimal) -> tuple[Decimal, Decimal]:
    configured_low = os.getenv("BLOFIN_SUPPORT_LOW")
    configured_high = os.getenv("BLOFIN_RESISTANCE_HIGH") or os.getenv("BLOFIN_SUPPORT_HIGH")
    low_range = decimal_from(configured_low, str(current_price - SUPPORT_DEFAULT_BELOW))
    high_range = decimal_from(configured_high, str(current_price + RESISTANCE_DEFAULT_ABOVE))
    if low_range > high_range:
        low_range, high_range = high_range, low_range
    return low_range, high_range


def cluster_swings(
    swings: List[Dict[str, Any]],
    *,
    price_key: str,
    low_range: Decimal,
    high_range: Decimal,
    tick_size: Decimal,
) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    min_cluster_width = tick_size * Decimal("10") if tick_size > 0 else Decimal("0")

    for candle in sorted(swings, key=lambda item: item[price_key]):
        price = candle[price_key]
        tolerance = max(price * CLUSTER_PCT, min_cluster_width)
        target = None

        for cluster in clusters:
            if abs(price - cluster["level"]) <= tolerance:
                target = cluster
                break

        if target is None:
            clusters.append({
                "level": price,
                "touches": 1,
                "last_ts": candle["ts"],
                "prices": [price],
            })
            continue

        target["prices"].append(price)
        target["touches"] += 1
        target["last_ts"] = max(target["last_ts"], candle["ts"])
        target["level"] = sum(target["prices"], Decimal("0")) / Decimal(len(target["prices"]))

    levels = []
    for cluster in clusters:
        level = quantize_down(cluster["level"], tick_size)
        if low_range <= level <= high_range:
            levels.append({
                "level": level,
                "touches": cluster["touches"],
                "last_ts": cluster["last_ts"],
            })

    return sorted(levels, key=lambda item: (item["touches"], item["last_ts"]), reverse=True)


def find_support_levels(
    candles: List[Dict[str, Any]],
    *,
    low_range: Decimal,
    high_range: Decimal,
    tick_size: Decimal,
) -> List[Dict[str, Any]]:
    candles = sorted(candles, key=lambda item: item["ts"])
    if len(candles) < (LOOKBACK * 2) + 1:
        return []

    swings = []
    for index in range(LOOKBACK, len(candles) - LOOKBACK):
        candle = candles[index]
        low = candle["low"]
        if low < low_range or low > high_range:
            continue
        neighbors = candles[index - LOOKBACK:index] + candles[index + 1:index + LOOKBACK + 1]
        if all(low <= neighbor["low"] for neighbor in neighbors):
            swings.append(candle)

    return cluster_swings(
        swings,
        price_key="low",
        low_range=low_range,
        high_range=high_range,
        tick_size=tick_size,
    )


def find_resistance_levels(
    candles: List[Dict[str, Any]],
    *,
    low_range: Decimal,
    high_range: Decimal,
    tick_size: Decimal,
) -> List[Dict[str, Any]]:
    candles = sorted(candles, key=lambda item: item["ts"])
    if len(candles) < (LOOKBACK * 2) + 1:
        return []

    swings = []
    for index in range(LOOKBACK, len(candles) - LOOKBACK):
        candle = candles[index]
        high = candle["high"]
        if high < low_range or high > high_range:
            continue
        neighbors = candles[index - LOOKBACK:index] + candles[index + 1:index + LOOKBACK + 1]
        if all(high >= neighbor["high"] for neighbor in neighbors):
            swings.append(candle)

    return cluster_swings(
        swings,
        price_key="high",
        low_range=low_range,
        high_range=high_range,
        tick_size=tick_size,
    )


def public_candle(candle: Dict[str, Any]) -> Dict[str, float]:
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
                "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

    async def set_candles(self, candles: List[Dict[str, Any]]) -> None:
        async with self.lock:
            self.candles = sorted(candles, key=lambda item: item["ts"])
            if self.candles and self.current_price is None:
                self.current_price = self.candles[-1]["close"]
            self._recalculate_locked()

    async def upsert_candle(self, candle: Dict[str, Any]) -> None:
        async with self.lock:
            by_ts = {item["ts"]: item for item in self.candles}
            by_ts[candle["ts"]] = candle
            self.candles = sorted(by_ts.values(), key=lambda item: item["ts"])[-int(CANDLE_LIMIT):]
            self.current_price = candle["close"]
            self._recalculate_locked()

    async def set_price(self, price: Decimal) -> None:
        async with self.lock:
            self.current_price = price
            if self.candles:
                self.candles[-1]["close"] = price

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
    message = json.dumps(await state.snapshot())
    stale_clients = []
    for client in list(state.clients):
        try:
            await client.send(message)
        except Exception:
            stale_clients.append(client)
    for client in stale_clients:
        state.clients.discard(client)


def fetch_rest_candles(market_api: MarketAPI) -> List[Dict[str, Any]]:
    response = market_api.getCandlesticks(instId=INST_ID, bar=BAR, limit=CANDLE_LIMIT)
    return [
        record
        for candle in response.get("data", [])
        if (record := candle_to_record(candle)) is not None
    ]


def fetch_tick_size(market_api: MarketAPI) -> Decimal:
    response = market_api.getInstruments(instId=INST_ID)
    data = response.get("data")
    instrument = data[0] if isinstance(data, list) and data else {}
    return decimal_from(instrument.get("tickSize"), "0.1")


async def ws_handler(client: Any, state: LiveChartState) -> None:
    state.clients.add(client)
    await client.send(json.dumps(await state.snapshot()))
    try:
        await client.wait_closed()
    finally:
        state.clients.discard(client)


async def refresh_candles_loop(state: LiveChartState, market_api: MarketAPI) -> None:
    while True:
        candles = await asyncio.to_thread(fetch_rest_candles, market_api)
        await state.set_candles(candles)
        await broadcast(state)
        await asyncio.sleep(RECALCULATE_SECONDS)


async def blofin_stream_loop(state: LiveChartState) -> None:
    while True:
        client = BlofinWsPublicClient(isDemo=True)
        try:
            await client.connect()
            await client.subscribeTickers(INST_ID)
            await client.subscribeCandles(INST_ID, BAR)
            print(f"Connected to BloFin demo public websocket for {INST_ID}.")

            async for message in client.listen():
                channel = message.get("arg", {}).get("channel")
                data = message.get("data")
                if not data:
                    continue

                if channel == "tickers" and isinstance(data, list):
                    price = decimal_from(data[0].get("last"))
                    if price > 0:
                        await state.set_price(price)
                        await broadcast(state)

                if channel == f"candle{BAR}" and isinstance(data, list):
                    candle = candle_to_record(data[0])
                    if candle:
                        await state.upsert_candle(candle)
                        await broadcast(state)

        except Exception as exc:
            print(f"BloFin stream error: {exc}. Reconnecting soon.")
            await asyncio.sleep(2)
        finally:
            try:
                await client.close()
            except Exception:
                pass


def start_http_server(directory: Path, host: str, port: int) -> ThreadingHTTPServer:
    class ChartHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), ChartHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live BloFin demo chart with support/resistance levels.")
    parser.add_argument("--host", default=HOST, help="Local host for HTTP and websocket servers.")
    parser.add_argument("--http-port", type=int, default=HTTP_PORT, help="HTTP chart port.")
    parser.add_argument("--ws-port", type=int, default=WS_PORT, help="Websocket data port.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    chart_file = project_dir / "live-chart.html"
    if not chart_file.exists():
        raise SystemExit(f"Missing chart file: {chart_file}")

    market_api = MarketAPI(DemoClient())
    state = LiveChartState()
    state.tick_size = await asyncio.to_thread(fetch_tick_size, market_api)
    await state.set_candles(await asyncio.to_thread(fetch_rest_candles, market_api))

    http_server = start_http_server(project_dir, args.host, args.http_port)
    ws_server = await websockets.serve(lambda client: ws_handler(client, state), args.host, args.ws_port)

    print(f"Live chart: http://{args.host}:{args.http_port}/live-chart.html")
    print(f"Data websocket: ws://{args.host}:{args.ws_port}")
    print("Press Ctrl+C to stop.")

    try:
        await asyncio.gather(
            refresh_candles_loop(state, market_api),
            blofin_stream_loop(state),
        )
    finally:
        ws_server.close()
        await ws_server.wait_closed()
        http_server.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped live chart server.")
