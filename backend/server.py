"""Everything that talks to the outside world: the local HTTP server that
serves the chart page, the local websocket server that pushes live updates
to it, and the three background loops that keep LiveChartState fresh:

  - refresh_candles_loop  — periodic REST candle refetch (source of truth)
  - poll_ticker_loop      — frequent REST price polling (fallback/backup)
  - blofin_stream_loop    — the primary path: BloFin's public websocket feed
"""

import asyncio
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import websockets
from blofin.rest_market import MarketAPI
from blofin.websocket_client import BlofinWsPublicClient

from config import (
    BAR,
    BOOK_DEPTH,
    DATA_DIR,
    FEATURE_SAMPLE_INTERVAL_MS,
    FRONTEND_DIR,
    INST_ID,
    LABEL_HORIZONS,
    LABEL_THRESHOLD_BPS,
    RECALCULATE_SECONDS,
    RECORD_FEATURES,
    RECORD_RAW,
    TAPE_WINDOW_SECONDS,
    TICKER_POLL_SECONDS,
    USE_DEMO,
)
from market_data import candle_to_record, fetch_rest_candles, fetch_rest_ticker_price, ticker_to_price
from state import LiveChartState, broadcast
from trading.ingest import MicrostructureFeed
from trading.recorder import LabelConfig


async def ws_handler(client: Any, state: LiveChartState) -> None:
    """Register a new websocket client, send it an initial snapshot, then
    just keep it open until the browser tab disconnects."""
    state.clients.add(client)
    await client.send(json.dumps(await state.snapshot()))
    try:
        await client.wait_closed()
    finally:
        state.clients.discard(client)


async def refresh_candles_loop(state: LiveChartState, market_api: MarketAPI) -> None:
    """Periodically re-fetch the full candle history from REST. This is the
    slow-but-reliable path that corrects any drift from the streaming path."""
    while True:
        candles = await asyncio.to_thread(fetch_rest_candles, market_api)
        await state.set_candles(candles)
        await broadcast(state)
        await asyncio.sleep(RECALCULATE_SECONDS)


async def poll_ticker_loop(state: LiveChartState, market_api: MarketAPI) -> None:
    """Frequently poll the REST ticker as a backup price source in case the
    websocket stream is down or lagging."""
    while True:
        try:
            price = await asyncio.to_thread(fetch_rest_ticker_price, market_api)
            if price is not None and await state.set_price(price):
                await broadcast(state)
        except Exception as exc:
            print(f"Ticker poll error: {exc}")
        await asyncio.sleep(TICKER_POLL_SECONDS)


async def blofin_stream_loop(state: LiveChartState) -> None:
    """The primary, low-latency data path: BloFin's public websocket feed
    for live tickers and candles. Reconnects automatically on any error."""
    while True:
        client = BlofinWsPublicClient(isDemo=USE_DEMO)
        try:
            await client.connect()
            await client.subscribeTickers(INST_ID)
            await client.subscribeCandles(INST_ID, BAR)
            mode = "demo" if USE_DEMO else "production"
            print(f"Connected to BloFin {mode} public websocket for {INST_ID}.")

            async for message in client.listen():
                channel = message.get("arg", {}).get("channel")
                data = message.get("data")
                if not data:
                    continue

                if channel == "tickers" and isinstance(data, list):
                    price = ticker_to_price(data[0])
                    if price is not None:
                        if await state.set_price(price):
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


def build_microstructure_feed() -> MicrostructureFeed:
    """Construct the order book / trade / feature feed from config."""
    return MicrostructureFeed(
        INST_ID,
        use_demo=USE_DEMO,
        book_depth=BOOK_DEPTH,
        data_dir=DATA_DIR,
        record=RECORD_FEATURES,
        record_raw=RECORD_RAW,
        sample_interval_ms=FEATURE_SAMPLE_INTERVAL_MS,
        tape_window_seconds=TAPE_WINDOW_SECONDS,
        label_config=LabelConfig(
            horizons_seconds=LABEL_HORIZONS,
            threshold_bps=LABEL_THRESHOLD_BPS,
        ),
    )


async def publish_features_loop(
    state: LiveChartState,
    feed: MicrostructureFeed,
    interval_seconds: float = 0.25,
) -> None:
    """Push the latest features to browsers at a fixed, modest cadence.

    The feature engine recomputes on every book and trade event — often 50+
    times a second. Broadcasting each one would saturate the websocket and the
    browser for no visual benefit, so we sample `feed.latest` on a timer
    instead. The engine keeps full resolution internally; only the *display*
    is downsampled.
    """
    while True:
        try:
            snapshot = feed.latest
            await state.set_features(snapshot.to_dict(), feed.status())
            await broadcast(state)
        except Exception as exc:
            print(f"Feature publish error: {exc}")
        await asyncio.sleep(interval_seconds)


def start_http_server(directory: Path, host: str, port: int) -> ThreadingHTTPServer:
    """Serve the frontend/ folder as static files (the chart HTML/JS/CSS and
    its node_modules dependencies) on a background thread."""

    class ChartHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), ChartHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def run_servers(
    state: LiveChartState,
    market_api: MarketAPI,
    host: str,
    http_port: int,
    ws_port: int,
    feed: Optional[MicrostructureFeed] = None,
):
    """Start the HTTP + websocket servers and run all background loops until
    cancelled/interrupted. Returns nothing; runs forever (or until Ctrl+C).

    `feed` is optional so the chart still runs standalone if the
    microstructure stack is disabled or fails to construct.
    """
    http_server = start_http_server(FRONTEND_DIR, host, http_port)
    ws_server = await websockets.serve(lambda client: ws_handler(client, state), host, ws_port)

    print(f"Live chart: http://{host}:{http_port}/live-chart.html")
    print(f"Data websocket: ws://{host}:{ws_port}")

    tasks = [
        refresh_candles_loop(state, market_api),
        poll_ticker_loop(state, market_api),
        blofin_stream_loop(state),
    ]
    if feed is not None:
        tasks.append(feed.run())
        tasks.append(publish_features_loop(state, feed))
        if RECORD_FEATURES:
            print(f"Recording labelled features to: {DATA_DIR}")
        else:
            print("Feature recording is OFF (BLOFIN_RECORD_FEATURES=false).")
        if RECORD_RAW:
            print(f"Archiving raw events to:        {DATA_DIR / 'raw'}")
        else:
            print("Raw archiving is OFF (BLOFIN_RECORD_RAW=false) - future "
                  "features cannot be backfilled.")

    print("Press Ctrl+C to stop.")

    try:
        await asyncio.gather(*tasks)
    finally:
        if feed is not None:
            # Flush any buffered CSV rows before the process exits.
            feed.close()
        ws_server.close()
        await ws_server.wait_closed()
        http_server.shutdown()
