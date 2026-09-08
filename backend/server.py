"""Everything that talks to the outside world: the local HTTP server that
serves the chart page, the local websocket server that pushes live updates
to it, and the three background loops that keep LiveChartState fresh:

  - refresh_candles_loop  — periodic REST candle refetch (source of truth)
  - poll_ticker_loop      — frequent REST price polling (fallback/backup)
  - blofin_stream_loop    — the primary path: BloFin's public websocket feed
"""

import asyncio
import datetime as dt
import json
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional, Tuple

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


# A supervised loop that has been alive at least this long is treated as
# healthy, so its restart backoff resets. Without it, a loop that runs fine for
# six hours and then hits one bad response would inherit the backoff from a
# failure last night and sit out a minute for no reason.
HEALTHY_AFTER_SECONDS = 60.0

# Cap on the restart backoff. Long enough not to hammer a venue that is down,
# short enough that an unattended overnight run recovers promptly once it is
# back.
MAX_RESTART_BACKOFF_SECONDS = 60.0


def log(message: str) -> None:
    """Print with a UTC timestamp, unbuffered.

    Both halves matter for a process meant to run for days. Without the
    timestamp there is no way to line a failure up against the exchange's own
    logs; without `flush`, a redirected stdout is block-buffered and the last
    few KB — which is exactly the part describing the failure — is lost when
    the process dies.
    """
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp}Z  {message}", flush=True)


async def supervise(name: str, start: Callable[[], Awaitable[None]]) -> None:
    """Run a background loop forever, restarting it if it ever falls over.

    This exists because of a specific, expensive failure. `run_servers` used
    to `asyncio.gather` the raw loops, and `gather` propagates the first
    exception it sees — so an unhandled error in *any* loop ended the whole
    process, including the microstructure feed, the recorder and the raw
    archive. On 2026-09-08 a single dropped HTTP connection on the chart's
    candle endpoint did precisely that, seven hours into an overnight run.

    The chart is a convenience and the recording is not: raw events cannot be
    re-collected, that market moment is gone. So no loop is permitted to take
    another one down with it. `start` is a factory rather than a coroutine
    because a coroutine can only be awaited once, and this may need to build
    several.

    `CancelledError` is re-raised untouched — that is Ctrl+C and shutdown, not
    a failure, and swallowing it would make the process unkillable.
    """
    backoff = 1.0
    restarts = 0
    while True:
        began = time.monotonic()
        try:
            await start()
            log(f"[{name}] returned unexpectedly (it should run forever).")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if time.monotonic() - began >= HEALTHY_AFTER_SECONDS:
                backoff = 1.0
            restarts += 1
            log(f"[{name}] crashed: {type(exc).__name__}: {exc}")
            log(f"[{name}] restart #{restarts} in {backoff:.0f}s. "
                f"Traceback follows.")
            traceback.print_exc()
            sys.stdout.flush()
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, MAX_RESTART_BACKOFF_SECONDS)


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
    slow-but-reliable path that corrects any drift from the streaming path.

    Every REST call here is allowed to fail. A long-lived connection pool will
    eventually meet a `RemoteDisconnected` — the server hangs up on an idle
    keep-alive connection and `requests` surfaces it as a ConnectionError —
    and on 2026-09-08 at 03:02 UTC exactly that killed a collection run seven
    hours in, because this was the one loop in the file without a handler.

    Nothing here is irreplaceable: these candles feed the *chart*. Losing a
    refresh costs a stale display until the next one.
    """
    backoff = 1.0
    while True:
        try:
            candles = await asyncio.to_thread(fetch_rest_candles, market_api)
            await state.set_candles(candles)
            await broadcast(state)
            backoff = 1.0
            delay: float = RECALCULATE_SECONDS
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log(f"Candle refresh error ({type(exc).__name__}: {exc}); "
                f"retrying in {backoff:.0f}s.")
            # Retry sooner than the normal cadence, but back off so a venue
            # that is properly down is not hammered.
            delay = backoff
            backoff = min(backoff * 2, float(RECALCULATE_SECONDS))
        await asyncio.sleep(delay)


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

    # (name, factory). Factories rather than coroutines because `supervise`
    # may have to build a fresh one on every restart.
    loops: List[Tuple[str, Callable[[], Awaitable[None]]]] = [
        ("candle refresh", lambda: refresh_candles_loop(state, market_api)),
        ("ticker poll", lambda: poll_ticker_loop(state, market_api)),
        ("chart stream", lambda: blofin_stream_loop(state)),
    ]
    if feed is not None:
        loops.append(("microstructure feed", feed.run))
        loops.append(("feature publish", lambda: publish_features_loop(state, feed)))
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
        # Every loop is supervised, so none of them can end the run by
        # failing. The only way out of this gather is cancellation.
        await asyncio.gather(*(supervise(name, start) for name, start in loops))
    finally:
        if feed is not None:
            # Flush any buffered CSV rows before the process exits.
            feed.close()
        ws_server.close()
        await ws_server.wait_closed()
        http_server.shutdown()
