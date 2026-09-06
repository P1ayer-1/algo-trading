"""Entrypoint: run this file to start the live BloFin chart.

    python backend\\live-chart.py

This file itself does almost nothing — it just wires together the pieces
that live in the other modules in this folder:

    config.py              - all settings (env vars, paths)
    market_data.py         - fetching/parsing BloFin candles & prices
    support_resistance.py  - the swing/cluster level-finding logic
    state.py               - the shared LiveChartState + broadcasting
    server.py               - HTTP/websocket servers + background loops

If you're new to this codebase, read those files in roughly that order.
"""

import argparse
import asyncio
import builtins
import logging
from functools import partial

import config  # noqa: F401  (import first: sets up sys.path for the SDK, loads .env)
from config import HOST, HTTP_PORT, USE_DEMO, WS_PORT

try:
    from blofin.client import Client, DemoClient
    from blofin.rest_market import MarketAPI

    from market_data import fetch_rest_candles, fetch_tick_size
    from server import run_servers
    from state import LiveChartState
except ModuleNotFoundError as exc:
    missing = exc.name or "a dependency"
    raise SystemExit(
        f"Missing Python package '{missing}'. From the repo root, install dependencies with: "
        "python -m pip install -r blofin-sdk-python\\requirements.txt"
    ) from exc

# Make print() flush immediately so log lines show up right away when piped.
print = partial(builtins.print, flush=True)
logging.getLogger("websockets.server").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live BloFin demo chart with support/resistance levels.")
    parser.add_argument("--host", default=HOST, help="Local host for HTTP and websocket servers.")
    parser.add_argument("--http-port", type=int, default=HTTP_PORT, help="HTTP chart port.")
    parser.add_argument("--ws-port", type=int, default=WS_PORT, help="Websocket data port.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    market_api = MarketAPI(DemoClient() if USE_DEMO else Client())
    state = LiveChartState()
    state.tick_size = await asyncio.to_thread(fetch_tick_size, market_api)
    await state.set_candles(await asyncio.to_thread(fetch_rest_candles, market_api))

    await run_servers(state, market_api, args.host, args.http_port, args.ws_port)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped live chart server.")
