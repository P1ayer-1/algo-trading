"""Paper-quote the lead strategy live. Dry unless --confirm; demo unless --production.

    python backend\\run_lead_quote.py --instruments SUI-USDT --measure-only
    python backend\\run_lead_quote.py --instruments SUI-USDT --minutes 60
    python backend\\run_lead_quote.py --instruments SUI-USDT --minutes 60 --confirm
    python backend\\run_lead_quote.py --summary data\\SUI-USDT\\lead_quote\\2026-09-12T183000-paper.jsonl

What each mode does
-------------------
Every mode reads PRODUCTION market data: Binance's public `bookTicker` for the
leader and BloFin's public books and trades for the follower. Binance is a
data feed here, never an account.

  --measure-only   listen for the warm-up, print tick, spread and the two feed
                   lags on this host against the gates of step 9ae, and exit.
  (default)        the same gate, then quote on PAPER: the state machine posts
                   and cancels in its own head and fills itself from the
                   production tape. Nothing is sent anywhere.
  --probe N        with --confirm: N far-from-touch post_only orders, each
                   cancelled, to time the venue round trip from this host.
                   The number to compare a Germany box with a Tokyo one.
  --http           aiohttp (default) or requests, for the demo order path.
                   aiohttp measured 4-6 ms faster of ~185 (README 9ae).
  --confirm        additionally mirror every intent to the DEMO account as a
                   real `post_only` order, cancel, or reduce-only exit, timing
                   each acknowledgement and logging the demo order stream.
                   The demo book is not the market; this measures latency and
                   the venue's order handling, not the edge.
  --production     is refused together with --confirm. The strategy has
                   never quoted anywhere; production is a decision for after
                   the demo log agrees with the backtest, and it is asked for
                   by name, not by flag.

One instrument at a time per process: two feeds and a clock per instrument
is plenty to reason about, and two processes cannot share a log.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import urllib.request
import json
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from trading.strategies.lead_quote import QuoteConfig, summarise  # noqa: E402
from trading.strategies.lead_quote.execute import (  # noqa: E402
    AsyncBlofinQuoteBroker, BlofinQuoteBroker, LeadQuoteRunner, RunLog)

DEMO_BASE_URL = "https://demo-trading-openapi.blofin.com"
PRODUCTION_BASE_URL = "https://openapi.blofin.com"


def instrument_rules(inst_id: str, base_url: str) -> dict:
    url = base_url + "/api/v1/market/instruments?instType=SWAP&instId=" + inst_id
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for row in payload.get("data") or []:
        if row.get("instId") == inst_id:
            return row
    raise SystemExit(inst_id + " is not listed at " + base_url)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instruments", default="SUI-USDT", help="one BloFin instId")
    parser.add_argument("--edge-bps", type=float, default=7.0)
    parser.add_argument("--stop-bps", type=float, default=3.0)
    parser.add_argument("--order-ttl-s", type=float, default=10.0)
    parser.add_argument("--hold-s", type=float, default=120.0)
    parser.add_argument("--size", type=Decimal, default=None,
                        help="contracts per demo order; default the instrument's minimum")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=20.0)
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--probe", type=int, default=0,
                        help="with --confirm: N post_only orders 5%% under the bid, each cancelled, "
                             "to time the venue round trip from this host; no fill possible")
    parser.add_argument("--http", choices=("aiohttp", "requests"), default="aiohttp",
                        help="REST transport for demo orders: the SDK's aiohttp AsyncClient awaited "
                             "on the loop, or its requests Client in a worker thread")
    parser.add_argument("--confirm", action="store_true", help="mirror intents to the demo account")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--summary", help="summarise this run log and exit")
    args = parser.parse_args(argv)
    # Runs are left going as `nohup ... > lq_<INST>.out`, and redirected stdout is
    # block-buffered: on 2026-09-13 an 8-hour SUI run's .out held nothing past
    # startup, its paper fill lines still in the buffer. One line, one write.
    sys.stdout.reconfigure(line_buffering=True)

    if args.summary:
        for line in summarise(Path(args.summary)).lines():
            print(line)
        return 0
    if args.probe and not args.confirm:
        raise SystemExit("--probe sends (and cancels) real demo orders; it needs --confirm.")
    if args.production and args.confirm:
        raise SystemExit(
            "--production with --confirm would quote real money with a strategy that has\n"
            "never quoted anywhere. Run it on demo, read the log back with --summary,\n"
            "compare it with README step 9ae, and then ask for production by name.")
    inst_id = args.instruments.strip()
    if "," in inst_id:
        raise SystemExit("one instrument per process; start another process for the next")

    base_url = PRODUCTION_BASE_URL if args.production else DEMO_BASE_URL
    environment = "production" if args.production else "demo"
    # Tick size from PRODUCTION: it is the price grid the paper quote lives on.
    # Size rules AND the tick orders are priced on from the host the demo orders
    # go to, as the carry book learned: demo listed FIL-USDT at 0.001 against
    # production's 0.0001 (2026-09-13) and rejected most posts for precision.
    prod = instrument_rules(inst_id, PRODUCTION_BASE_URL)
    tick = float(prod["tickSize"])
    size = args.size
    demo_tick = None
    if args.confirm:
        rules = instrument_rules(inst_id, base_url)
        demo_tick = Decimal(str(rules["tickSize"]))
        if size is None:
            size = Decimal(str(rules.get("minSize") or rules.get("lotSize") or "1"))
        if demo_tick != Decimal(str(prod["tickSize"])):
            print("{} tick is {} on {} but {} on production: the paper quote stays on production's "
                  "grid, orders are rounded to {}'s (bids down, asks up) and can sit behind it".format(
                      inst_id, demo_tick, environment, prod["tickSize"], environment))
    quote = QuoteConfig(tick=tick, edge_bps=args.edge_bps, stop_bps=args.stop_bps,
                        order_ttl_ms=int(args.order_ttl_s * 1000), hold_ms=int(args.hold_s * 1000),
                        maker_bps=float(config.MAKER_FEE_BPS), taker_bps=float(config.TAKER_FEE_BPS))
    # One file per run, not per day: a dry run and a demo run an hour apart are
    # different experiments and must not be summarised as one.
    log_path = Path("data") / inst_id / "lead_quote" / (
        time.strftime("%Y-%m-%dT%H%M%S", time.gmtime()) + ("-demo" if args.confirm else "-paper") + ".jsonl")
    log = RunLog(log_path)

    broker = None
    private_feed = None
    if args.confirm:
        from blofin.client import Client
        from blofin.rest_trading import TradingAPI
        from blofin.websocket_client import BlofinWsPrivateClient
        api_key, secret, passphrase = (os.environ.get(k) for k in ("API_KEY", "SECRET", "PASSPHRASE"))
        if not (api_key and secret and passphrase):
            raise SystemExit("API_KEY, SECRET and PASSPHRASE must be set in .env.")
        if args.http == "aiohttp":
            from blofin.async_client import AsyncClient
            client = AsyncClient(apiKey=api_key, apiSecret=secret, passphrase=passphrase, baseUrl=base_url)
            broker = AsyncBlofinQuoteBroker(client, TradingAPI(client))
        else:
            client = Client(apiKey=api_key, apiSecret=secret, passphrase=passphrase, baseUrl=base_url)
            broker = BlofinQuoteBroker(client, TradingAPI(client))
        private_feed = lambda: BlofinWsPrivateClient(api_key, secret, passphrase, isDemo=not args.production)  # noqa: E731
        print("--confirm given: intents WILL be mirrored to the {} account as post_only orders, "
              "size {} contracts, over {}".format(environment, size, args.http))
    else:
        print("dry run: paper fills from the production tape, nothing sent. --confirm mirrors to demo.")

    runner = LeadQuoteRunner(inst_id, quote, log=log, broker=broker, size=size or Decimal("1"),
                             warmup_seconds=args.warmup_seconds, private_feed=private_feed,
                             demo_tick=demo_tick)
    print("log: " + str(log_path))
    try:                       # a faster event loop where it is installed (Linux: pip install uvloop)
        import uvloop
        uvloop.install()
        print("event loop: uvloop")
    except ImportError:
        print("event loop: asyncio default (pip install uvloop on Linux for a faster one)")
    try:
        plan = asyncio.run(runner.run(minutes=args.minutes, measure_only=args.measure_only,
                                      probe_cycles=args.probe))
    except KeyboardInterrupt:
        print("interrupted")
        plan = runner.plan
    finally:
        log.close()
    print()
    for line in summarise(log_path).lines():
        print(line)
    if plan is not None and not plan.ok:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
