"""Move the cross-sectional carry book to its plan. Dry run unless --confirm.

    python backend\\run_carry_xs.py --notional 4000 --min-volume 2000000
    python backend\\run_carry_xs.py --notional 4000 --min-volume 2000000 --confirm

Without `--confirm` this rehearses: it builds the plan against live prices,
reads the positions already on the account, prints every order it would send
and in what sequence, and sends nothing. With `--confirm` it sends them.

Demo by default and demo on purpose. `--production` exists so the flag has to
be typed, and `--production --confirm` together is refused outright - this
strategy has never been traded, and the first time it is should be a decision
somebody makes twice.

One verb, and it is a rebalance
-------------------------------
There is no open and no close, because this book is rebalanced weekly and both
are special cases of the same operation: `reconcile` moves the exchange from
whatever it is holding to whatever the plan wants. From flat it opens; against
an existing book it rebalances; with a refused plan it does nothing. It is
idempotent, so an interrupted run is finished by running it again - and with
twelve legs, being interrupted is the normal case rather than the exceptional
one.

Read in this order
------------------
**REFUSALS** first: the plan's gates are the executor's gates, and a refused
plan is never sent. Then the **sequence**, which is not the order the legs are
listed in - orders are interleaved so the partly-filled book stays close to
neutral, and `worst intermediate net` is the measured bound on how directional
this gets between the first fill and the last. Then, after a live run, the
**verification**, which reads the positions back from the exchange rather than
trusting what this process thinks it sent.
"""

from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TAKER_FEE_BPS  # noqa: E402
from plan_carry import DEMO_BASE_URL, PRODUCTION_BASE_URL  # noqa: E402
from plan_carry_xs import gather, show  # noqa: E402
from trading.risk import RiskLimits  # noqa: E402
from trading.strategies.carry_xs import BookConfig, plan_book  # noqa: E402
from trading.strategies.carry_xs.broker import BlofinPerpBroker  # noqa: E402
from trading.strategies.carry_xs.execute import BookExecutor  # noqa: E402


def log(message: str = "") -> None:
    print(message, flush=True)


def report(result) -> None:
    log()
    log("=" * 78)
    log(("REHEARSAL - nothing sent" if result.dry_run
         else "EXECUTION").center(78))
    log("=" * 78)

    if result.already_correct:
        log(result.summary or "The book already matches the plan.")
        log("Nothing to do.")
        return
    if not result.orders:
        log("No orders.")
        for problem in result.problems:
            log("  - " + problem)
        return

    log("{:<16}{:<10}{:>12}{:>14}{:>12}{:>14}".format(
        "instrument", "action", "side", "contracts", "notional $",
        "net after $"))
    for order in result.orders:
        flag = ""
        if order.error:
            flag = "  FAILED: " + order.error
        elif order.sent:
            flag = "  sent"
        log("{:<16}{:<10}{:>12}{:>14}{:>14,.2f}{:>14,.2f}{}".format(
            order.inst_id,
            order.reason + ("*" if order.reduce_only else ""),
            order.side, order.contracts, order.notional_usd,
            order.net_after, flag))
    log("  * reduce_only: can only shrink a position, never open one")
    log()
    log("worst intermediate net exposure  ${:,.2f}".format(result.max_net_usd))
    log("  This is how directional the book gets between the first fill and "
        "the last.\n  Orders are interleaved to keep it near one leg rather "
        "than half the book.")

    if not result.dry_run:
        log()
        log("VERIFICATION, read back from the exchange")
        log("  gross before   ${:,.2f}".format(result.gross_before_usd))
        log("  gross after    ${:,.2f}".format(result.gross_after_usd))
        log("  net after      ${:,.2f}{}".format(
            result.net_after_usd,
            "  ({:.2%} of gross)".format(
                abs(result.net_after_usd) / result.gross_after_usd)
            if result.gross_after_usd else ""))
        if result.repaired:
            log("  the book was directional and was reduced back toward "
                "neutral")

    for warning in result.warnings:
        log("  warning: " + warning)

    log()
    if result.ok and result.dry_run:
        log("Nothing was sent. Re-run with --confirm to place these orders on "
            "the demo\naccount.")
    elif result.ok:
        log("Reconciled. The book on the exchange now matches the plan.")
    else:
        log("PROBLEMS, all of them:")
        for problem in result.problems:
            log("  - " + problem)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--notional", type=Decimal, default=Decimal("2000"),
                        help="gross notional: half long, half short. Start small.")
    parser.add_argument("--leverage", type=Decimal, default=Decimal("2"))
    parser.add_argument("--top-frac", type=float, default=0.3)
    parser.add_argument("--hold-days", type=int, default=7)
    parser.add_argument("--carry-days", type=int, default=7)
    parser.add_argument("--min-volume", type=float, default=2e6)
    parser.add_argument("--min-funding-days", type=int, default=30)
    parser.add_argument("--max-instruments", type=int, default=80)
    parser.add_argument("--max-spread-bps", type=float, default=30.0,
                        help="skip an opening leg whose spread is wider than "
                             "this when it is about to be sent")
    parser.add_argument("--repair-only", action="store_true",
                        help="do not rebalance: just bring the book back to "
                             "neutral with reduce_only trims. This is what the "
                             "monitor's not-neutral alert points at.")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually send the orders. Without this it is a "
                             "rehearsal.")
    parser.add_argument("--production", action="store_true",
                        help="Against the live account rather than demo. Do "
                             "not use this until the book has run end to end "
                             "on demo.")
    args = parser.parse_args(argv)

    if args.production and args.confirm:
        raise SystemExit(
            "--production with --confirm sends real orders with real money, "
            "and this\nstrategy has never been traded at all. Run it on demo "
            "first, hold it for a\nfull rebalance, and compare the realised "
            "result against the prediction.\nThen decide deliberately.")

    from blofin.client import Client
    from blofin.rest_market import MarketAPI
    from blofin.rest_trading import TradingAPI

    base_url = PRODUCTION_BASE_URL if args.production else DEMO_BASE_URL
    environment = "production" if args.production else "demo"

    api_key = os.environ.get("API_KEY")
    secret = os.environ.get("SECRET")
    passphrase = os.environ.get("PASSPHRASE")
    if not (api_key and secret and passphrase):
        raise SystemExit("API_KEY, SECRET and PASSPHRASE must be set in .env.")

    client = Client(apiKey=api_key, apiSecret=secret, passphrase=passphrase,
                    baseUrl=base_url)
    # Market data is read from PRODUCTION whichever account is being traded.
    # Demo's book is not the market, and a plan built on it would rank coins by
    # a funding rate and a spread that nobody is actually quoting.
    public = Client()
    market = MarketAPI(public)
    # Size rules from the host the orders go to; prices from production.
    account_market = MarketAPI(client)

    config = BookConfig(
        top_frac=args.top_frac, hold_days=args.hold_days,
        min_volume_usd=args.min_volume,
        min_funding_days=args.min_funding_days,
        gross_notional_usd=args.notional, leverage=args.leverage)

    executor_kwargs = dict(dry_run=not args.confirm, on_log=log,
                           max_spread_bps=args.max_spread_bps,
                           net_tolerance_frac=config.delta_tolerance_frac)

    if args.repair_only:
        if args.confirm:
            log("--confirm given: these reduce_only trims WILL be sent to "
                + environment + ".")
        executor = BookExecutor(
            BlofinPerpBroker(client, TradingAPI(client), account_market),
            **executor_kwargs)
        result = executor.repair_only()
        report(result)
        return 0 if result.ok else 1

    log(environment + ": building the plan from live prices...")
    candidates = gather(market, min_volume=args.min_volume,
                        carry_days=args.carry_days,
                        min_funding_days=args.min_funding_days,
                        limit=args.max_instruments,
                        rules_api=account_market)
    if not candidates:
        raise SystemExit(
            "No instruments cleared the ${:,.0f}/24h filter.".format(
                args.min_volume))

    limits = RiskLimits(max_notional=max(args.notional,
                                         RiskLimits().max_notional))
    plan = plan_book(candidates, config, limits,
                     taker_fee_bps=float(TAKER_FEE_BPS))
    show(plan, config)
    if not plan.ok:
        return 1

    if args.confirm:
        log()
        log("--confirm given: these orders WILL be sent to " + environment + ".")

    executor = BookExecutor(
        BlofinPerpBroker(client, TradingAPI(client), account_market),
        **executor_kwargs)
    result = executor.reconcile(plan)
    report(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
