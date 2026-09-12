"""Read the cross-sectional carry book back and score it. READ ONLY.

    python backend\\monitor_carry_xs.py
    python backend\\monitor_carry_xs.py --production
    python backend\\monitor_carry_xs.py --archive-epoch

There is no `--confirm` here and there is nothing for one to do. This is the
command you can run at 3am without reading the source first to check what it
might do: it reads positions, equity, fills and public funding rates, and
prints. `trading/strategies/carry_xs/monitor.py` does not import the broker at
all - it defines its own four-method `Reader`, none of which writes - so the
file is structurally unable to place an order rather than merely choosing not
to.

What it is for
--------------
The plan says "+9.2 bps expected over 7 days" and the executor says the fills
went in. Neither of them ever comes back. The two numbers that settle whether
either was right are the funding actually credited and the neutrality actually
carried, and both have to be read from the exchange.

Read the ALERTS first. `critical` means something has happened that the
strategy does not survive unattended - a leg liquidated, the book gone
directional, funding turned negative - and the exit codes say so, so a
scheduler can act without parsing prose.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_carry import DEMO_BASE_URL, PRODUCTION_BASE_URL  # noqa: E402
from trading.risk import RiskLimits  # noqa: E402
from trading.strategies.carry_xs.monitor import (  # noqa: E402
    BookReport,
    archive_epoch,
    compare,
    fees_since,
    implied_funding,
    load_or_freeze,
    read_snapshot,
)

STATE = Path(__file__).resolve().parent.parent / "data" / "carry_xs"


class BlofinReader:
    """Four read calls. No `place` anything, by construction."""

    def __init__(self, client, market):
        self.client = client
        self.market = market

    def positions(self) -> List[Dict[str, Any]]:
        payload = self.client.get("/api/v1/account/positions", params={},
                                  sign=True)
        return [row for row in (payload.get("data") or []) if isinstance(row, dict)]

    def balance(self) -> Dict[str, Any]:
        payload = self.client.get("/api/v1/account/balance", params={}, sign=True)
        return payload.get("data") or {}

    def fills(self, *, since_ms: int) -> List[Dict[str, Any]]:
        payload = self.client.get("/api/v1/trade/fills-history",
                                  params={"begin": str(since_ms), "limit": "100"},
                                  sign=True)
        return [row for row in (payload.get("data") or []) if isinstance(row, dict)]

    def funding_history(self, inst_id: str, *,
                        since_ms: int) -> List[Dict[str, Any]]:
        payload = self.market.getFundingRateHistory(inst_id, limit="100")
        return [row for row in (payload.get("data") or []) if isinstance(row, dict)]

    def contract_values(self) -> Dict[str, Decimal]:
        payload = self.market.getInstruments()
        out: Dict[str, Decimal] = {}
        for row in payload.get("data") or []:
            inst_id = str(row.get("instId") or "")
            try:
                value = Decimal(str(row.get("contractValue")))
            except Exception:                          # noqa: BLE001
                continue
            if inst_id and value > 0:
                out[inst_id] = value
        return out


def show(report: BookReport, snapshot, environment: str) -> None:
    print()
    print("=" * 78)
    print(("CROSS-SECTIONAL CARRY BOOK — " + environment).center(78))
    print("=" * 78)

    if not report.open:
        print("No open positions on this account.")
        return

    print("{:<14}{:>12}{:>14}{:>13}{:>13}{:>12}".format(
        "instrument", "contracts", "notional $", "unreal $", "realised $",
        "liq price"))
    for leg in sorted(snapshot.legs, key=lambda item: -abs(item.notional_usd)):
        print("{:<14}{:>12}{:>14,.2f}{:>13,.4f}{:>13,.4f}{:>12}".format(
            leg.inst_id, leg.contracts, leg.notional_usd,
            leg.unrealized_pnl_usd, leg.realized_pnl_usd,
            "{:.6g}".format(leg.liquidation_price)
            if leg.liquidation_price else "-"))

    print()
    print("BOOK, held {:.2f} days".format(report.days_elapsed))
    print("  legs                {}".format(report.legs))
    print("  gross notional      ${:,.2f}".format(report.gross_usd))
    print("  net exposure        ${:,.2f}  ({:.2%} of gross)".format(
        report.net_usd, report.net_frac))
    print("  account equity      ${:,.2f}".format(report.total_equity_usd))
    if report.margin_ratio is not None:
        print("  margin ratio        {:.1f}  (account-level: the legs share one "
              "cross pool)".format(report.margin_ratio))

    print()
    print("FUNDING, derived and then controlled")
    print("  derived             ${:+,.4f}   = sum(realizedPnl) + fees".format(
        report.funding_derived_usd))
    print("  public-rate control ${:+,.4f}   = settlements x notional, by side"
          .format(report.funding_implied_usd))
    print("  fees paid           ${:+,.4f}".format(report.fees_usd))
    print("  unrealised price    ${:+,.4f}   (not forecast; the variance, not "
          "the edge)".format(report.unrealized_usd))
    if report.realised_funding_bps_per_day is not None:
        print("  realised rate       {:+.3f} bps/day on gross".format(
            report.realised_funding_bps_per_day))
        if report.planned_funding_bps_per_day > 0:
            print("  planned rate        {:+.3f} bps/day".format(
                report.planned_funding_bps_per_day))

    print()
    if not report.alerts:
        print("No alerts.")
        return
    print("ALERTS")
    for alert in report.alerts:
        print("  [{:<8}] {:<18} {}".format(alert.level, alert.code, alert.message))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--production", action="store_true",
                        help="Read the live account rather than demo.")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--hold-days", type=int, default=7)
    parser.add_argument("--planned-funding-bps-per-day", type=Decimal,
                        default=Decimal("0"),
                        help="what the plan forecast, so the baseline can "
                             "score against it")
    parser.add_argument("--archive-epoch", action="store_true",
                        help="the book has been rebalanced: file the old "
                             "baseline and freeze a new one on the next run")
    args = parser.parse_args(argv)

    from blofin.client import Client
    from blofin.rest_market import MarketAPI

    base_url = PRODUCTION_BASE_URL if args.production else DEMO_BASE_URL
    environment = "production" if args.production else "demo"

    api_key = os.environ.get("API_KEY")
    secret = os.environ.get("SECRET")
    passphrase = os.environ.get("PASSPHRASE")
    if not (api_key and secret and passphrase):
        raise SystemExit("API_KEY, SECRET and PASSPHRASE must be set in .env.")

    client = Client(apiKey=api_key, apiSecret=secret, passphrase=passphrase,
                    baseUrl=base_url)
    reader = BlofinReader(client, MarketAPI(client))

    now_ms = int(time.time() * 1000)
    snapshot = read_snapshot(reader, reader.contract_values(), now_ms=now_ms)
    directory = args.state / environment

    if args.archive_epoch:
        try:
            existing = load_or_freeze(snapshot, directory,
                                      hold_days=args.hold_days)
            path = archive_epoch(directory, existing, snapshot)
            print("Archived the epoch to " + str(path))
            print("The next run freezes a new baseline against the current book.")
            return 0
        except Exception as exc:                       # noqa: BLE001
            raise SystemExit("Could not archive: " + str(exc))

    baseline = load_or_freeze(
        snapshot, directory, hold_days=args.hold_days,
        planned_funding_bps_per_day=args.planned_funding_bps_per_day)

    fills = reader.fills(since_ms=baseline.opened_ms)
    report = compare(baseline, snapshot,
                     fees_usd=fees_since(fills),
                     implied_usd=implied_funding(reader, snapshot,
                                                 since_ms=baseline.opened_ms),
                     limits=RiskLimits())
    show(report, snapshot, environment)

    if report.epoch_ended:
        print()
        print("This book is not the one the baseline was frozen against. Run "
              "with\n  --archive-epoch  to file it and start scoring the new "
              "one.")
    return 2 if report.critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
