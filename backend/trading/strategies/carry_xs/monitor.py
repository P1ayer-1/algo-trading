"""Reads the book back and scores it. NO PATH TO placeOrder, at all.

`plan.py` sizes the book and `execute.py` puts it on. Both finish in seconds.
What they produce is a twelve-leg position that runs for a week, and until now
nothing looked at it again: the plan printed "+9.2 bps expected", the executor
printed the fills, and the numbers that would settle whether either was true -
funding actually credited, and the neutrality actually carried - were never
read back.

This does not import the broker. `execute.py`'s `BlofinPerpBroker` can place
orders, and a monitor that held one would be one typo away from being an
executor; so this defines its own `Reader`, which has four methods and none of
them writes. The contract in `strategies/__init__.py` says a monitor is the one
you can run at 3am without reading the source first to check what it might do,
and that is only true if it is structurally unable to do anything.

What a cross-margined book makes easy, and what it makes hard
-------------------------------------------------------------
Easy: there is ONE margin ratio for the whole account rather than twelve, and
the exchange computes it knowing the legs offset. Measured on the live demo
book, all eight positions report the identical `marginRatio`, which is what
confirms they share a pool. So the risk question has a single answer and it is
read rather than modelled - unlike the plan's per-leg `solo liq`, which assumes
isolated margin and is deliberately a conservative bound.

Hard: a cross account's `realizedPnl` is per position but its margin is not, so
a leg cannot be scored in isolation. Everything below is therefore
book-level, and per-leg numbers are reported for inspection rather than used
for verdicts.

Funding, and why it is derived rather than read
-----------------------------------------------
Same constraint `strategies/carry` hit and for the same reason: this API
version publishes no account-bills endpoint, so there is no line item saying
"funding, +$0.14". `realizedPnl` on a position accumulates fees, closed-trade
PnL and funding together, and on a book that has not closed anything the
closed-trade term is zero:

    funding = sum(realizedPnl) + sum(fees)

**A derived number gets a control.** The same funding is estimated
independently from the public funding-rate history - every settlement since
the book opened, times each leg's notional and signed by its side - and the two
are compared. They will not agree exactly, because the implied one prices every
period at today's notional rather than the notional at each settlement. When
they disagree badly the report says so rather than quietly preferring the one
it computed itself.

The baseline, and why a rebalancing book needs epochs
------------------------------------------------------
The two-leg carry freezes one baseline and holds it for a month. This book is
rebalanced weekly, so a single frozen baseline would be describing a position
that no longer exists by the second week. The baseline is therefore frozen per
EPOCH: it records the set of legs it was taken against, and when that set
changes materially the report says the epoch ended rather than scoring new
positions against an old forecast. Old epochs are archived, never rewritten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from trading.risk import RiskLimits

ZERO = Decimal("0")

# A position is stamped AFTER the fill that opened it. Measured on BloFin
# 2026-09-12 across ten legs: 20-35 ms late, worst 35 (ZEC). Only used when a
# venue reports no position id to match on; five seconds is 140x the worst
# observed skew and still far short of any real gap between a close and a
# re-open on the same instrument.
CREATE_TIME_SKEW_MS = 5_000


def _d(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:                                  # noqa: BLE001
        return Decimal(default)


class Reader(Protocol):
    """Read-only. There is no `place` anything here, and that is the point."""

    def positions(self) -> List[Dict[str, Any]]: ...

    def balance(self) -> Dict[str, Any]: ...

    def fills(self, *, since_ms: int) -> List[Dict[str, Any]]: ...

    def funding_history(self, inst_id: str, *,
                        since_ms: int) -> List[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# What is on the exchange
# ---------------------------------------------------------------------------


@dataclass
class Leg:
    inst_id: str
    contracts: Decimal            # signed
    mark: Decimal
    notional_usd: Decimal         # signed, so the book's net is a sum
    unrealized_pnl_usd: Decimal
    realized_pnl_usd: Decimal
    liquidation_price: Optional[Decimal]
    created_ms: int
    # The exchange's own id for this position. Fills carry it too, which is the
    # only EXACT way to say which fees belong to a position still open - see
    # `fees_since`, where matching on timestamps instead was wrong by $4.05.
    position_id: str = ""

    @property
    def side(self) -> str:
        return "long" if self.contracts > 0 else "short"


@dataclass
class Snapshot:
    """The book as the exchange reports it, at one instant."""

    taken_ms: int
    legs: List[Leg] = field(default_factory=list)
    total_equity_usd: Decimal = ZERO
    margin_ratio: Optional[Decimal] = None

    @property
    def gross_usd(self) -> Decimal:
        return sum((abs(leg.notional_usd) for leg in self.legs), ZERO)

    @property
    def net_usd(self) -> Decimal:
        return sum((leg.notional_usd for leg in self.legs), ZERO)

    @property
    def unrealized_usd(self) -> Decimal:
        return sum((leg.unrealized_pnl_usd for leg in self.legs), ZERO)

    @property
    def realized_usd(self) -> Decimal:
        return sum((leg.realized_pnl_usd for leg in self.legs), ZERO)

    @property
    def instruments(self) -> List[str]:
        return sorted(leg.inst_id for leg in self.legs)


def read_snapshot(reader: Reader, contract_values: Dict[str, Decimal],
                  *, now_ms: int) -> Snapshot:
    """Positions and equity, in one reading.

    Notional is `contracts x contract value x mark`, signed. Every other module
    here has had to learn that a contract is a different amount of money on
    every instrument; this one is told.
    """
    snapshot = Snapshot(taken_ms=now_ms)
    for row in reader.positions():
        inst_id = str(row.get("instId") or "")
        contracts = _d(row.get("positions"))
        if not inst_id or contracts == 0:
            continue
        mark = _d(row.get("markPrice"))
        value = contract_values.get(inst_id, Decimal("1"))
        snapshot.legs.append(Leg(
            inst_id=inst_id, contracts=contracts, mark=mark,
            notional_usd=contracts * value * mark,
            unrealized_pnl_usd=_d(row.get("unrealizedPnl")),
            realized_pnl_usd=_d(row.get("realizedPnl")),
            liquidation_price=(_d(row.get("liquidationPrice"))
                               if row.get("liquidationPrice") else None),
            created_ms=int(_d(row.get("createTime"))),
            position_id=str(row.get("positionId") or ""),
        ))
        # Cross margin gives every position the SAME account-level ratio, so
        # reading it from any one of them is reading the account's.
        if snapshot.margin_ratio is None and row.get("marginRatio"):
            snapshot.margin_ratio = _d(row.get("marginRatio"))

    details = (reader.balance() or {}).get("details") or []
    for row in details:
        if str(row.get("currency")) == "USDT":
            snapshot.total_equity_usd = _d(row.get("equity"))
            break
    return snapshot


def fees_since(fills: Iterable[Dict[str, Any]], legs: Sequence["Leg"] = (),
               *, since_ms: int = 0) -> Decimal:
    """Fees paid ON THE POSITIONS THAT ARE STILL OPEN, per leg.

    Signed as the venue reports them, not through `abs()`: a maker rebate
    reported as a negative fee has to cancel correctly in the derivation, and
    forcing a sign here would turn a credit into a charge.

    The per-leg window is the whole correctness argument. `funding =
    sum(realizedPnl) + fees` only holds when the two terms cover the same
    trades, and `realizedPnl` is a property of a position that VANISHES when
    the position closes. Summing every fill in the window instead charges the
    book for positions whose matching PnL is already gone.

    Measured on the live demo book 2026-09-12, which is why this is not
    hypothetical. The baseline had been frozen against an 8-leg book; that book
    was closed by hand and a 10-leg one opened. Scoring the new positions
    against the old baseline's window swept in $4.4844 of fees from the
    previous epoch - 7 opens, a repair and 8 closes - and the derivation
    reported +$4.05 of funding on a book 72 minutes old, a realised +87.107
    bps/day against a planned +6.275. Fourteen times the forecast, and every
    cent of it stale fees.

    Bounding each leg at its own `createTime` makes the window the positions'
    rather than the baseline's, so the number is right even when the baseline
    is stale - which is exactly when nobody is checking. Same bound
    `implied_funding` uses, for the same reason. On that book it returns
    $5.7981 against `sum(realizedPnl)` of -$5.7981: zero funding, which is the
    true answer 72 minutes in with no settlement crossed, and it agrees with
    the public-rate control to the cent.

    Matched on `positionId`, which both a position and its fills carry, so the
    question "did this fee belong to a position that is still open" is answered
    by an identity and not by a clock. The first attempt at this bounded each
    leg by its own `createTime` instead and returned $0.0000 on that same book,
    because every position is stamped 20-35 ms AFTER the fill that opened it -
    BCH 27 ms, ZEC 35 ms, measured - so `ts >= created_ms` excluded every
    opening fill. Two clocks compared at millisecond precision, which is the
    same bug shape as the funding settlements that print milliseconds late and
    landed on the wrong side of midnight in `panel_daily.py`.

    The timestamp path survives only as a fallback for a venue that reports no
    position id, with enough tolerance to absorb that skew.

    With no legs supplied there is nothing to match against, so it sums the
    window as given.
    """
    if not legs:
        total = ZERO
        for fill in fills:
            total += _d(fill.get("fee"))
        return total

    ids = {leg.position_id for leg in legs if leg.position_id}
    total = ZERO

    if ids:
        for fill in fills:
            # A fill whose position is gone took its realizedPnl with it, so
            # counting the fee would be one side of a cancellation.
            if str(fill.get("positionId") or "") in ids:
                total += _d(fill.get("fee"))
        return total

    starts: Dict[str, int] = {}
    for leg in legs:
        starts[leg.inst_id] = max(int(since_ms),
                                  int(leg.created_ms) - CREATE_TIME_SKEW_MS)
    for fill in fills:
        start = starts.get(str(fill.get("instId") or ""))
        if start is None or int(_d(fill.get("ts"))) < start:
            continue
        total += _d(fill.get("fee"))
    return total


def implied_funding(reader: Reader, snapshot: Snapshot, *,
                    since_ms: int) -> Decimal:
    """What the public funding history says the book should have collected.

    The control on the derived number. Signed by side: a LONG pays a positive
    rate, a SHORT receives it, so the contribution is `-sign(position) * rate *
    notional`. Priced at today's notional for every settlement, which is why it
    is a control and not the answer.
    """
    total = ZERO
    for leg in snapshot.legs:
        rates = reader.funding_history(leg.inst_id,
                                       since_ms=max(since_ms, leg.created_ms))
        accrued = ZERO
        for row in rates:
            try:
                stamp = int(_d(row.get("fundingTime")))
            except Exception:                          # noqa: BLE001
                continue
            if stamp <= max(since_ms, leg.created_ms):
                continue
            accrued += _d(row.get("fundingRate"))
        direction = Decimal("-1") if leg.contracts > 0 else Decimal("1")
        total += direction * accrued * abs(leg.notional_usd)
    return total


# ---------------------------------------------------------------------------
# The baseline, per epoch
# ---------------------------------------------------------------------------


@dataclass
class Baseline:
    """Frozen the first time a book is seen, and never rewritten.

    `instruments` is what makes an epoch: this book is rebalanced, so a
    baseline that outlived its legs would be scoring new positions against an
    old forecast. When the set changes the epoch has ended and the report says
    so instead.
    """

    opened_ms: int
    instruments: List[str]
    gross_usd: Decimal
    equity_usd: Decimal
    planned_funding_bps_per_day: Decimal = ZERO
    planned_round_trip_bps: Decimal = ZERO
    hold_days: int = 7

    def to_json(self) -> Dict[str, Any]:
        return {
            "opened_ms": self.opened_ms,
            "instruments": list(self.instruments),
            "gross_usd": str(self.gross_usd),
            "equity_usd": str(self.equity_usd),
            "planned_funding_bps_per_day": str(self.planned_funding_bps_per_day),
            "planned_round_trip_bps": str(self.planned_round_trip_bps),
            "hold_days": self.hold_days,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "Baseline":
        return cls(
            opened_ms=int(payload["opened_ms"]),
            instruments=list(payload.get("instruments") or []),
            gross_usd=_d(payload.get("gross_usd")),
            equity_usd=_d(payload.get("equity_usd")),
            planned_funding_bps_per_day=_d(
                payload.get("planned_funding_bps_per_day")),
            planned_round_trip_bps=_d(payload.get("planned_round_trip_bps")),
            hold_days=int(payload.get("hold_days") or 7),
        )


def baseline_path(directory: Path) -> Path:
    return directory / "baseline.json"


def load_or_freeze(snapshot: Snapshot, directory: Path, *,
                   planned_funding_bps_per_day: Decimal = ZERO,
                   planned_round_trip_bps: Decimal = ZERO,
                   hold_days: int = 7) -> Baseline:
    """Read the frozen baseline, or freeze this snapshot as one.

    Freezing happens once per epoch and the file is never rewritten, so the
    forecast the book is being scored against stops moving. A baseline that
    updated itself every reading would always agree with the position, which
    is the one thing a scoreboard must not do.
    """
    path = baseline_path(directory)
    if path.exists():
        try:
            return Baseline.from_json(json.loads(path.read_text()))
        except (ValueError, KeyError, TypeError):
            pass

    opened = min((leg.created_ms for leg in snapshot.legs),
                 default=snapshot.taken_ms)
    baseline = Baseline(
        opened_ms=opened, instruments=snapshot.instruments,
        gross_usd=snapshot.gross_usd, equity_usd=snapshot.total_equity_usd,
        planned_funding_bps_per_day=planned_funding_bps_per_day,
        planned_round_trip_bps=planned_round_trip_bps, hold_days=hold_days)
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline.to_json(), indent=1))
    return baseline


def archive_epoch(directory: Path, baseline: Baseline, snapshot: Snapshot) -> Path:
    """Move a finished epoch aside so the next one can freeze its own.

    Appending rather than overwriting: the record of what a book actually did
    is the only thing that can ever settle whether the strategy works, and it
    is the first thing a rewrite would destroy.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(baseline.opened_ms / 1000.0,
                                   timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = directory / ("epoch-" + stamp + ".json")
    path.write_text(json.dumps({
        "baseline": baseline.to_json(),
        "closed_ms": snapshot.taken_ms,
        "closing_instruments": snapshot.instruments,
    }, indent=1))
    baseline_path(directory).unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    level: str          # "info" | "warn" | "critical"
    code: str
    message: str


@dataclass
class BookReport:
    """Realised against predicted. Conforms to the `Report` Protocol."""

    inst_id: str = "carry_xs"
    open: bool = True
    epoch_ended: bool = False
    days_elapsed: Decimal = ZERO

    legs: int = 0
    gross_usd: Decimal = ZERO
    net_usd: Decimal = ZERO
    net_frac: Decimal = ZERO
    total_equity_usd: Decimal = ZERO
    margin_ratio: Optional[Decimal] = None

    funding_derived_usd: Decimal = ZERO
    funding_implied_usd: Decimal = ZERO
    fees_usd: Decimal = ZERO
    unrealized_usd: Decimal = ZERO
    realised_funding_bps_per_day: Optional[Decimal] = None
    planned_funding_bps_per_day: Decimal = ZERO

    alerts: List[Alert] = field(default_factory=list)

    @property
    def critical(self) -> bool:
        return any(alert.level == "critical" for alert in self.alerts)

    def add(self, level: str, code: str, message: str) -> None:
        self.alerts.append(Alert(level=level, code=code, message=message))


def compare(baseline: Baseline, snapshot: Snapshot, *, fees_usd: Decimal,
            implied_usd: Decimal, limits: Optional[RiskLimits] = None,
            net_tolerance_frac: float = 0.02) -> BookReport:
    """Score the book against its baseline, and raise what needs acting on.

    Everything here is realised. The one comparison to the plan is the funding
    RATE, because that is the only part of the forecast the book can be held to
    - the price leg was never forecast and step 9q is explicit that it is the
    variance rather than the edge.
    """
    limits = limits or RiskLimits()
    report = BookReport()
    report.legs = len(snapshot.legs)
    report.gross_usd = snapshot.gross_usd
    report.net_usd = snapshot.net_usd
    report.total_equity_usd = snapshot.total_equity_usd
    report.margin_ratio = snapshot.margin_ratio
    report.unrealized_usd = snapshot.unrealized_usd
    report.planned_funding_bps_per_day = baseline.planned_funding_bps_per_day

    elapsed_ms = max(0, snapshot.taken_ms - baseline.opened_ms)
    report.days_elapsed = Decimal(elapsed_ms) / Decimal(86_400_000)

    if not snapshot.legs:
        report.open = False
        report.add("info", "flat", "No open positions. The book is not on.")
        return report

    if snapshot.gross_usd > 0:
        report.net_frac = abs(snapshot.net_usd) / snapshot.gross_usd

    # Funding: derived, then controlled.
    report.fees_usd = fees_usd
    report.funding_derived_usd = snapshot.realized_usd + fees_usd
    report.funding_implied_usd = implied_usd
    if report.days_elapsed > 0 and snapshot.gross_usd > 0:
        report.realised_funding_bps_per_day = (
            report.funding_derived_usd / snapshot.gross_usd
            * Decimal(10_000) / report.days_elapsed)

    _raise_alerts(report, baseline, snapshot, limits=limits,
                  net_tolerance_frac=net_tolerance_frac)
    return report


def _raise_alerts(report: BookReport, baseline: Baseline, snapshot: Snapshot, *,
                  limits: RiskLimits, net_tolerance_frac: float) -> None:
    current = set(snapshot.instruments)
    expected = set(baseline.instruments)

    if current != expected:
        report.epoch_ended = True
        gone, added = sorted(expected - current), sorted(current - expected)
        report.add(
            "info", "epoch-ended",
            "The book is not the one this baseline was frozen against"
            + (" (gone: " + ", ".join(gone) + ")" if gone else "")
            + (" (new: " + ", ".join(added) + ")" if added else "")
            + ". Scoring stops here; archive the epoch and freeze a new one.")
        if gone and not added:
            report.add(
                "critical", "legs-vanished",
                "{} leg(s) disappeared without being replaced: {}. A book "
                "missing one side is directional, and a position that vanished "
                "on its own was liquidated.".format(len(gone), ", ".join(gone)))

    if report.net_frac > Decimal(str(net_tolerance_frac)):
        report.add(
            "critical", "not-neutral",
            "Net exposure ${:,.2f} is {:.2%} of ${:,.2f} gross, over the {:.2%} "
            "tolerance. A dollar-neutral strategy carrying this much beta is a "
            "different strategy.".format(
                snapshot.net_usd, report.net_frac, snapshot.gross_usd,
                net_tolerance_frac))

    # The margin ratio is the account's, because the legs share a pool. BloFin
    # reports it as a percentage where larger is safer.
    if snapshot.margin_ratio is not None and snapshot.margin_ratio < Decimal("150"):
        report.add(
            "critical", "margin",
            "Account margin ratio {:.1f}. The legs share one cross pool, so "
            "this is the whole book's distance from liquidation and not one "
            "leg's.".format(snapshot.margin_ratio))
    elif snapshot.margin_ratio is not None and snapshot.margin_ratio < Decimal("300"):
        report.add(
            "warn", "margin",
            "Account margin ratio {:.1f}, getting thin.".format(
                snapshot.margin_ratio))

    derived, implied = report.funding_derived_usd, report.funding_implied_usd
    gap = abs(derived - implied)
    if gap > max(Decimal("1"), abs(implied) * Decimal("0.5")):
        report.add(
            "warn", "funding-disagrees",
            "Derived funding ${:+,.4f} and the public-rate estimate ${:+,.4f} "
            "disagree. The derivation subtracts fees from realizedPnl and the "
            "estimate prices every settlement at today's notional, so they "
            "never match exactly - but not by this much.".format(
                derived, implied))

    if (report.realised_funding_bps_per_day is not None
            and report.days_elapsed > Decimal("1")):
        realised = report.realised_funding_bps_per_day
        # Negative funding is checked WITHOUT reference to the plan. An earlier
        # version nested it under "was a forecast supplied", so a book run
        # without --planned-funding-bps-per-day - the default - could pay out
        # indefinitely and raise nothing. The one thing this strategy cannot
        # survive is being on the wrong side of the cash flow, and noticing
        # that must not depend on a caller having passed an optional argument.
        if realised < 0:
            report.add(
                "critical", "funding-negative",
                "Realised funding is NEGATIVE at {:.3f} bps/day. The book is "
                "paying to hold the position it was opened to be paid for."
                .format(realised))
        elif (baseline.planned_funding_bps_per_day > 0
                and realised < baseline.planned_funding_bps_per_day / 2):
            report.add(
                "warn", "funding-short",
                "Realised funding {:.3f} bps/day against a planned {:.3f}. The "
                "carry spread the book was opened on has narrowed.".format(
                    realised, baseline.planned_funding_bps_per_day))

    if report.days_elapsed > Decimal(baseline.hold_days):
        report.add(
            "info", "due",
            "Held {:.1f} days against a {}-day rebalance. The scores in step "
            "9q are for non-overlapping holds of that length.".format(
                report.days_elapsed, baseline.hold_days))

    for leg in snapshot.legs:
        if leg.liquidation_price is None or leg.mark <= 0:
            continue
        distance = abs(leg.mark - leg.liquidation_price) / leg.mark
        if distance < limits.min_open_liquidation_buffer_pct:
            report.add(
                "critical", "leg-liquidation",
                "{} is {:.1%} from its liquidation price of {}, inside the "
                "{:.1%} buffer.".format(
                    leg.inst_id, distance, leg.liquidation_price,
                    limits.min_open_liquidation_buffer_pct))
