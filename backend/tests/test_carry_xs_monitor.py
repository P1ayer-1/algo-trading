"""The monitor: read-only, and it must not agree with itself.

A scoreboard that updates its own forecast always says the position is on
track, so the baseline is frozen once and the derived funding gets an
independent control. The other half of this file is the safety property: the
monitor cannot place an order, structurally, and that is asserted rather than
assumed.
"""

import json
from decimal import Decimal

import pytest

from trading.risk import RiskLimits
from trading.strategies import Report
from trading.strategies.carry_xs.monitor import (
    Baseline,
    BookReport,
    archive_epoch,
    compare,
    fees_since,
    implied_funding,
    load_or_freeze,
    read_snapshot,
)

DAY_MS = 86_400_000
T0 = 1_789_000_000_000


class FakeReader:
    def __init__(self, positions=None, equity="50000", fills=None, rates=None):
        self._positions = list(positions or [])
        self._equity = equity
        self._fills = list(fills or [])
        self._rates = dict(rates or {})

    def positions(self):
        return self._positions

    def balance(self):
        return {"details": [{"currency": "USDT", "equity": self._equity}]}

    def fills(self, *, since_ms):
        return self._fills

    def funding_history(self, inst_id, *, since_ms):
        return self._rates.get(inst_id, [])


def position(inst_id, contracts, *, mark="100", unreal="0", realised="0",
             liq=None, created=T0, margin_ratio="2000", position_id=None):
    row = {"instId": inst_id, "positions": str(contracts), "markPrice": mark,
           "unrealizedPnl": unreal, "realizedPnl": realised,
           "liquidationPrice": str(liq) if liq is not None else "",
           "createTime": str(created), "marginRatio": margin_ratio}
    if position_id is not None:
        row["positionId"] = str(position_id)
    return row


VALUES = {name: Decimal("1") for name in ("A-USDT", "B-USDT", "C-USDT", "D-USDT")}


def snapshot_of(positions, *, now=T0 + DAY_MS, **kwargs):
    return read_snapshot(FakeReader(positions, **kwargs), VALUES, now_ms=now)


def neutral_book():
    return [position("A-USDT", 5), position("B-USDT", 5),
            position("C-USDT", -5), position("D-USDT", -5)]


def a_baseline(**kwargs):
    defaults = dict(opened_ms=T0,
                    instruments=["A-USDT", "B-USDT", "C-USDT", "D-USDT"],
                    gross_usd=Decimal("2000"), equity_usd=Decimal("50000"),
                    hold_days=7)
    defaults.update(kwargs)
    return Baseline(**defaults)


# ---------------------------------------------------------------------------
# The safety property
# ---------------------------------------------------------------------------


def test_the_monitor_cannot_place_an_order():
    """`execute.py`'s broker can send orders and this file must not hold one.

    The contract in `strategies/__init__.py` says a monitor is the one you can
    run at 3am without reading the source first, and that is only true if it is
    structurally unable to do anything.
    """
    import ast

    import trading.strategies.carry_xs.monitor as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())

    # Names referenced in CODE. A substring search over the source would match
    # the docstring that explains this very rule, so it would pass or fail on
    # the prose rather than on the behaviour.
    referenced = {node.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute)}
    referenced |= {node.id for node in ast.walk(tree)
                   if isinstance(node, ast.Name)}
    assert "placeOrder" not in referenced
    assert "place_perp" not in referenced

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {alias.name for alias in node.names}
    assert not any("broker" in name.lower() for name in imported), imported


def test_a_book_report_is_a_report():
    assert isinstance(BookReport(), Report)


# ---------------------------------------------------------------------------
# Reading the book
# ---------------------------------------------------------------------------


def test_notional_is_signed_so_the_net_is_a_sum():
    snapshot = snapshot_of(neutral_book())
    assert snapshot.gross_usd == Decimal("2000")
    assert snapshot.net_usd == Decimal("0")


def test_notional_uses_contract_value():
    """The same multiplier trap the executor fell into. A BTC contract is
    0.001 BTC; reading position size as money is wrong by a thousand."""
    reader = FakeReader([position("BTC-USDT", 10, mark="80000")])
    snapshot = read_snapshot(reader, {"BTC-USDT": Decimal("0.001")}, now_ms=T0)
    assert snapshot.legs[0].notional_usd == Decimal("800")


def test_the_margin_ratio_is_read_once_because_it_is_the_accounts():
    """Under cross margin every position reports the SAME account-level ratio.

    Measured on the live demo book: all eight positions carried an identical
    `marginRatio`. Averaging them or taking a per-leg minimum would invent a
    per-leg risk number the exchange does not compute.
    """
    snapshot = snapshot_of(neutral_book())
    assert snapshot.margin_ratio == Decimal("2000")


def test_fees_keep_their_sign_so_a_rebate_still_cancels():
    """Forcing a sign would turn a maker credit into a charge and bias the
    derived funding by twice the rebate."""
    assert fees_since([{"fee": "-0.5"}, {"fee": "1.5"}]) == Decimal("1.0")


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


def test_funding_is_realised_pnl_plus_fees():
    """No bills endpoint exists, so funding falls out of the difference.

    `realizedPnl` accumulates fees, closed-trade PnL and funding; on a book
    that has closed nothing the middle term is zero.
    """
    positions = [position("A-USDT", 5, realised="-1.20")]
    report = compare(a_baseline(instruments=["A-USDT"]), snapshot_of(positions),
                     fees_usd=Decimal("1.50"), implied_usd=Decimal("0.30"))
    assert report.funding_derived_usd == Decimal("0.30")


def test_the_implied_control_pays_shorts_and_charges_longs():
    """A long PAYS a positive funding rate and a short RECEIVES it.

    Getting this backwards makes the control agree with a book that is losing
    money, which is worse than having no control.
    """
    rates = {"A-USDT": [{"fundingTime": str(T0 + 1000), "fundingRate": "0.001"}],
             "C-USDT": [{"fundingTime": str(T0 + 1000), "fundingRate": "0.001"}]}
    reader = FakeReader([position("A-USDT", 5), position("C-USDT", -5)],
                        rates=rates)
    snapshot = read_snapshot(reader, VALUES, now_ms=T0 + DAY_MS)
    long_only = implied_funding(
        FakeReader([position("A-USDT", 5)], rates=rates),
        read_snapshot(FakeReader([position("A-USDT", 5)]), VALUES, now_ms=T0),
        since_ms=T0 - 1)
    assert long_only < 0
    # Long and short of the same size at the same rate cancel exactly.
    assert implied_funding(reader, snapshot, since_ms=T0 - 1) == Decimal("0")


def test_a_derived_number_that_disagrees_with_its_control_is_flagged():
    report = compare(a_baseline(instruments=["A-USDT"]),
                     snapshot_of([position("A-USDT", 5, realised="20")]),
                     fees_usd=Decimal("0"), implied_usd=Decimal("0.10"))
    assert any(alert.code == "funding-disagrees" for alert in report.alerts)


# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------


def test_the_baseline_is_frozen_once_and_never_rewritten(tmp_path):
    """A baseline that updated every reading would always agree with the
    position, which is the one thing a scoreboard must not do."""
    first = load_or_freeze(snapshot_of(neutral_book()), tmp_path)
    bigger = neutral_book() + [position("A-USDT", 50)]
    second = load_or_freeze(snapshot_of(bigger), tmp_path)
    assert second.gross_usd == first.gross_usd
    assert second.instruments == first.instruments


def test_a_changed_book_ends_the_epoch_rather_than_scoring_against_the_old_one():
    """This book is REBALANCED, so a baseline that outlived its legs would be
    scoring new positions against an old forecast."""
    rebalanced = [position("A-USDT", 5), position("B-USDT", 5),
                  position("C-USDT", -5), position("E-USDT", -5)]
    report = compare(a_baseline(), snapshot_of(rebalanced),
                     fees_usd=Decimal("0"), implied_usd=Decimal("0"))
    assert report.epoch_ended
    assert any(alert.code == "epoch-ended" for alert in report.alerts)


def test_legs_that_vanish_without_replacement_are_critical():
    """A position that disappeared on its own was liquidated, and a book
    missing one side is directional."""
    report = compare(a_baseline(), snapshot_of(
        [position("A-USDT", 5), position("B-USDT", 5), position("C-USDT", -5)]),
        fees_usd=Decimal("0"), implied_usd=Decimal("0"))
    assert report.critical
    assert any(alert.code == "legs-vanished" for alert in report.alerts)


def test_archiving_an_epoch_keeps_it_and_clears_the_baseline(tmp_path):
    baseline = load_or_freeze(snapshot_of(neutral_book()), tmp_path)
    path = archive_epoch(tmp_path, baseline, snapshot_of(neutral_book()))
    assert path.exists()
    assert json.loads(path.read_text())["baseline"]["instruments"]
    assert not (tmp_path / "baseline.json").exists()


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


def test_a_book_that_has_gone_directional_is_critical():
    lopsided = [position("A-USDT", 10), position("B-USDT", 5),
                position("C-USDT", -5), position("D-USDT", -5)]
    report = compare(a_baseline(), snapshot_of(lopsided),
                     fees_usd=Decimal("0"), implied_usd=Decimal("0"))
    assert report.critical
    assert any(alert.code == "not-neutral" for alert in report.alerts)


def test_a_thin_account_margin_ratio_is_critical():
    thin = [position(name, size, margin_ratio="80")
            for name, size in (("A-USDT", 5), ("B-USDT", 5),
                               ("C-USDT", -5), ("D-USDT", -5))]
    report = compare(a_baseline(), snapshot_of(thin),
                     fees_usd=Decimal("0"), implied_usd=Decimal("0"))
    assert any(alert.code == "margin" and alert.level == "critical"
               for alert in report.alerts)


def test_negative_realised_funding_is_critical():
    """The book exists to be PAID to hold these positions."""
    losing = [position("A-USDT", 5, realised="-4"), position("B-USDT", 5),
              position("C-USDT", -5), position("D-USDT", -5)]
    report = compare(a_baseline(), snapshot_of(losing, now=T0 + 3 * DAY_MS),
                     fees_usd=Decimal("0"), implied_usd=Decimal("-4"))
    assert report.realised_funding_bps_per_day < 0
    assert any(alert.code == "funding-negative" for alert in report.alerts)


def test_funding_far_under_the_plan_is_a_warning_not_a_crisis():
    report = compare(
        a_baseline(planned_funding_bps_per_day=Decimal("6")),
        snapshot_of([position("A-USDT", 5, realised="0.02"),
                     position("B-USDT", 5), position("C-USDT", -5),
                     position("D-USDT", -5)], now=T0 + 3 * DAY_MS),
        fees_usd=Decimal("0"), implied_usd=Decimal("0.02"))
    assert any(alert.code == "funding-short" and alert.level == "warn"
               for alert in report.alerts)
    assert not report.critical


def test_a_leg_near_its_liquidation_price_is_critical():
    close = [position("A-USDT", 5, mark="100", liq="98"), position("B-USDT", 5),
             position("C-USDT", -5), position("D-USDT", -5)]
    report = compare(a_baseline(), snapshot_of(close),
                     fees_usd=Decimal("0"), implied_usd=Decimal("0"),
                     limits=RiskLimits())
    assert any(alert.code == "leg-liquidation" for alert in report.alerts)


def test_a_held_book_past_its_rebalance_is_noted():
    report = compare(a_baseline(), snapshot_of(neutral_book(),
                                               now=T0 + 9 * DAY_MS),
                     fees_usd=Decimal("0"), implied_usd=Decimal("0"))
    assert any(alert.code == "due" for alert in report.alerts)
    assert not report.critical


def test_an_empty_account_reports_flat_and_not_a_crisis():
    report = compare(a_baseline(), snapshot_of([]),
                     fees_usd=Decimal("0"), implied_usd=Decimal("0"))
    assert not report.open
    assert not report.critical


# ---------------------------------------------------------------------------
# The fee window, which a stale baseline made wrong on the live book
# ---------------------------------------------------------------------------


def test_fees_exclude_positions_the_book_no_longer_holds():
    """`funding = sum(realizedPnl) + fees` only holds when both terms cover the
    same trades, and `realizedPnl` VANISHES when a position closes.

    Measured on the live demo book 2026-09-12. The baseline had been frozen
    against an 8-leg book; that book was closed by hand and a 10-leg one
    opened. Summing every fill in the baseline's window swept in $4.4844 of
    fees from the previous epoch and reported +$4.05 of funding on a book 72
    minutes old - a realised +87.107 bps/day against a planned +6.275, fourteen
    times the forecast, every cent of it stale fees.
    """
    reader = FakeReader(positions=[
        position("A-USDT", "10", created=T0 + 40 * 60_000, realised="-3"),
        position("B-USDT", "-10", created=T0 + 40 * 60_000, realised="-2.798122"),
    ])
    snapshot = read_snapshot(reader, VALUES, now_ms=T0 + 30 * DAY_MS)

    fills = [
        # The previous epoch: opened, repaired and closed before these legs
        # existed. Their realizedPnl went with the positions.
        {"instId": "C-USDT", "ts": str(T0), "fee": "2.2"},
        {"instId": "D-USDT", "ts": str(T0 + 60_000), "fee": "2.2844"},
        # A name the book holds AGAIN, but traded before this position opened.
        {"instId": "A-USDT", "ts": str(T0 + 60_000), "fee": "1.5"},
        # The current epoch.
        {"instId": "A-USDT", "ts": str(T0 + 40 * 60_000), "fee": "3"},
        {"instId": "B-USDT", "ts": str(T0 + 40 * 60_000), "fee": "2.798122"},
    ]

    contaminated = fees_since(fills)
    assert contaminated == Decimal("11.782522")

    fees = fees_since(fills, snapshot.legs, since_ms=T0)
    assert fees == Decimal("5.798122")

    # Which is the whole point: it cancels realizedPnl exactly, so the
    # derivation returns zero - the true answer before any settlement.
    realised = sum(leg.realized_pnl_usd for leg in snapshot.legs)
    assert realised + fees == Decimal("0")


def test_the_fee_window_is_the_positions_own_not_the_baselines():
    """A stale baseline is exactly when nobody is checking.

    Bounding each leg at its own `createTime` rather than at the baseline's
    `opened_ms` makes the derivation right even when the baseline describes a
    book that no longer exists.
    """
    reader = FakeReader(positions=[
        position("A-USDT", "10", created=T0 + 10 * DAY_MS, realised="-1"),
    ])
    snapshot = read_snapshot(reader, VALUES, now_ms=T0 + 30 * DAY_MS)
    fills = [
        {"instId": "A-USDT", "ts": str(T0 + 1 * DAY_MS), "fee": "9"},
        {"instId": "A-USDT", "ts": str(T0 + 10 * DAY_MS), "fee": "1"},
    ]
    # A baseline ten days stale would otherwise sweep in the older fill.
    assert fees_since(fills, snapshot.legs, since_ms=T0) == Decimal("1")


def test_the_fee_window_still_respects_a_baseline_later_than_the_position():
    """`since_ms` is a floor, not a suggestion: a leg opened before the epoch
    began is scored from the epoch, not from the position."""
    reader = FakeReader(positions=[
        position("A-USDT", "10", created=T0, realised="-1"),
    ])
    snapshot = read_snapshot(reader, VALUES, now_ms=T0 + 30 * DAY_MS)
    fills = [
        {"instId": "A-USDT", "ts": str(T0), "fee": "9"},
        {"instId": "A-USDT", "ts": str(T0 + 5 * DAY_MS), "fee": "1"},
    ]
    assert fees_since(fills, snapshot.legs,
                      since_ms=T0 + 5 * DAY_MS) == Decimal("1")


def test_a_maker_rebate_still_cancels_with_the_right_sign():
    """Not through abs(): forcing a sign turns a credit into a charge."""
    reader = FakeReader(positions=[
        position("A-USDT", "10", created=T0, realised="0"),
    ])
    snapshot = read_snapshot(reader, VALUES, now_ms=T0 + 30 * DAY_MS)
    fills = [{"instId": "A-USDT", "ts": str(T0), "fee": "-0.25"}]
    assert fees_since(fills, snapshot.legs, since_ms=T0) == Decimal("-0.25")


def test_fees_are_matched_to_positions_by_id_not_by_time():
    """The exact version of the same question, and the one actually used.

    Both a position and its fills carry `positionId`, so "did this fee belong
    to a position that is still open" is answered by an identity rather than by
    comparing two clocks.
    """
    reader = FakeReader(positions=[
        position("A-USDT", "10", created=T0, realised="-3", position_id="p1"),
        position("B-USDT", "-10", created=T0, realised="-2.798122",
                 position_id="p2"),
    ])
    snapshot = read_snapshot(reader, VALUES, now_ms=T0 + DAY_MS)
    fills = [
        {"instId": "A-USDT", "ts": str(T0), "fee": "3", "positionId": "p1"},
        {"instId": "B-USDT", "ts": str(T0), "fee": "2.798122",
         "positionId": "p2"},
        # Same instrument, an earlier position that has since closed. Its
        # realizedPnl went with it, so its fee must not be counted.
        {"instId": "A-USDT", "ts": str(T0), "fee": "4.050492",
         "positionId": "p0"},
    ]
    fees = fees_since(fills, snapshot.legs, since_ms=T0)
    assert fees == Decimal("5.798122")
    assert sum(leg.realized_pnl_usd for leg in snapshot.legs) + fees == Decimal("0")


def test_a_position_stamped_after_its_own_opening_fill_still_counts():
    """The bug the id match exists to avoid, at the size it actually occurred.

    On BloFin every position is stamped 20-35 ms AFTER the fill that opened it
    - BCH 27 ms, ZEC 35 ms, measured 2026-09-12. A `ts >= created_ms` bound
    therefore excluded every opening fill and reported $0.0000 of fees on a
    book that had just paid $5.80 of them. Two clocks compared at millisecond
    precision, which is the same shape as the funding settlements that print
    milliseconds late in `panel_daily.py`.
    """
    reader = FakeReader(positions=[
        position("A-USDT", "10", created=T0 + 35, realised="-1",
                 position_id="p1"),
    ])
    snapshot = read_snapshot(reader, VALUES, now_ms=T0 + DAY_MS)
    fill = [{"instId": "A-USDT", "ts": str(T0), "fee": "1",
             "positionId": "p1"}]
    assert fees_since(fill, snapshot.legs, since_ms=T0) == Decimal("1")

    # And the fallback, for a venue that reports no id, absorbs the same skew.
    no_id = FakeReader(positions=[
        position("A-USDT", "10", created=T0 + 35, realised="-1"),
    ])
    legs = read_snapshot(no_id, VALUES, now_ms=T0 + DAY_MS).legs
    assert not legs[0].position_id
    assert fees_since(fill, legs, since_ms=0) == Decimal("1")
