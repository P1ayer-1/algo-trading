"""The cross-sectional carry planner: it sizes a book and cannot send it.

The failures guarded against here are the ones that turn a dollar-neutral
strategy into a directional one without saying so - lot rounding absorbed
instead of reported, a side renormalised as a whole after a leg is dropped, a
cap that silently concentrates - plus the one property that makes this file
safe to keep in the repo at all: there is no path from a plan to an order.
"""

from decimal import Decimal

import pytest

from trading.risk import RiskLimits
from trading.strategies import Plan
from trading.strategies.carry_xs import (
    BookConfig,
    BookPlan,
    Candidate,
    LegPlan,
    plan_book,
)
from trading.strategies.carry_xs.plan import _side_weights, eligible_candidates


def candidate(name, carry, *, price="100", vol=300.0, volume=2e7,
              contract_value="1", lot="0.01", min_size="0.01",
              max_leverage="20", funding_days=200, spread="0.02"):
    bid = Decimal(price)
    return Candidate(
        inst_id=name, carry_bps_per_day=carry, bid=bid,
        ask=bid + Decimal(spread), volume_usd_24h=volume, vol_30d_bps=vol,
        contract_value=Decimal(contract_value), lot_size=Decimal(lot),
        min_size=Decimal(min_size), max_leverage=Decimal(max_leverage),
        funding_days=funding_days)


def universe(n=20, **kwargs):
    """Carry running from -6 to +6 bps/day, so the ranking is unambiguous."""
    return [candidate("C{:02d}-USDT".format(i), (i - (n - 1) / 2) * 0.6, **kwargs)
            for i in range(n)]


# ---------------------------------------------------------------------------
# The safety property
# ---------------------------------------------------------------------------


def test_the_planner_package_imports_no_broker():
    """The grep this project checks by hand, as a test.

    `plan.py` may not reach `placeOrder` by any route. Asserting it on the
    module's own import graph means a future refactor that adds a convenient
    broker import fails here rather than being noticed in review or not at all.
    """
    import trading.strategies.carry_xs.plan as module

    source = open(module.__file__, encoding="utf-8").read()
    code = "\n".join(line for line in source.splitlines()
                     if line.strip().startswith(("import ", "from ")))
    assert "broker" not in code.lower()
    assert "placeOrder" not in code


def test_a_leg_and_a_book_are_both_plans():
    """Both satisfy the `Plan` Protocol structurally, so a multi-leg strategy
    reports refusals the same way the single-instrument carry does."""
    assert isinstance(LegPlan(inst_id="SUI-USDT"), Plan)
    assert isinstance(BookPlan(), Plan)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


def test_the_book_is_long_the_highest_carry_score_and_short_the_lowest():
    """The score is NEGATED funding, so a high score is a cheap perp to hold.

    An inverted book would collect the funding spread backwards and pay it,
    which is the single most expensive sign error available here. The sign is
    fixed in `Candidate`'s field name so it cannot be decided after a result.
    """
    plan = plan_book(universe(20), BookConfig(gross_notional_usd=Decimal("20000")),
                     RiskLimits(max_notional=Decimal("100000")))
    longs = {leg.inst_id for leg in plan.long_legs if leg.ok}
    shorts = {leg.inst_id for leg in plan.short_legs if leg.ok}
    assert "C19-USDT" in longs and "C00-USDT" in shorts
    assert not (longs & shorts)
    assert min(leg.carry_bps_per_day for leg in plan.long_legs if leg.ok) > \
        max(leg.carry_bps_per_day for leg in plan.short_legs if leg.ok)


def test_legs_cross_the_spread_rather_than_pricing_at_the_mid():
    """Buy the ask, sell the bid.

    Pricing a plan at the mid understates the round trip by half a spread per
    leg. On this venue the median spread is 2.86 bps, so a mid-priced plan
    invents about 2.9 bps of edge per rebalance against a measured funding
    spread of roughly 9.
    """
    plan = plan_book(universe(20), BookConfig(gross_notional_usd=Decimal("20000")),
                     RiskLimits(max_notional=Decimal("100000")))
    for leg in plan.legs:
        if not leg.ok:
            continue
        assert leg.price == (Decimal("100.02") if leg.side == "buy"
                             else Decimal("100"))


# ---------------------------------------------------------------------------
# Neutrality
# ---------------------------------------------------------------------------


def test_lot_rounding_leaves_the_book_neutral_within_tolerance():
    plan = plan_book(universe(20), BookConfig(gross_notional_usd=Decimal("20000")),
                     RiskLimits(max_notional=Decimal("100000")))
    assert plan.ok, plan.reasons
    drift = abs(plan.net_notional_usd) / plan.gross_notional_usd
    assert drift < Decimal("0.02")


def test_coarse_lots_shrink_the_book_to_neutral_rather_than_tilting_it():
    """Coarse lots on one side cap what that side can fill; the book shrinks.

    The short side here trades in whole units of a $100 instrument, so it
    cannot reach its $2,000 target and stops at $1,500. The right answer is to
    take the long side down to meet it - a smaller neutral book - rather than
    to ship a $500 directional position or to refuse a trade that is perfectly
    possible at a smaller size. The size actually achieved is stated, because
    a book that quietly became 75% of the one requested would otherwise show up
    later as a return on the wrong denominator.
    """
    coarse = universe(10)
    for index in range(5):                 # the low-carry half, which is shorted
        coarse[index] = candidate(coarse[index].inst_id,
                                  coarse[index].carry_bps_per_day,
                                  lot="5", min_size="5")
    plan = plan_book(coarse, BookConfig(gross_notional_usd=Decimal("4000"),
                                        delta_tolerance_frac=0.005))
    long_usd = sum(leg.notional_usd for leg in plan.long_legs if leg.ok)
    short_usd = sum(leg.notional_usd for leg in plan.short_legs if leg.ok)
    assert short_usd == Decimal("1500.00")
    # Not exactly equal: the long side buys the ask and the short sells the
    # bid, so a leg sized to the same dollar target buys marginally less. The
    # residual is a rounding artifact of half a spread, not a tilt, and it is
    # what the delta tolerance exists to distinguish.
    assert abs(long_usd - short_usd) / short_usd < Decimal("0.005")
    assert plan.gross_notional_usd < Decimal("4000")
    assert plan.ok, plan.reasons
    assert any("re-sized down" in warning for warning in plan.warnings)


def test_the_delta_gate_is_still_there_as_a_backstop():
    """Sizing converges to neutral in every case constructed above, so the
    tolerance check now fires only on something the passes cannot fix. It stays
    because a silent assumption that sizing always converges is exactly the
    kind of thing that stops being true after a refactor.
    """
    plan = BookPlan()
    plan.gross_notional_usd = Decimal("1000")
    plan.net_notional_usd = Decimal("100")
    drift = abs(plan.net_notional_usd) / plan.gross_notional_usd
    assert drift > Decimal("0.02")


def test_a_side_that_cannot_be_sized_at_all_is_refused():
    """Shrinking to neutral is right; shrinking to nothing is not a book."""
    names = universe(10)
    for index in range(3):                 # the whole short side, unsizeable
        names[index] = candidate(names[index].inst_id,
                                 names[index].carry_bps_per_day,
                                 min_size="1000000")
    plan = plan_book(names, BookConfig(gross_notional_usd=Decimal("4000")))
    assert not plan.ok
    assert any("survived sizing" in reason for reason in plan.reasons)


def test_dropping_a_leg_renormalises_its_own_side_only():
    """A side whose leg fails must not be rescaled against the other side.

    Scaling the whole vector would leave the book directional by exactly the
    size of whatever failed, and what fails is not a random instrument - it is
    systematically the smallest and least liquid one.
    """
    names = universe(10)
    names[0] = candidate(names[0].inst_id, names[0].carry_bps_per_day,
                         min_size="1000000")     # cannot be sized
    plan = plan_book(names, BookConfig(gross_notional_usd=Decimal("6000"),
                                       min_names_per_side=2))
    failed = [leg for leg in plan.legs if not leg.ok]
    assert len(failed) == 1 and failed[0].inst_id == names[0].inst_id
    long_usd = sum(leg.notional_usd for leg in plan.long_legs if leg.ok)
    short_usd = sum(leg.notional_usd for leg in plan.short_legs if leg.ok)
    assert abs(long_usd - short_usd) / long_usd < Decimal("0.02")


# ---------------------------------------------------------------------------
# Refusals, plural
# ---------------------------------------------------------------------------


def test_every_failing_gate_is_reported_not_just_the_first():
    plan = plan_book(universe(20),
                     BookConfig(gross_notional_usd=Decimal("100000"),
                                leverage=Decimal("50")),
                     RiskLimits(max_notional=Decimal("5000"),
                                max_leverage=Decimal("5")))
    assert not plan.ok
    assert len(plan.reasons) >= 2
    assert any("notional limit" in r for r in plan.reasons)
    assert any("leverage" in r for r in plan.reasons)


def test_a_universe_too_narrow_to_be_a_cross_section_is_refused():
    """Three eligible names at top 30% is one position a side.

    The evidence is a cross-sectional premium measured over dozens of names. A
    book of one long and one short is two idiosyncratic bets wearing its name,
    and it is exactly what a live venue produces when the volume floor is set
    too high - which is how this gate got written.
    """
    plan = plan_book(universe(4), BookConfig(min_names_per_side=3))
    assert not plan.ok
    assert any("cross-section" in reason for reason in plan.reasons)


def test_funding_that_does_not_cover_the_round_trip_is_refused():
    """The live BloFin run refused for exactly this, and should have.

    Carry of 0.1 bps/day across the book is 0.35 bps over a week against a
    round trip of ten or more. A planner that reported this as ok would be
    contradicting the only gate the research actually established.
    """
    flat = [candidate("C{:02d}-USDT".format(i), i * 0.01) for i in range(20)]
    plan = plan_book(flat, BookConfig(gross_notional_usd=Decimal("20000")),
                     RiskLimits(max_notional=Decimal("100000")),
                     taker_fee_bps=5.0)
    assert not plan.ok
    assert any("round trip" in reason for reason in plan.reasons)


def test_liquidation_too_close_refuses_the_leg_not_the_book_silently():
    plan = plan_book(universe(20),
                     BookConfig(gross_notional_usd=Decimal("20000"),
                                leverage=Decimal("4")),
                     RiskLimits(max_notional=Decimal("100000"),
                                max_leverage=Decimal("10"),
                                min_liquidation_buffer_pct=Decimal("0.9")))
    assert all(not leg.ok for leg in plan.legs)
    assert any("liquidation" in r for leg in plan.legs for r in leg.reasons)
    assert not plan.ok


# ---------------------------------------------------------------------------
# Eligibility and sizing
# ---------------------------------------------------------------------------


def test_exclusions_are_reported_with_their_reason():
    """A silently narrowed universe looks like a book somebody chose."""
    names = universe(6)
    names[0] = candidate(names[0].inst_id, names[0].carry_bps_per_day,
                         volume=1e5, funding_days=2)
    kept, notes = eligible_candidates(names, BookConfig())
    assert len(kept) == 5
    assert len(notes) == 1
    assert "under the" in notes[0] and "funding history" in notes[0]


def test_inverse_volatility_sizing_puts_less_money_in_the_wilder_name():
    side = [candidate("QUIET-USDT", 1.0, vol=100.0),
            candidate("WILD-USDT", 1.0, vol=400.0)]
    weights, _ = _side_weights(side, BookConfig(max_weight_frac=0.0))
    assert weights["QUIET-USDT"] == pytest.approx(0.8)
    assert weights["WILD-USDT"] == pytest.approx(0.2)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_the_weight_cap_binds_and_the_remainder_is_redistributed():
    """February 2024 was three correlated meme perps on one side, and
    inverse-vol sizing made it worse because it sizes on trailing volatility.
    The cap is the control that does not depend on estimating risk correctly."""
    side = [candidate("QUIET-USDT", 1.0, vol=10.0),
            candidate("A-USDT", 1.0, vol=400.0),
            candidate("B-USDT", 1.0, vol=400.0),
            candidate("C-USDT", 1.0, vol=400.0)]
    weights, warnings = _side_weights(side, BookConfig(max_weight_frac=0.4))
    assert weights["QUIET-USDT"] <= 0.4 + 1e-9
    assert sum(weights.values()) == pytest.approx(1.0)
    assert any("capped" in warning for warning in warnings)


def test_an_equal_weight_already_over_the_cap_is_not_reported_as_capped():
    """Three names a side means 33% each and the cap cannot bind.

    Reporting all three as "capped at 33% (uncapped 33%)" is noise that trains
    the reader to skip the warnings that matter, and the live run produced
    exactly that.
    """
    side = [candidate("A-USDT", 1.0, vol=300.0),
            candidate("B-USDT", 1.0, vol=300.0),
            candidate("C-USDT", 1.0, vol=300.0)]
    weights, warnings = _side_weights(side, BookConfig(max_weight_frac=0.25))
    assert all(abs(value - 1 / 3) < 1e-9 for value in weights.values())
    assert warnings == []


def test_the_price_exposure_warning_is_always_present():
    """This book is dollar-neutral and NOT risk-neutral, unlike the spot/perp
    carry beside it. That difference has to be stated on every plan, because
    the two strategies share a name and do not share a risk."""
    plan = plan_book(universe(20), BookConfig(gross_notional_usd=Decimal("20000")),
                     RiskLimits(max_notional=Decimal("100000")))
    assert any("PRICE exposure" in warning for warning in plan.warnings)
    assert any("Leg ordering" in warning for warning in plan.warnings)
