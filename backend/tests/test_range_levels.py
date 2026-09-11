"""The range and its brackets: the one definition backtest and planner share.

Hand-computed throughout. If these drift, the backtest is of a strategy that
the planner would not place.
"""

import pytest

from trading.risk import Side
from trading.strategies.range_trade import (
    Range,
    RangeParams,
    brackets,
    find_range,
    range_problems,
)


def test_the_range_is_the_extremes_of_the_bars_given():
    rng = find_range([10.0, 12.0, 11.0], [9.0, 8.0, 10.0], [9.5, 11.0, 10.5])
    assert rng.high == 12.0 and rng.low == 8.0
    assert rng.width == pytest.approx(4.0)
    assert rng.mid == pytest.approx(10.0)
    assert rng.width_bps == pytest.approx(4000.0)
    # |10.5 - 9.5| / 4
    assert rng.trend_ratio == pytest.approx(0.25)


def test_mismatched_or_empty_bars_give_no_range():
    assert find_range([], [], []) is None
    assert find_range([1.0, 2.0], [1.0], [1.0, 2.0]) is None


def test_brackets_sit_inside_the_edges_with_stops_outside():
    rng = Range(high=12.0, low=8.0, first_close=10.0, last_close=10.0)
    long, short = brackets(rng, RangeParams(entry_frac=0.1, stop_frac=0.25, target_frac=0.5))

    assert long.side is Side.LONG
    assert long.entry == pytest.approx(8.4)     # 8 + 0.1 * 4
    assert long.stop == pytest.approx(7.0)      # 8 - 0.25 * 4
    assert long.target == pytest.approx(10.0)   # 8 + 0.5 * 4

    assert short.side is Side.SHORT
    assert short.entry == pytest.approx(11.6)
    assert short.stop == pytest.approx(13.0)
    assert short.target == pytest.approx(10.0)

    # 1.4 / 8.4 and 1.6 / 8.4 of the entry
    assert long.stop_bps == pytest.approx(1666.667, abs=0.01)
    assert long.target_bps == pytest.approx(1904.762, abs=0.01)


def test_a_narrow_trending_range_reports_both_reasons():
    """Refusals are plural: the second reason still matters once the first is found."""
    rng = Range(high=100.2, low=100.0, first_close=100.0, last_close=100.2)
    reasons = range_problems(rng, RangeParams(min_width_bps=50.0, max_trend_ratio=0.5))
    assert len(reasons) == 2
    assert "too narrow" in reasons[0]
    assert "trend" in reasons[1]


def test_a_flat_range_is_refused_rather_than_divided_by():
    rng = Range(high=100.0, low=100.0, first_close=100.0, last_close=100.0)
    assert rng.trend_ratio == float("inf")
    assert range_problems(rng, RangeParams(min_width_bps=0.0)) == [
        "the range has no width - every bar printed the same price"]


def test_no_trend_filter_means_no_trend_refusal():
    rng = Range(high=101.0, low=99.0, first_close=99.0, last_close=101.0)
    assert range_problems(rng, RangeParams(min_width_bps=50.0)) == []


def test_incoherent_params_list_every_problem():
    params = RangeParams(entry_frac=0.3, target_frac=0.2, stop_frac=0.0)
    problems = params.problems()
    assert any("target_frac" in p for p in problems)
    assert any("stop_frac" in p for p in problems)
    assert RangeParams().problems() == []


def test_the_time_stop_defaults_to_one_lookback():
    assert RangeParams(lookback_minutes=240).max_hold_minutes == 240
    assert RangeParams(lookback_minutes=240, hold_minutes=30).max_hold_minutes == 30
