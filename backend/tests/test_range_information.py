"""Pricing a forecast: the future-window mirrors, the blend, and the skill table.

The blend is the whole measurement - if it does not reduce exactly to the
trailing range at alpha 0 and to the true future range at alpha 1, the sweep
measures neither, and the break-even skill everything downstream quotes is a
number about nothing.
"""

import numpy as np
import pytest

from analysis.range_backtest import trailing_max
from analysis.range_information import (
    CENTRE_VALUE_BPS,
    blend,
    breakeven_centre_skill,
    future_max,
    future_min,
    future_valid,
    value_of_centre_skill,
)

MINUTE_MS = 60_000
T0 = 20_000 * 86_400_000 + 59_999


def test_the_future_window_is_the_mirror_of_the_trailing_one():
    values = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    highs = future_max(values, 2)
    lows = future_min(values, 2)
    # out[i] looks at values[i+1 : i+3]
    assert highs[0] == 5.0 and highs[1] == 9.0 and highs[2] == 9.0
    assert lows[0] == 2.0 and lows[1] == 2.0 and lows[2] == 3.0
    # The last `window` bars have no complete future window.
    assert np.isnan(highs[3]) and np.isnan(highs[4])
    # And it never includes bar i itself.
    assert highs[0] != values[0]


def test_the_future_window_agrees_with_the_trailing_one_on_the_same_bars():
    values = np.random.default_rng(0).normal(size=60)
    forward = future_max(values, 5)
    backward = trailing_max(values, 5)
    for i in range(0, 50):
        assert forward[i] == backward[i + 6] or np.isnan(forward[i])


def test_a_gap_invalidates_the_forward_window_that_spans_it():
    ts = T0 + MINUTE_MS * np.array([0, 1, 2, 4, 5, 6])   # minute 3 missing
    valid = future_valid(ts, 2)
    # from 0: bars 0,1,2 consecutive -> True. from 1: 1,2,4 -> False.
    assert valid.tolist() == [True, False, False, True, False, False]


def test_alpha_zero_is_the_trailing_range_and_alpha_one_is_the_truth():
    trailing_high = np.array([110.0])
    trailing_low = np.array([90.0])
    future_high = np.array([130.0])
    future_low = np.array([120.0])

    high, low = blend(trailing_high, trailing_low, future_high, future_low,
                      centre_alpha=0.0, width_alpha=0.0)
    assert high[0] == pytest.approx(110.0) and low[0] == pytest.approx(90.0)

    high, low = blend(trailing_high, trailing_low, future_high, future_low,
                      centre_alpha=1.0, width_alpha=1.0)
    assert high[0] == pytest.approx(130.0) and low[0] == pytest.approx(120.0)


def test_moving_the_centre_leaves_the_width_alone_and_the_reverse():
    """The two halves have to be separable or the sweep cannot say which pays."""
    trailing_high, trailing_low = np.array([110.0]), np.array([90.0])
    future_high, future_low = np.array([130.0]), np.array([120.0])

    high, low = blend(trailing_high, trailing_low, future_high, future_low,
                      centre_alpha=1.0, width_alpha=0.0)
    # width in log space is the trailing one: log(110/90)
    assert np.log(high[0] / low[0]) == pytest.approx(np.log(110.0 / 90.0))
    # centred on the future range: sqrt(130 * 120)
    assert np.sqrt(high[0] * low[0]) == pytest.approx(np.sqrt(130.0 * 120.0))

    high, low = blend(trailing_high, trailing_low, future_high, future_low,
                      centre_alpha=0.0, width_alpha=1.0)
    assert np.log(high[0] / low[0]) == pytest.approx(np.log(130.0 / 120.0))
    assert np.sqrt(high[0] * low[0]) == pytest.approx(np.sqrt(110.0 * 90.0))


def test_the_skill_table_interpolates_between_measured_points():
    assert value_of_centre_skill(0.0) == pytest.approx(-7.1)
    assert value_of_centre_skill(1.0) == pytest.approx(102.1)
    # halfway between (0, -7.1) and (0.25, +11.2)
    assert value_of_centre_skill(0.125) == pytest.approx(-7.1 + 0.5 * 18.3)
    # outside the measured range it clamps rather than extrapolating a fantasy
    assert value_of_centre_skill(2.0) == pytest.approx(102.1)


def test_break_even_is_where_the_measured_curve_crosses_zero():
    """Hand-computed: between (0, -7.1) and (0.25, +11.2), zero sits at
    0.25 * 7.1 / 18.3 = 0.097."""
    assert breakeven_centre_skill() == pytest.approx(0.25 * 7.1 / 18.3, abs=1e-6)
    assert breakeven_centre_skill(((0.0, -10.0), (1.0, 10.0))) == pytest.approx(0.5)


def test_a_curve_that_never_pays_has_no_break_even():
    """None, not a number: a strategy that loses at every skill level has no
    threshold to clear, and reporting one would invent a target."""
    assert breakeven_centre_skill(((0.0, -5.0), (0.5, -3.0), (1.0, -1.0))) is None


def test_the_shipped_table_is_monotone_in_skill():
    values = [value for _, value in CENTRE_VALUE_BPS]
    assert values == sorted(values), "more centre skill must never be worth less"
