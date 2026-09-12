"""Telling tradeable reversion from a bid-ask bounce.

The failure this guards against is the one that motivated the module: a
variance ratio below 1 read as mean reversion when it is the spread. The
distinction is entirely in how the ratio behaves as the base interval grows,
so these pin that behaviour on series whose answer is known by construction.
"""

import numpy as np
import pytest

from analysis.reversion_scale import (
    classify,
    ou_bars,
    variance_ratio,
    variance_ratio_curve,
)


def alternating(n, level=100.0, tick=0.01):
    """Price flipping between two prices - a pure bid-ask bounce, no drift."""
    out = np.full(n, level)
    out[1::2] += tick
    return out


def random_walk(n, seed=0, step=0.001):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0, step, n)))


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


def test_a_random_walk_sits_near_one():
    ratio = variance_ratio(random_walk(200_000), base_minutes=1, window_minutes=1440)
    assert 0.8 < ratio < 1.2


def test_a_bouncing_price_looks_violently_mean_reverting():
    """The whole trap in one series: it goes nowhere over a day while flipping
    every minute, so the ratio collapses - and there is nothing to trade."""
    ratio = variance_ratio(alternating(100_000), base_minutes=1, window_minutes=1440)
    assert ratio < 0.01


def test_the_bounce_washes_out_as_the_base_lengthens():
    """Sampled every 15 minutes the flip is invisible, because the price is
    back where it started. This is what separates it from real reversion."""
    close = random_walk(200_000) + (alternating(200_000, level=0.0, tick=0.05))
    curve = variance_ratio_curve(close, bases=(1, 15))
    assert curve[1] < curve[15]
    assert curve[1] < 0.9

def test_real_reversion_is_the_same_at_every_base():
    """An OU path pulls back whether it is sampled every minute or every hour,
    so its ratio is flat - the property a bounce does not have."""
    ohlc = ou_bars(days=60, half_life_hours=24.0, seed=2)
    curve = variance_ratio_curve(ohlc.close, bases=(1, 15, 60))
    assert max(curve.values()) - min(curve.values()) < 0.15
    assert curve[60] < 0.95


def test_a_window_shorter_than_its_base_is_refused():
    with pytest.raises(ValueError):
        variance_ratio(random_walk(10_000), base_minutes=60, window_minutes=60)


def test_too_few_windows_give_nan_rather_than_a_number():
    assert np.isnan(variance_ratio(random_walk(2_000), base_minutes=1,
                                   window_minutes=1440))


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def test_reversion_only_at_the_shortest_base_is_called_an_artifact():
    assert classify({1: 0.60, 5: 0.80, 60: 1.01}) == "artifact"


def test_reversion_at_every_base_is_called_scale_invariant():
    assert classify({1: 0.79, 5: 0.80, 60: 0.79}) == "scale-invariant"


def test_a_ratio_near_one_everywhere_is_neither():
    assert classify({1: 0.98, 5: 1.00, 60: 1.03}) == "none"


def test_a_curve_with_nothing_measurable_is_neither():
    assert classify({1: float("nan"), 60: float("nan")}) == "none"


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


def test_the_generated_path_reverts_at_the_half_life_asked_for():
    """A 6h half-life must decay a deviation to about half in six hours, or the
    threshold the sweep reports is calibrated against the wrong thing."""
    ohlc = ou_bars(days=90, half_life_hours=6.0, seed=3)
    log_price = np.log(ohlc.close / 100.0)
    lag = 6 * 60
    correlation = np.corrcoef(log_price[:-lag], log_price[lag:])[0, 1]
    assert 0.35 < correlation < 0.65


def test_a_random_walk_path_does_not_revert():
    ohlc = ou_bars(days=90, half_life_hours=None, seed=3)
    log_price = np.log(ohlc.close / 100.0)
    lag = 6 * 60
    assert np.corrcoef(log_price[:-lag], log_price[lag:])[0, 1] > 0.9


def test_bars_bracket_their_own_open_and_close():
    """The high and low come from ticks inside the minute, so they must contain
    the open and close - a fade lives on exactly those extremes."""
    ohlc = ou_bars(days=2, half_life_hours=12.0, seed=4)
    assert np.all(ohlc.high >= np.maximum(ohlc.open, ohlc.close))
    assert np.all(ohlc.low <= np.minimum(ohlc.open, ohlc.close))
    assert np.all(ohlc.high >= ohlc.low)


def test_the_bars_are_consecutive_minutes():
    ohlc = ou_bars(days=1, half_life_hours=None, seed=5)
    assert len(ohlc) == 1440
    assert np.all(np.diff(ohlc.ts) == 60_000)
