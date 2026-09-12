"""The regime boundary of the touch formula, and the arithmetic under it.

The finding these guard is that the reflection principle is a moderate-distance
instrument: calibrated out to about 2 sigma and wrong by orders of magnitude at
4. The failure they exist to prevent is the opposite conclusion being reached
by a bug - a tail test that silently averages the two sides, or a "calibrated
out to k" that keeps walking past a failure.
"""

import numpy as np
import pytest

from analysis.touch_calibration import (
    calibrated_to,
    exceedance_table,
    gaussian_tail,
    standardise,
)


def test_gaussian_tail_matches_the_textbook_values():
    assert gaussian_tail(0.0) == pytest.approx(0.5)
    assert gaussian_tail(1.0) == pytest.approx(0.158655, abs=1e-6)
    assert gaussian_tail(1.96) == pytest.approx(0.025, abs=1e-4)
    assert gaussian_tail(2.0) == pytest.approx(0.022750, abs=1e-6)
    assert gaussian_tail(3.0) == pytest.approx(0.001350, abs=1e-6)


def test_gaussian_tail_keeps_its_precision_in_the_far_tail():
    """`1 - Phi(k)` loses every significant digit past about 8 sigma, which is
    exactly where this gets asked: the worst observed period was 5.1 sigma and
    the question "how overconfident is that" divides by the answer."""
    assert gaussian_tail(8.0) > 0.0
    assert gaussian_tail(8.0) == pytest.approx(6.22e-16, rel=0.01)
    # Strictly decreasing, not flushed to zero.
    assert gaussian_tail(9.0) < gaussian_tail(8.0)


def test_standardise_centres_and_scales():
    z = standardise([1.0, 2.0, 3.0])
    assert list(z) == [-1.0, 0.0, 1.0]


def test_standardise_refuses_what_it_cannot_scale():
    """A constant series or a single point has no sd, and dividing by it would
    put an inf or a nan into a pooled array where it would be counted as an
    exceedance."""
    assert len(standardise([5.0])) == 0
    assert len(standardise([2.0, 2.0, 2.0])) == 0
    assert len(standardise([])) == 0


def test_standardise_drops_non_finite_values():
    z = standardise([1.0, 2.0, np.nan, 3.0, np.inf])
    assert list(z) == [-1.0, 0.0, 1.0]


def test_the_exceedance_test_counts_only_the_loss_side():
    """A symmetric test averages an overshoot on one side against an undershoot
    on the other and reports a fit neither side has.

    Here every observation is a large GAIN, so a symmetric test would find
    eight exceedances and a one-sided loss test must find none.
    """
    z = np.array([3.0] * 8 + [0.0] * 92)
    rows = {row["k"]: row for row in exceedance_table(z)}
    assert rows[2.0]["actual"] == 0
    assert rows[3.0]["actual"] == 0


def test_the_exceedance_test_counts_losses_by_hand():
    z = np.array([-4.0, -3.5, -2.2, -1.1, 0.0, 1.0, 2.0, 3.0])
    rows = {row["k"]: row for row in exceedance_table(z)}
    assert rows[1.0]["actual"] == 4       # -4.0, -3.5, -2.2, -1.1
    assert rows[2.0]["actual"] == 3       # -4.0, -3.5, -2.2
    assert rows[3.0]["actual"] == 2       # -4.0, -3.5
    assert rows[4.0]["actual"] == 0       # strict: -4.0 is not < -4.0
    # 8 observations x P(Z < -2) = 8 x 0.02275
    assert rows[2.0]["predicted"] == pytest.approx(0.182, abs=1e-3)


def test_calibrated_to_stops_at_the_first_failure():
    """One passing cell beyond a failure is noise, not a reprieve.

    The measured single-asset table passes at 2.5 and then fails at 3.0, 3.5
    and 4.0 before "passing" at 5.0 with zero observed against 0.003 expected.
    Reporting the furthest passing distance would call that 5.0.
    """
    rows = [
        {"k": 1.0, "ratio": 0.8, "predicted": 1633.0, "actual": 1294.0},
        {"k": 2.0, "ratio": 0.9, "predicted": 234.0, "actual": 218.0},
        {"k": 2.5, "ratio": 1.7, "predicted": 63.9, "actual": 106.0},
        {"k": 3.0, "ratio": 3.2, "predicted": 13.9, "actual": 45.0},
        {"k": 5.0, "ratio": 0.0, "predicted": 0.003, "actual": 0.0},
    ]
    assert calibrated_to(rows) == 2.5


def test_calibrated_to_does_not_credit_a_cell_that_expects_under_one_event():
    """Fewer than one expected exceedance cannot confirm anything.

    At 5 sigma over 10,298 observations a Gaussian expects 0.003 events, so
    observing none is what happens whether the model is right or wrong. Reading
    that as agreement is how a tail model passes a test it never took.
    """
    rows = [
        {"k": 1.0, "ratio": 0.9, "predicted": 100.0, "actual": 90.0},
        {"k": 4.0, "ratio": 1.0, "predicted": 0.5, "actual": 0.5},
    ]
    assert calibrated_to(rows) == 1.0


def test_calibrated_to_is_zero_when_even_the_first_distance_fails():
    rows = [{"k": 1.0, "ratio": 9.0, "predicted": 100.0, "actual": 900.0}]
    assert calibrated_to(rows) == 0.0


def test_a_gaussian_sample_is_calibrated_all_the_way_out():
    """The control. If a genuinely Gaussian sample did not pass, the table
    would be measuring a bug rather than the market's tails."""
    rng = np.random.default_rng(11)
    z = standardise(rng.standard_normal(200_000))
    rows = exceedance_table(z)
    for row in rows:
        if row["predicted"] < 1.0:
            continue
        assert 0.5 <= row["ratio"] <= 2.0, row
    assert calibrated_to(rows) >= 3.5
