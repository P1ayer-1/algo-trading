"""Tests for the model trainer's evaluation machinery.

The model itself is LightGBM's problem. What is this repo's problem is
everything wrapped around it: the split that decides what "out-of-sample"
means, the decimation that decides how many observations a number really rests
on, and the bootstrap that decides whether the number differs from nothing.
Those are what get tested here, and none of them needs to train anything.
"""

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="trainer requires lightgbm")

from analysis.train_model import (  # noqa: E402
    bootstrap_top_decile,
    decimate,
    paired_top_decile_difference,
    score_report,
    three_way_split,
)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_three_way_split_leaves_a_purge_gap_at_both_seams():
    train, valid, test = three_way_split(
        1000, train_fraction=0.6, valid_fraction=0.2, purge_rows=10
    )
    assert train == slice(0, 600)
    assert valid == slice(610, 810)
    assert test == slice(820, 1000)
    # The gaps are real gaps, not off-by-one adjacency.
    assert valid.start - train.stop == 10
    assert test.start - valid.stop == 10


def test_three_way_split_blocks_never_overlap():
    train, valid, test = three_way_split(
        5000, train_fraction=0.7, valid_fraction=0.15, purge_rows=37
    )
    rows = set(range(train.start, train.stop))
    assert not rows & set(range(valid.start, valid.stop))
    assert not rows & set(range(test.start, test.stop))
    assert not set(range(valid.start, valid.stop)) & set(
        range(test.start, test.stop)
    )


def test_three_way_split_is_time_ordered():
    train, valid, test = three_way_split(
        1000, train_fraction=0.6, valid_fraction=0.2, purge_rows=5
    )
    assert train.stop <= valid.start < valid.stop <= test.start


def test_three_way_split_refuses_when_the_purge_gap_eats_the_data():
    with pytest.raises(SystemExit):
        three_way_split(100, train_fraction=0.6, valid_fraction=0.2,
                        purge_rows=50)


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------


def test_decimate_keeps_every_stride_th_row():
    assert list(decimate(10, 3)) == [0, 3, 6, 9]
    assert list(decimate(5, 1)) == [0, 1, 2, 3, 4]


def test_decimate_treats_a_zero_stride_as_one():
    assert list(decimate(4, 0)) == [0, 1, 2, 3]


def test_decimation_shrinks_a_test_set_by_the_overlap_factor():
    # 20,960 test rows at a 900s horizon sampled every 300s are really ~6,987
    # independent observations. Reporting a mean over the full set implies
    # three times the confidence the data supports.
    assert len(decimate(20_960, 3)) == 6987


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_interval_brackets_its_own_point_estimate():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000)
    returns = scores * 2.0 + rng.normal(size=4000)      # genuinely predictive
    point, low, high = bootstrap_top_decile(scores, returns, seed=0)
    assert low < point < high
    assert low > 0          # a real relationship this strong clears zero


def test_bootstrap_on_pure_noise_includes_zero():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=4000)
    returns = rng.normal(size=4000)      # no relationship at all
    point, low, high = bootstrap_top_decile(scores, returns, seed=0)
    assert low < 0 < high


def test_bootstrap_is_deterministic_for_a_given_seed():
    rng = np.random.default_rng(2)
    scores, returns = rng.normal(size=500), rng.normal(size=500)
    assert (bootstrap_top_decile(scores, returns, seed=5)
            == bootstrap_top_decile(scores, returns, seed=5))


# ---------------------------------------------------------------------------
# The paired comparison - the statistic the verdict rests on
# ---------------------------------------------------------------------------


def test_identical_scorers_differ_by_exactly_zero():
    rng = np.random.default_rng(3)
    scores, returns = rng.normal(size=2000), rng.normal(size=2000)
    point, low, high = paired_top_decile_difference(
        scores, scores, returns, seed=0
    )
    assert point == 0.0
    assert low == 0.0 and high == 0.0


def test_a_better_ranker_beats_a_worse_one_significantly():
    rng = np.random.default_rng(4)
    returns = rng.normal(size=6000)
    good = returns + rng.normal(scale=0.3, size=6000)   # nearly clairvoyant
    bad = rng.normal(size=6000)                          # no information
    point, low, high = paired_top_decile_difference(good, bad, returns, seed=0)
    assert point > 0
    assert low > 0, "a genuinely better ranker must clear zero"


def test_two_uninformative_scorers_are_not_separable():
    rng = np.random.default_rng(5)
    returns = rng.normal(size=6000)
    point, low, high = paired_top_decile_difference(
        rng.normal(size=6000), rng.normal(size=6000), returns, seed=0
    )
    assert low < 0 < high, "noise must not read as a difference"


def test_pairing_is_tighter_than_comparing_two_separate_intervals():
    # The reason the verdict uses a paired test: both models are scored on one
    # test set, so most of each interval is the same shared uncertainty. An
    # unpaired comparison keeps counting it twice and hides real differences.
    rng = np.random.default_rng(6)
    returns = rng.normal(size=6000)
    good = returns + rng.normal(scale=0.5, size=6000)
    bad = returns + rng.normal(scale=3.0, size=6000)

    paired_point, paired_low, paired_high = paired_top_decile_difference(
        good, bad, returns, seed=0
    )
    _, good_low, good_high = bootstrap_top_decile(good, returns, seed=0)
    _, bad_low, bad_high = bootstrap_top_decile(bad, returns, seed=0)

    paired_width = paired_high - paired_low
    unpaired_width = (good_high - good_low) + (bad_high - bad_low)
    assert paired_width < unpaired_width


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_score_report_carries_everything_the_verdict_reads():
    rng = np.random.default_rng(7)
    returns = rng.normal(size=3000)
    report = score_report("t", returns + rng.normal(size=3000), returns, seed=0)
    for key in ("name", "auc", "top_decile_bps", "ci_low_bps", "ci_high_bps",
                "monotonicity", "deciles"):
        assert key in report
    assert len(report["deciles"]) == 10
    assert 0.0 <= report["auc"] <= 1.0
    assert report["ci_low_bps"] <= report["top_decile_bps"] <= report["ci_high_bps"]


def test_a_predictive_score_reports_positive_monotonicity():
    rng = np.random.default_rng(8)
    returns = rng.normal(size=4000)
    report = score_report("t", returns, returns, seed=0)   # perfect ranking
    assert report["monotonicity"] > 0.9
    assert report["deciles"] == sorted(report["deciles"])
