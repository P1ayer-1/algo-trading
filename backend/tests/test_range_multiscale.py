"""Ranges at several scales at once: alignment, per-side breaks, like-for-like.

The break targets are DERIVED from the centre and width rather than recomputed,
so the test that matters is that the derivation equals what the bars actually
did. If it does not, the study scores a target nobody defined.
"""

import numpy as np
import pytest

from analysis.range_backtest import Ohlc, trailing_max, trailing_min
from analysis.range_harness import FEATURES, build_dataset
from analysis.range_information import future_max, future_min
from analysis.range_multiscale import (
    asymmetry,
    break_targets,
    build_multiscale,
    fit_side_models,
    main,
    scale_label,
    single_scale_view,
)
from analysis.range_harness import split_timestamp

MINUTE_MS = 60_000
T0 = 20_000 * 86_400_000 + 59_999
SCALES = (120, 240)          # 2h and 4h, so a short fixture still has rows


def bars(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.concatenate(([100.0], close[:-1]))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return Ohlc(ts=T0 + MINUTE_MS * np.arange(n, dtype=np.int64),
                open=open_, high=high, low=low, close=close)


def test_scale_labels_read_as_hours_then_days():
    assert scale_label(240) == "4h"
    assert scale_label(1440) == "1d"
    assert scale_label(10080) == "7d"


def test_a_break_flag_matches_what_the_bars_did():
    """Derived from centre and width; checked against the extremes directly."""
    ohlc = bars(4000)
    lookback, horizon = 120, 60
    data = build_dataset(ohlc, horizon_minutes=horizon, lookback_minutes=lookback,
                         sample_minutes=30, symbol="X")
    flags = break_targets(data, "2h")

    high = trailing_max(ohlc.high, lookback)
    low = trailing_min(ohlc.low, lookback)
    future_high = future_max(ohlc.high, horizon)
    future_low = future_min(ohlc.low, horizon)
    rows = np.searchsorted(ohlc.ts, data.ts)

    expected_up = (future_high[rows] > high[rows]).astype(float)
    expected_down = (future_low[rows] < low[rows]).astype(float)
    assert flags["break_up@2h"] == pytest.approx(expected_up)
    assert flags["break_down@2h"] == pytest.approx(expected_down)


def test_the_two_sides_are_separate_flags():
    """The symmetric `contained` target collapsed these into one bit and lost
    the direction, which is the half that pays."""
    ohlc = bars(4000)
    data = build_dataset(ohlc, horizon_minutes=60, lookback_minutes=120,
                         sample_minutes=30, symbol="X")
    flags = break_targets(data, "2h")
    up, down = flags["break_up@2h"], flags["break_down@2h"]
    held = data.targets["contained"]
    # contained means neither side broke
    assert np.all((held == 1.0) == ((up == 0.0) & (down == 0.0)))
    assert up.sum() > 0 and down.sum() > 0          # both actually occur


def test_rows_are_the_intersection_across_scales():
    """A long lookback needs more history before its first valid row; a row
    missing one scale is a row where that scale would be guessed."""
    ohlc = bars(6000)
    data = build_multiscale(ohlc, scales_minutes=SCALES, horizon_minutes=60,
                            sample_minutes=30, symbol="X")
    per_scale = [build_dataset(ohlc, horizon_minutes=60, lookback_minutes=scale,
                               sample_minutes=30, symbol="X") for scale in SCALES]
    expected = set(per_scale[0].ts) & set(per_scale[1].ts)
    assert set(data.ts) == expected
    assert len(data) < max(len(one) for one in per_scale)


def test_every_scale_contributes_a_feature_block():
    ohlc = bars(6000)
    data = build_multiscale(ohlc, scales_minutes=SCALES, horizon_minutes=60,
                            sample_minutes=30, symbol="X")
    assert data.X.shape[1] == len(FEATURES) * len(SCALES)
    assert data.feature_names[0].endswith("@2h")
    assert data.feature_names[-1].endswith("@4h")
    for scale in SCALES:
        label = scale_label(scale)
        assert f"break_up@{label}" in data.targets
        assert f"break_down@{label}" in data.targets


def test_centre_and_width_are_carried_once_and_agree_across_scales():
    """They describe the forward window and the close, so they cannot depend on
    the lookback - taking them from the first scale has to be safe."""
    ohlc = bars(6000)
    data = build_multiscale(ohlc, scales_minutes=SCALES, horizon_minutes=60,
                            sample_minutes=30, symbol="X")
    for scale in SCALES:
        one = build_dataset(ohlc, horizon_minutes=60, lookback_minutes=scale,
                            sample_minutes=30, symbol="X")
        rows = np.searchsorted(one.ts, data.ts)
        assert one.targets["centre"][rows] == pytest.approx(data.targets["centre"])
        assert one.targets["width"][rows] == pytest.approx(data.targets["width"])


def test_the_single_scale_view_is_that_scales_block_and_nothing_else():
    """The comparison has to be like-for-like: same rows, same split, same
    control, only the feature set narrows."""
    ohlc = bars(6000)
    data = build_multiscale(ohlc, scales_minutes=SCALES, horizon_minutes=60,
                            sample_minutes=30, symbol="X")
    view = single_scale_view(data, SCALES, 240)
    assert view.X.shape == (len(data), len(FEATURES))
    assert view.X == pytest.approx(data.X[:, len(FEATURES):])
    assert view.ts is data.ts
    assert view.targets is data.targets


def test_a_series_too_short_for_the_longest_scale_yields_nothing():
    assert build_multiscale(bars(300), scales_minutes=(120, 1440),
                            horizon_minutes=60, sample_minutes=30,
                            symbol="X") is None


def test_at_least_two_scales_are_required():
    with pytest.raises(SystemExit):
        main(["--scale-hours", "24", "--symbols", "BTCUSDT"])


def test_the_baseline_must_be_one_of_the_scales():
    with pytest.raises(SystemExit) as excinfo:
        main(["--scale-hours", "4,24", "--baseline-hours", "72",
              "--symbols", "BTCUSDT"])
    assert "baseline-hours" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Which way, rather than how far
# ---------------------------------------------------------------------------


def multiscale_sets(n=3000, symbols=2):
    out = []
    for index in range(symbols):
        data = build_multiscale(bars(n, seed=index), scales_minutes=SCALES,
                                horizon_minutes=60, sample_minutes=30,
                                symbol=f"SYM{index}")
        assert data is not None
        out.append(data)
    return out


def test_the_side_models_are_fitted_on_training_rows_only():
    """The asymmetry is read straight off these predictions, so if they were
    fitted through the split there would be nothing left to catch it."""
    datasets = multiscale_sets()
    labels = [scale_label(scale) for scale in SCALES]
    models, mean, deviation, split = fit_side_models(
        datasets, labels, train_fraction=0.7, l2=1.0)

    assert split == split_timestamp(datasets, 0.7)
    assert set(models) == {f"break_{side}@{label}"
                           for label in labels for side in ("up", "down")}
    cutoff = split - datasets[0].horizon_minutes * 60_000
    trained_on = sum(int(np.sum(data.ts < cutoff)) for data in datasets)
    assert trained_on < sum(len(data) for data in datasets)


def test_identical_sides_leave_no_asymmetry():
    """Two models that say the same thing have a gap of exactly zero, so the
    read-out cannot manufacture a direction out of agreement."""
    datasets = multiscale_sets()
    labels = [scale_label(scale) for scale in SCALES]
    models, mean, deviation, split = fit_side_models(
        datasets, labels, train_fraction=0.7, l2=1.0)
    for label in labels:
        models[f"break_down@{label}"] = models[f"break_up@{label}"]

    sides = asymmetry(datasets, labels, models, mean, deviation, split)
    for label in labels:
        together, gap_ic = sides[label]
        assert together == pytest.approx(1.0)
        assert gap_ic == 0.0


def test_the_asymmetry_reports_a_correlation_and_an_ic_per_scale():
    datasets = multiscale_sets()
    labels = [scale_label(scale) for scale in SCALES]
    models, mean, deviation, split = fit_side_models(
        datasets, labels, train_fraction=0.7, l2=1.0)
    sides = asymmetry(datasets, labels, models, mean, deviation, split)

    assert set(sides) == set(labels) | {"all"}
    for label in labels:
        together, gap_ic = sides[label]
        assert -1.0 <= together <= 1.0
        assert -1.0 <= gap_ic <= 1.0
