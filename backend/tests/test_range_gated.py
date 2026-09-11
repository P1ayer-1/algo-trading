"""Gating the fade on a forecast: the mask, the threshold, and the control.

The failures these guard against are the two ways a gate flatters itself - a
permission that reaches backwards into bars the forecast could not have seen,
and a threshold chosen on the data being scored.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from analysis.range_gated import SymbolRun, allow_mask, fit_gate, report
from analysis.range_backtest import Trades
from analysis.range_harness import FEATURES, Dataset

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
T0 = 20_000 * 86_400_000 + 59_999


def datasets_of(n_rows=400, n_symbols=2, horizon=1440, sample=60, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for index in range(n_symbols):
        X = rng.normal(size=(n_rows, len(FEATURES)))
        held = (X[:, 0] > 0).astype(float)
        ts = T0 + HOUR_MS * np.arange(n_rows, dtype=np.int64)
        out.append(Dataset(symbol=f"SYM{index}", ts=ts, X=X,
                           targets={"centre": X[:, 1], "width": X[:, 2],
                                    "contained": held},
                           price=np.full(n_rows, 100.0), horizon_minutes=horizon,
                           lookback_minutes=1440, sample_minutes=sample))
    return out


# ---------------------------------------------------------------------------
# The mask
# ---------------------------------------------------------------------------


def test_a_permission_covers_the_bars_after_the_decision_only():
    """A row decided at T used bars before T, so it governs from T forward.
    Letting it reach backwards would be a gate reading the future."""
    bars = T0 + MINUTE_MS * np.arange(10, dtype=np.int64)
    rows = bars[[0, 5]]
    allow = allow_mask(bars, rows, np.array([True, False]), 3)
    assert allow.tolist() == [True, True, True, False, False,
                              False, False, False, False, False]


def test_each_decision_governs_until_the_next_one():
    bars = T0 + MINUTE_MS * np.arange(10, dtype=np.int64)
    rows = bars[[2, 5]]
    allow = allow_mask(bars, rows, np.array([True, True]), 3)
    assert allow[2:5].all() and allow[5:8].all()
    assert not allow[0:2].any() and not allow[8:].any()


def test_a_gate_that_permits_nothing_masks_every_bar():
    bars = T0 + MINUTE_MS * np.arange(6, dtype=np.int64)
    allow = allow_mask(bars, bars[[0, 3]], np.array([False, False]), 3)
    assert not allow.any()


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------


def test_the_threshold_is_cut_on_training_predictions_only():
    """Choosing the cut on the rows being scored is how a gate is made to look
    decisive after the fact."""
    datasets = datasets_of()
    model, mean, deviation, threshold, split = fit_gate(
        datasets, train_fraction=0.7, keep=0.5)

    horizon_ms = datasets[0].horizon_minutes * MINUTE_MS
    train = datasets[0].ts < split - horizon_ms
    train_predictions = np.concatenate([
        model.predict((dataset.X[dataset.ts < split - horizon_ms] - mean) / deviation)
        for dataset in datasets])
    # keep=0.5 means the cut is the median of what the model said in training
    assert threshold == pytest.approx(np.quantile(train_predictions, 0.5))
    assert train.sum() < len(datasets[0])


def test_keeping_less_raises_the_threshold():
    datasets = datasets_of()
    _, _, _, half, _ = fit_gate(datasets, train_fraction=0.7, keep=0.5)
    _, _, _, tenth, _ = fit_gate(datasets, train_fraction=0.7, keep=0.1)
    assert tenth > half


def test_the_gate_learns_a_planted_containment_signal():
    datasets = datasets_of()
    model, mean, deviation, threshold, split = fit_gate(
        datasets, train_fraction=0.7, keep=0.5)
    data = datasets[0]
    test = data.ts >= split
    predictions = model.predict((data.X[test] - mean) / deviation)
    kept = predictions >= threshold
    # contained was planted as X[:, 0] > 0, so the permitted rows should carry
    # a far higher containment rate than the refused ones
    held = data.targets["contained"][test]
    assert held[kept].mean() > held[~kept].mean() + 0.5


def test_too_little_training_data_is_refused():
    with pytest.raises(SystemExit):
        fit_gate(datasets_of(n_rows=30, n_symbols=1), train_fraction=0.7, keep=0.5)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def trades_of(nets, day_step=1):
    rows = [(0, 0, T0 + i * day_step * 86_400_000, T0 + i * day_step * 86_400_000,
             1, 1.0, 1.0, 0, net, 2.0) for i, net in enumerate(nets)]
    return Trades.from_rows(rows)


def test_the_verdict_refuses_to_credit_a_gate_that_only_trades_less():
    """Same edge per trade, fewer trades: a smaller position in the same losing
    strategy, which must not read as an improvement."""
    runs = [SymbolRun(symbol="BTCUSDT",
                      ungated=trades_of([-10.0] * 40),
                      gated=trades_of([-10.0] * 20),
                      control=trades_of([-10.0] * 20),
                      allowed_share=0.5)]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(runs, keep=0.5, optimistic=False, draws=200, seed=1, split_ts=T0)
    text = buffer.getvalue()
    assert "THE GATE DOES NOT PAY" in text
    assert "removed 50% of the trades" in text


def test_a_gate_that_turns_the_fade_positive_is_reported_as_such():
    runs = [SymbolRun(symbol="BTCUSDT",
                      ungated=trades_of([-10.0] * 40),
                      gated=trades_of([12.0] * 20),
                      control=trades_of([-9.0] * 20),
                      allowed_share=0.5)]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(runs, keep=0.5, optimistic=False, draws=200, seed=1, split_ts=T0)
    assert "MAKES MONEY OUT OF SAMPLE" in buffer.getvalue()
