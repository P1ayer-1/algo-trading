"""The scoreboard: no lookahead, a purged split, an honest control, and a
contract an outside model cannot silently mis-align with.

The failures these guard against are the ones that make a model look good:
features that can see the bar they predict, a training set that overlaps the
test window, an IC computed on the row count rather than on independent
windows, and a prediction file one row out of step.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from analysis.range_backtest import Ohlc
from analysis.range_harness import (
    FEATURES,
    Dataset,
    Persistence,
    Report,
    Ridge,
    SymbolScore,
    build_dataset,
    dump,
    evaluate,
    information_coefficient,
    print_report,
    score_predictions,
    split_timestamp,
    trailing_sum,
    verdict,
)

MINUTE_MS = 60_000
T0 = 20_000 * 86_400_000 + 59_999
HOUR_MS = 3_600_000


def bars(n, seed=0, start=T0):
    """A wiggling series with a real high/low spread, on consecutive minutes."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.concatenate(([100.0], close[:-1]))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    ts = start + MINUTE_MS * np.arange(n, dtype=np.int64)
    return Ohlc(ts=ts, open=open_, high=high, low=low, close=close)


def dataset_of(n_rows, n_symbols=2, horizon=1440, sample=60, seed=1, planted=True):
    """Synthetic feature/target sets where column 0 carries all the signal."""
    rng = np.random.default_rng(seed)
    out = []
    for index in range(n_symbols):
        X = rng.normal(size=(n_rows, len(FEATURES)))
        centre = 2 * X[:, 0] + (0.1 if planted else 50.0) * rng.normal(size=n_rows)
        ts = T0 + HOUR_MS * np.arange(n_rows, dtype=np.int64)
        out.append(Dataset(symbol=f"SYM{index}", ts=ts, X=X,
                           targets={"centre": centre, "width": X[:, 1],
                                    "contained": (X[:, 2] > 0).astype(float)},
                           price=np.full(n_rows, 100.0), horizon_minutes=horizon,
                           lookback_minutes=1440, sample_minutes=sample))
    return out


# ---------------------------------------------------------------------------
# Rolling helper
# ---------------------------------------------------------------------------


def test_trailing_sum_excludes_the_bar_it_is_computed_on():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = trailing_sum(values, 2)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == 3.0 and out[3] == 5.0 and out[4] == 7.0


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def test_targets_are_measured_over_the_bars_after_the_row():
    ohlc = bars(60)
    data = build_dataset(ohlc, horizon_minutes=2, lookback_minutes=2, sample_minutes=1)
    assert len(data) > 0
    index = {int(ts): i for i, ts in enumerate(data.ts)}
    position = 20
    row = index[int(ohlc.ts[position])]

    future_high = ohlc.high[position + 1:position + 3].max()
    future_low = ohlc.low[position + 1:position + 3].min()
    expected_centre = np.log(np.sqrt(future_high * future_low) / ohlc.close[position])
    assert data.targets["centre"][row] == pytest.approx(expected_centre)
    assert data.targets["width"][row] == pytest.approx(np.log(future_high / future_low))

    trailing_high = ohlc.high[position - 2:position].max()
    trailing_low = ohlc.low[position - 2:position].min()
    held = future_high <= trailing_high and future_low >= trailing_low
    assert data.targets["contained"][row] == float(held)


def test_features_are_computed_from_bars_before_the_row():
    ohlc = bars(60)
    data = build_dataset(ohlc, horizon_minutes=2, lookback_minutes=2, sample_minutes=1)
    index = {int(ts): i for i, ts in enumerate(data.ts)}
    position = 20
    row = index[int(ohlc.ts[position])]

    trailing_high = ohlc.high[position - 2:position].max()
    trailing_low = ohlc.low[position - 2:position].min()
    close = ohlc.close[position]
    columns = dict(zip(FEATURES, data.X[row]))
    assert columns["log_width"] == pytest.approx(np.log(trailing_high / trailing_low))
    assert columns["pos_in_range"] == pytest.approx(
        (close - trailing_low) / (trailing_high - trailing_low))
    assert columns["dist_high"] == pytest.approx(np.log(trailing_high / close))
    assert columns["dist_low"] == pytest.approx(np.log(close / trailing_low))


def test_a_later_bar_cannot_change_a_features_value():
    """The leak that looks exactly like success: a feature that can see the bar
    it is supposed to predict."""
    ohlc = bars(60)
    tampered = Ohlc(ts=ohlc.ts.copy(), open=ohlc.open.copy(), high=ohlc.high.copy(),
                    low=ohlc.low.copy(), close=ohlc.close.copy())
    tampered.high[30:] *= 1.5                      # a spike strictly after row 20

    before = build_dataset(ohlc, horizon_minutes=2, lookback_minutes=2, sample_minutes=1)
    after = build_dataset(tampered, horizon_minutes=2, lookback_minutes=2, sample_minutes=1)
    row_before = {int(ts): i for i, ts in enumerate(before.ts)}[int(ohlc.ts[20])]
    row_after = {int(ts): i for i, ts in enumerate(after.ts)}[int(ohlc.ts[20])]
    assert before.X[row_before] == pytest.approx(after.X[row_after])


def test_an_earlier_bar_cannot_change_a_target():
    ohlc = bars(60)
    tampered = Ohlc(ts=ohlc.ts.copy(), open=ohlc.open.copy(), high=ohlc.high.copy(),
                    low=ohlc.low.copy(), close=ohlc.close.copy())
    tampered.high[:10] *= 1.5                      # a spike strictly before row 20

    before = build_dataset(ohlc, horizon_minutes=2, lookback_minutes=2, sample_minutes=1)
    after = build_dataset(tampered, horizon_minutes=2, lookback_minutes=2, sample_minutes=1)
    row_before = {int(ts): i for i, ts in enumerate(before.ts)}[int(ohlc.ts[20])]
    row_after = {int(ts): i for i, ts in enumerate(after.ts)}[int(ohlc.ts[20])]
    assert (before.targets["centre"][row_before]
            == pytest.approx(after.targets["centre"][row_after]))


def test_rows_whose_windows_span_a_gap_are_dropped():
    ohlc = bars(60)
    ohlc.ts[30:] += 5 * MINUTE_MS                  # five minutes missing at bar 30
    data = build_dataset(ohlc, horizon_minutes=2, lookback_minutes=2, sample_minutes=1)
    stamps = set(int(ts) for ts in data.ts)
    assert int(ohlc.ts[29]) not in stamps          # its forward window crosses the hole
    assert int(ohlc.ts[31]) not in stamps          # its trailing window does


def test_a_series_too_short_for_one_window_yields_nothing():
    assert len(build_dataset(bars(20), horizon_minutes=10, lookback_minutes=10)) == 0


# ---------------------------------------------------------------------------
# Splitting and scoring
# ---------------------------------------------------------------------------


def test_training_stops_one_horizon_before_the_test_starts():
    """Without the purge the last training rows carry targets that reach into
    the test window, and the model is scored on labels it was shown."""
    datasets = dataset_of(200, n_symbols=1, horizon=1440)
    seen = {}

    class Recorder:
        name = "recorder"

        def fit(self, X, y):
            seen["rows"] = len(X)

        def predict(self, X):
            return X[:, 0]

    evaluate(datasets, Recorder, "centre", train_fraction=0.7, control_seeds=0)
    split = split_timestamp(datasets, 0.7)
    cutoff = split - 1440 * MINUTE_MS
    assert seen["rows"] == int(np.sum(datasets[0].ts < cutoff))
    assert seen["rows"] < int(np.sum(datasets[0].ts < split))


def test_ridge_finds_a_planted_signal_and_the_control_does_not():
    report = evaluate(dataset_of(400), Ridge, "centre", control_seeds=3)
    assert report.pooled_ic > 0.9
    assert report.positive_symbols == 2
    assert report.control_ceiling < 0.4
    # near the top of the measured table, and never above it
    assert 95.0 < report.bps <= 102.1


def test_a_dataset_of_noise_scores_like_its_control():
    report = evaluate(dataset_of(400, planted=False), Ridge, "centre", control_seeds=3)
    assert abs(report.pooled_ic) < 0.3


def test_effective_n_divides_the_rows_by_the_overlap():
    """Hourly rows with a 24h forward window are 24x overlapping: 200 rows are
    about 8 independent observations, and the t-statistic must use those."""
    report = evaluate(dataset_of(200, n_symbols=1), Ridge, "centre", control_seeds=0)
    score = report.per_symbol[0]
    assert score.effective_n == score.rows // 24
    assert abs(report.pooled_t) < abs(report.pooled_ic) * np.sqrt(score.rows)


def test_a_constant_prediction_scores_zero_rather_than_dividing_by_zero():
    assert information_coefficient(np.ones(50), np.arange(50.0)) == 0.0


def test_persistence_predicts_the_feature_it_names():
    model = Persistence(index=FEATURES.index("log_width"))
    X = np.arange(20.0).reshape(2, 10)
    model.fit(X, np.zeros(2))
    assert model.predict(X) == pytest.approx(X[:, 0])


def test_an_unknown_target_is_refused():
    with pytest.raises(SystemExit):
        evaluate(dataset_of(100), Ridge, "direction")


# ---------------------------------------------------------------------------
# The external-model contract
# ---------------------------------------------------------------------------


def test_a_dumped_dataset_scores_back_perfectly_when_predictions_are_the_truth(tmp_path):
    datasets = dataset_of(300)
    out = tmp_path / "dump"
    with redirect_stdout(io.StringIO()):
        dump(datasets, out, train_fraction=0.7, target="centre")

    predictions = tmp_path / "preds"
    predictions.mkdir()
    for data in datasets:
        payload = np.load(out / f"{data.symbol}.npz")
        test = payload["test"]
        np.savez(predictions / f"{data.symbol}.npz", ts=payload["ts"][test],
                 prediction=payload["target_centre"][test])

    report = score_predictions(datasets, predictions, "centre", train_fraction=0.7)
    assert report.pooled_ic == pytest.approx(1.0)
    assert report.model.startswith("external:")


def test_predictions_out_of_step_are_refused_not_scored(tmp_path):
    """A file one row off still scores, and scores like a signal. It must not
    be accepted quietly."""
    datasets = dataset_of(300)
    predictions = tmp_path / "preds"
    predictions.mkdir()
    split = split_timestamp(datasets, 0.7)
    for data in datasets:
        mask = data.ts >= split
        np.savez(predictions / f"{data.symbol}.npz", ts=data.ts[mask][1:],
                 prediction=data.targets["centre"][mask][1:])

    with pytest.raises(SystemExit) as excinfo:
        score_predictions(datasets, predictions, "centre", train_fraction=0.7)
    assert "out of step" in str(excinfo.value) or "rows against" in str(excinfo.value)


def test_a_missing_prediction_file_names_the_symbol(tmp_path):
    predictions = tmp_path / "preds"
    predictions.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        score_predictions(dataset_of(300), predictions, "centre", train_fraction=0.7)
    assert "SYM0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def test_the_verdict_calls_out_a_model_that_cannot_beat_its_control():
    report = Report(target="centre", model="ridge", pooled_ic=0.02,
                    control_ics=[0.05], required_ic=0.097, bps=-5.0,
                    per_symbol=[SymbolScore("BTCUSDT", 100, 4, 0.02, 0.1)])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_report(report)
        verdict(report)
    assert "SHUFFLED CONTROL" in buffer.getvalue().upper()


def test_the_verdict_quotes_the_money_threshold_when_skill_is_real_but_small():
    report = Report(target="centre", model="ridge", pooled_ic=0.03, effective_n=363,
                    control_ics=[0.01], required_ic=0.097, bps=-4.9,
                    per_symbol=[SymbolScore("BTCUSDT", 8712, 363, 0.03, 0.6)])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        verdict(report)
    text = buffer.getvalue()
    assert "BELOW BREAK-EVEN" in text
    assert "0.097" in text


def test_the_width_verdict_says_skill_there_is_not_money():
    report = Report(target="width", model="ridge", pooled_ic=0.42, control_ics=[0.02])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        verdict(report)
    assert "-15.9" in buffer.getvalue()
