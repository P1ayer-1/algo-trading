"""Tests for the bar-level Binance importer.

Weighted heavily toward lookahead, because that is the failure this importer
is most likely to have and the one that hurts most: a leak does not raise, it
produces a beautiful backtest and a losing account. Nothing here touches the
network.
"""

import math
import zipfile

import pytest

from analysis.bars_import import (
    METRICS_WINDOW_MS,
    Bars,
    as_of,
    bar_features,
    build_rows,
    daterange,
    depth_features,
    feature_columns,
    forward_index,
    horizon_tag,
    kline_urls,
    load_bars,
    metrics_features,
    window_start,
)
from datetime import date
from trading.recorder import FeatureRecorder

MINUTE = 60_000
BASE_TS = 1_700_000_000_000


def _counters():
    return {"warmup": 0, "bar_gap": 0, "metrics": 0, "premium": 0,
            "depth": 0, "no_future": 0}


def make_metrics(count, *, start=BASE_TS, oi=lambda i: 1000.0 + i):
    """One metrics snapshot every window, back far enough for a 240m lookback."""
    timestamps = [start + i * METRICS_WINDOW_MS for i in range(count)]
    rows = [{"oi": oi(i), "toptrader_pos_ls": 1.0, "toptrader_acct_ls": 1.0,
             "global_acct_ls": 1.0, "taker_ls_ratio": 0.5} for i in range(count)]
    return timestamps, rows


def make_bars(count, *, start=BASE_TS, step=MINUTE, price=lambda i: 100.0):
    bars = Bars()
    for i in range(count):
        value = price(i)
        bars.ts.append(start + i * step)
        bars.close.append(value)
        bars.high.append(value * 1.001)
        bars.low.append(value * 0.999)
        bars.volume.append(10.0)
        bars.count.append(100.0)
        bars.taker_buy.append(5.0)
    return bars


# ---------------------------------------------------------------------------
# As-of joins
# ---------------------------------------------------------------------------


def test_as_of_never_returns_a_future_observation():
    timestamps = [100, 200, 300]
    assert as_of(timestamps, 250) == 1        # the 200 one, not the 300 one
    assert as_of(timestamps, 200) == 1        # exactly on a sample: usable
    assert as_of(timestamps, 199) == 0
    assert as_of(timestamps, 99) is None      # nothing observable yet


def test_as_of_on_empty_history():
    assert as_of([], 1000) is None


# ---------------------------------------------------------------------------
# Window resolution
# ---------------------------------------------------------------------------


def test_window_start_finds_the_bar_a_lookback_ago():
    bars = make_bars(100)
    assert window_start(bars, 60, 60) == 0
    assert window_start(bars, 99, 15) == 84


def test_window_start_refuses_a_window_with_a_gap_inside_it():
    bars = make_bars(100)
    # Delete five bars from the middle of what would be a 60-minute window.
    for field in ("ts", "close", "high", "low", "volume", "count", "taker_buy"):
        values = getattr(bars, field)
        del values[50:55]
    # Index 90 is now 65 minutes of wall-clock behind index 30, so a lookback
    # resolved by index would silently measure the wrong horizon.
    assert window_start(bars, 90, 60) is None


def test_window_start_refuses_when_history_is_too_short():
    bars = make_bars(30)
    assert window_start(bars, 10, 60) is None


# ---------------------------------------------------------------------------
# Forward labels
# ---------------------------------------------------------------------------


def test_forward_index_is_strictly_forward():
    bars = make_bars(100)
    assert forward_index(bars, 0, 900) == 15
    assert forward_index(bars, 10, 1800) == 40


def test_forward_index_returns_none_past_the_end_of_the_data():
    bars = make_bars(20)
    assert forward_index(bars, 10, 1800) is None


def test_horizon_tag_matches_the_recorder_exactly():
    # The column names have to line up with live-recorded files, or a model
    # trained on imported data cannot be scored on recorded data.
    for horizon in (1.0, 5.0, 30.0, 300.0, 900.0, 1800.0, 0.5):
        assert horizon_tag(horizon) == FeatureRecorder._tag(horizon)


def test_forward_return_uses_the_recorders_simple_return_formula():
    # Simple return, not log: FeatureRecorder._label uses (future/now - 1), and
    # a row from here has to mean the same thing as a row from the live feed.
    minutes = 3000
    bars = make_bars(minutes, price=lambda i: 100.0 + (i % 89) * 0.02)
    metric_ts, metric_rows = make_metrics(minutes * MINUTE // METRICS_WINDOW_MS)
    premium_ts = [BASE_TS + i * 3_600_000 for i in range(minutes // 60 + 1)]

    rows = list(build_rows(
        bars, metric_ts, metric_rows, premium_ts, [0.0001] * len(premium_ts),
        [], [], horizons=(900.0,), threshold_bps=10.0, sample_minutes=5,
        with_depth=False, dropped=_counters(),
    ))
    assert rows

    by_ts = dict(zip(bars.ts, bars.close))
    checked = 0
    for row in rows:
        future = by_ts.get(row["ts"] + 900_000)
        if future is None:
            continue
        expected = (future / row["mid"] - 1.0) * 10_000.0
        assert row["fwd_ret_bps_900s"] == pytest.approx(expected, abs=1e-4)
        checked += 1
    assert checked > 100


# ---------------------------------------------------------------------------
# The metrics lag - the leak this importer actually had
# ---------------------------------------------------------------------------


def test_metrics_row_stamped_at_the_anchor_is_not_used():
    # A metrics row timestamped T describes [T, T+5min), so at anchor T it has
    # not finished happening. Using it leaks the next five minutes, which is
    # exactly the bug this importer shipped with and check_features caught.
    count = 100
    metric_ts, rows = make_metrics(count)
    anchor = metric_ts[-1]                      # anchor exactly on a snapshot
    rows[-1]["taker_ls_ratio"] = 999.0          # the not-yet-happened window
    rows[-2]["taker_ls_ratio"] = 0.2            # the completed one

    features = metrics_features(metric_ts, rows, anchor)
    assert features is not None
    assert features["taker_ls_ratio"] == 0.2
    assert features["metrics_age_s"] == METRICS_WINDOW_MS / 1000.0


def test_metrics_lag_applies_to_the_open_interest_lookbacks_too():
    count = 100
    metric_ts, rows = make_metrics(count)
    anchor = metric_ts[-1]

    features = metrics_features(metric_ts, rows, anchor)
    assert features is not None
    # Newest usable row is one window back; the 15-minute lookback is three
    # windows before that. Neither may be the row stamped at the anchor.
    current = count - 2
    past = current - 15 * MINUTE // METRICS_WINDOW_MS
    assert features["oi_chg_15m"] == pytest.approx(
        math.log(rows[current]["oi"] / rows[past]["oi"]) * 10_000.0
    )


def test_metrics_features_none_when_nothing_is_observable_yet():
    anchor = BASE_TS
    assert metrics_features([anchor], [{"oi": 1.0}], anchor) is None
    # Present but too short for the 240-minute lookback: also unusable.
    short_ts, short_rows = make_metrics(10)
    assert metrics_features(short_ts, short_rows, short_ts[-1]) is None


# ---------------------------------------------------------------------------
# Bar features
# ---------------------------------------------------------------------------


def test_flat_prices_produce_zero_returns_and_zero_volatility():
    bars = make_bars(2000)
    features = bar_features(bars, 1500)
    assert features is not None
    for name in ("ret_5m", "ret_60m", "ret_1440m", "ma_dist_60m", "rv_60m"):
        assert features[name] == pytest.approx(0.0, abs=1e-9)


def test_taker_flow_imbalance_matches_its_definition():
    bars = make_bars(2000)
    # 5 of every 10 units bought -> perfectly balanced.
    assert bar_features(bars, 1500)["tfi_15m"] == pytest.approx(0.0)
    for i in range(len(bars.ts)):
        bars.taker_buy[i] = 10.0          # every trade a buy
    assert bar_features(bars, 1500)["tfi_15m"] == pytest.approx(1.0)


def test_price_above_its_moving_average_gives_positive_ma_distance():
    bars = make_bars(2000, price=lambda i: 100.0 + i * 0.01)   # steady uptrend
    features = bar_features(bars, 1900)
    assert features["ma_dist_60m"] > 0
    assert features["ret_60m"] > 0
    assert 0.0 <= features["range_pos_240m"] <= 1.0


def test_bar_features_none_before_the_longest_lookback_is_available():
    bars = make_bars(200)
    assert bar_features(bars, 100) is None


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------


def test_depth_imbalance_sign_follows_the_negative_percentage_side():
    snapshot = {}
    for band in (0.2, 1.0, 2.0, 5.0):
        snapshot[f"neg_{band:g}"] = 300.0
        snapshot[f"pos_{band:g}"] = 100.0
    features = depth_features([BASE_TS], [snapshot], BASE_TS, None)
    assert features is not None
    assert features["depth_imb_1"] == pytest.approx(0.5)
    assert features["depth_age_s"] == 0.0


def test_depth_features_none_when_a_band_is_missing():
    snapshot = {"neg_1": 1.0, "pos_1": 1.0}
    assert depth_features([BASE_TS], [snapshot], BASE_TS, None) is None


# ---------------------------------------------------------------------------
# URLs and loading
# ---------------------------------------------------------------------------


def test_complete_months_use_monthly_archives_and_edges_use_daily():
    days = daterange(date(2025, 1, 30), date(2025, 3, 2))
    urls = kline_urls("BTCUSDT", days, "klines", "1m")
    monthly = [url for url in urls if "/monthly/" in url]
    daily = [url for url in urls if "/daily/" in url]
    # February is fully covered; January and March are not.
    assert len(monthly) == 1
    assert "2025-02" in monthly[0]
    assert len(daily) == 4          # Jan 30, Jan 31, Mar 1, Mar 2
    assert all("2025-02-" not in url for url in daily)


def _write_kline_zip(path, rows):
    with zipfile.ZipFile(path, "w") as archive:
        lines = ["open_time,open,high,low,close,volume,close_time,quote_volume,"
                 "count,taker_buy_volume,taker_buy_quote_volume,ignore"]
        for open_time, close in rows:
            lines.append(
                f"{open_time},{close},{close},{close},{close},1,"
                f"{open_time + MINUTE - 1},1,1,0.5,1,0"
            )
        archive.writestr(path.stem + ".csv", "\n".join(lines))


def test_overlapping_monthly_and_daily_files_are_deduplicated(tmp_path):
    first = tmp_path / "a.zip"
    second = tmp_path / "b.zip"
    _write_kline_zip(first, [(BASE_TS, 100.0), (BASE_TS + MINUTE, 101.0)])
    _write_kline_zip(second, [(BASE_TS + MINUTE, 101.0), (BASE_TS + 2 * MINUTE, 102.0)])

    bars = load_bars([first, second])
    assert len(bars) == 3
    assert bars.ts == sorted(bars.ts)
    assert len(set(bars.ts)) == 3


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_output_carries_every_column_check_features_requires():
    columns = feature_columns(with_depth=False)
    # check_features filters on these three and splits on ts.
    for required in ("ts", "mid", "is_valid", "history_seconds"):
        assert required in columns
    # ...and excludes exactly these from the feature matrix.
    assert "vol_regime" in columns
    assert "received_ts" in columns


def test_end_to_end_rows_are_time_ordered_and_labelled(tmp_path):
    minutes = 3000
    bars = make_bars(minutes, price=lambda i: 100.0 + (i % 97) * 0.01)
    metric_ts, metric_rows = make_metrics(minutes * MINUTE // METRICS_WINDOW_MS)
    premium_ts = [BASE_TS + i * 3_600_000 for i in range(minutes // 60 + 1)]
    premium_values = [0.0001] * len(premium_ts)

    dropped = _counters()
    rows = list(build_rows(
        bars, metric_ts, metric_rows, premium_ts, premium_values, [], [],
        horizons=(300.0, 900.0), threshold_bps=10.0, sample_minutes=5,
        with_depth=False, dropped=dropped,
    ))

    assert rows, dropped
    assert [row["ts"] for row in rows] == sorted(row["ts"] for row in rows)
    for row in rows:
        assert row["is_valid"] is True
        assert row["history_seconds"] >= 86_400      # clears the warmup filter
        assert "fwd_ret_bps_300s" in row and "label_900s" in row
        assert row["label_300s"] in (-1, 0, 1)
