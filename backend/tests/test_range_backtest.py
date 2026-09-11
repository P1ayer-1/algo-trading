"""The range-fade backtest: bars, brackets, the control, and the statistics.

Every scenario below is a handful of hand-built one-minute bars. Bars 0-3 make
a range of [99, 101] - width 2, 200 bps - so with entry 0.1, stop 0.25 and
target 0.5 the long rests at 99.2, stops at 98.5 and targets 100.0; the short
rests at 100.8, stops at 101.5, targets 100.0. Fees are 1 bp maker, 5 bps
taker, and a pessimistic fill needs 1 bp of trade-through (99.19008 / 100.81008).
"""

import io
import math
import zipfile
from decimal import Decimal

import numpy as np
import pytest

from analysis.bars_import import KLINE_COLUMNS
from analysis.importer_core import SchemaError
from analysis.range_backtest import (
    LIQUIDATED,
    OPEN,
    STOP,
    TARGET,
    TIME,
    Costs,
    Ohlc,
    RollingRange,
    Trades,
    bootstrap_means,
    contiguous,
    day_blocks,
    equity_path,
    liquidation_ratios,
    load_ohlc,
    paired_difference,
    read_klines,
    shuffle_within_days,
    sideways_now,
    simulate,
    trailing_max,
    trailing_min,
)
from trading.strategies.range_trade import RangeParams, find_range

DAY_MS = 86_400_000
T0 = 20_000 * DAY_MS + 59_999      # a bar closing one minute into a UTC day

RANGE = [
    (100.0, 101.0, 99.5, 100.0),
    (100.0, 100.5, 99.0, 100.0),
    (100.0, 100.6, 99.4, 100.0),
    (100.0, 100.4, 99.6, 100.0),
]
QUIET = [(100.0, 100.0, 100.0, 100.0)] * 3
LONG_FILL = (99.8, 99.9, 99.1, 99.3)          # rests at 99.2, low trades through


def make_ohlc(rows, start=T0):
    arr = np.array(rows, dtype=float)
    ts = start + 60_000 * np.arange(len(rows), dtype=np.int64)
    return Ohlc(ts=ts, open=arr[:, 0], high=arr[:, 1], low=arr[:, 2], close=arr[:, 3])


def run(rows, *, optimistic=False, leverage=5.0, hold=10, max_trend=None,
        slippage=0.0):
    ohlc = make_ohlc(rows)
    params = RangeParams(lookback_minutes=4, entry_frac=0.1, stop_frac=0.25,
                         target_frac=0.5, hold_minutes=hold, min_width_bps=50.0,
                         max_trend_ratio=max_trend)
    costs = Costs(maker_bps=1.0, taker_bps=5.0, slippage_bps=slippage, through_bps=1.0)
    return simulate(ohlc, RollingRange.build(ohlc, 4), params, costs,
                    optimistic=optimistic, leverage=leverage)


def only(trades):
    assert len(trades) == 1, f"expected one trade, got {len(trades)}"
    return trades.reason[0], trades.exit_index[0], trades.net_bps[0]


# ---------------------------------------------------------------------------
# The rolling range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [1, 2, 3, 5, 8, 13, 150])
def test_trailing_extremes_match_brute_force(window):
    values = np.random.default_rng(1).normal(size=200)
    highs, lows = trailing_max(values, window), trailing_min(values, window)
    assert np.all(np.isnan(highs[:window]))
    for i in range(window, 200):
        assert highs[i] == values[i - window:i].max()
        assert lows[i] == values[i - window:i].min()


def test_the_current_bar_is_never_in_its_own_range():
    """A spike at bar i must not appear until bar i+1, or the range moves out
    to meet the very price that is testing it."""
    values = np.zeros(10)
    values[6] = 5.0
    out = trailing_max(values, 3)
    assert out[6] == 0.0 and out[7] == 5.0


def test_a_window_longer_than_the_series_is_all_nan():
    assert np.all(np.isnan(trailing_max(np.arange(5.0), 5)))


def test_a_missing_minute_invalidates_the_windows_that_span_it():
    ts = T0 + 60_000 * np.array([0, 1, 2, 4, 5, 6, 7])    # minute 3 missing
    valid = contiguous(ts, 2)
    # windows ending at index 3 (minutes 1,2,4) and 4 (2,4,5) cross the hole
    assert valid.tolist() == [False, False, True, False, False, True, True]


def test_the_vectorised_range_is_find_range_at_every_bar():
    """The backtest and the planner compute the range two ways. If they ever
    disagree, the backtest is of a strategy nobody would trade."""
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 300)))
    rows = [(c, c * 1.001, c * 0.999, c) for c in close]
    ohlc = make_ohlc(rows)
    rolling = RollingRange.build(ohlc, 17)
    for i in (17, 18, 100, 299):
        window = slice(i - 17, i)
        expected = find_range(list(ohlc.high[window]), list(ohlc.low[window]),
                              list(ohlc.close[window]))
        assert rolling.high[i] == expected.high
        assert rolling.low[i] == expected.low
        assert rolling.width_bps[i] == pytest.approx(expected.width_bps)
        assert rolling.trend_ratio[i] == pytest.approx(expected.trend_ratio)


# ---------------------------------------------------------------------------
# One trade at a time, by hand
# ---------------------------------------------------------------------------


def test_a_long_that_reaches_the_middle_pays_maker_both_ways():
    rows = RANGE + [LONG_FILL, (99.3, 100.1, 99.25, 100.0)] + QUIET
    for optimistic in (False, True):
        reason, exit_index, net = only(run(rows, optimistic=optimistic))
        assert reason == TARGET and exit_index == 5
        # (100 - 99.2) / 99.2 = 80.645 bps, less 1 + 1
        assert net == pytest.approx(78.64516, abs=1e-4)


def test_a_long_that_breaks_down_is_stopped_and_pays_taker():
    rows = RANGE + [LONG_FILL, (99.0, 99.1, 98.4, 98.6)] + QUIET
    reason, exit_index, net = only(run(rows))
    assert reason == STOP and exit_index == 5
    # (98.5 - 99.2) / 99.2 = -70.565 bps, less 1 + 5
    assert net == pytest.approx(-76.56452, abs=1e-4)


def test_stop_slippage_comes_off_the_stop_price():
    rows = RANGE + [LONG_FILL, (99.0, 99.1, 98.4, 98.6)] + QUIET
    _, _, net = only(run(rows, slippage=10.0))
    # 98.5 * 0.999 = 98.4015; (98.4015 - 99.2) / 99.2 = -80.494 bps, less 6
    assert net == pytest.approx(-86.49395, abs=1e-4)


def test_a_bar_holding_both_stop_and_target_is_bracketed():
    """OHLC cannot say which printed first, so the two brackets disagree - and
    that disagreement is the uncertainty, reported rather than resolved."""
    rows = RANGE + [LONG_FILL, (99.5, 100.1, 98.4, 99.0)] + QUIET
    assert only(run(rows, optimistic=False))[0] == STOP
    assert only(run(rows, optimistic=True))[0] == TARGET


def test_a_gap_through_the_stop_fills_at_the_open():
    rows = RANGE + [LONG_FILL, (98.0, 98.1, 97.9, 98.0)] + QUIET
    reason, _, net = only(run(rows))
    assert reason == STOP
    # (98.0 - 99.2) / 99.2 = -120.968 bps, less 6 - not the -76.6 the stop price implies
    assert net == pytest.approx(-126.96774, abs=1e-4)


def test_a_stop_in_the_fill_bar_is_certain_in_both_brackets():
    """Price had to pass 99.2 on its way down to 98.4, so the fill came first."""
    rows = RANGE + [(99.8, 99.9, 98.4, 98.6)] + QUIET
    for optimistic in (False, True):
        reason, exit_index, net = only(run(rows, optimistic=optimistic))
        assert reason == STOP and exit_index == 4
        assert net == pytest.approx(-76.56452, abs=1e-4)


def test_a_target_in_the_fill_bar_counts_only_when_optimistic():
    """The 100.05 high may have printed before the dip that filled the entry."""
    rows = RANGE + [(99.8, 100.05, 99.1, 99.9), (99.9, 100.2, 99.8, 100.1)] + QUIET
    assert only(run(rows, optimistic=True))[:2] == (TARGET, 4)
    assert only(run(rows, optimistic=False))[:2] == (TARGET, 5)


def test_a_touch_fills_optimistically_but_not_pessimistically():
    rows = RANGE + [(99.8, 99.9, 99.195, 99.5), (99.5, 100.1, 99.4, 100.0)] + QUIET
    assert len(run(rows, optimistic=True)) == 1
    assert len(run(rows, optimistic=False)) == 0


def test_a_short_is_the_mirror_image():
    rows = RANGE + [(100.2, 100.9, 100.1, 100.6), (100.5, 100.6, 99.9, 100.0)] + QUIET
    trades = run(rows)
    reason, exit_index, net = only(trades)
    assert trades.side[0] == -1 and reason == TARGET and exit_index == 5
    # (100.8 - 100) / 100.8 = 79.365 bps, less 2
    assert net == pytest.approx(77.36508, abs=1e-4)


def test_the_time_stop_exits_at_the_close_as_taker():
    rows = RANGE + [LONG_FILL, (99.3, 99.6, 99.25, 99.5), (99.5, 99.7, 99.4, 99.6)] + QUIET
    reason, exit_index, net = only(run(rows, hold=2))
    assert reason == TIME and exit_index == 6
    # (99.6 - 99.2) / 99.2 = 40.323 bps, less 1 + 5
    assert net == pytest.approx(34.32258, abs=1e-4)


def test_a_trade_still_open_at_the_end_of_data_is_marked_open():
    rows = RANGE + [LONG_FILL, (99.3, 99.6, 99.25, 99.5), (99.5, 99.7, 99.4, 99.6)]
    reason, exit_index, _ = only(run(rows, hold=10))
    assert reason == OPEN and exit_index == 6


def test_one_bar_crossing_both_entries_is_skipped_and_counted():
    rows = RANGE + [(100.0, 100.9, 99.1, 100.0)] + QUIET
    trades = run(rows)
    assert len(trades) == 0 and trades.ambiguous == 1


def test_the_trend_filter_refuses_a_range_that_travelled_edge_to_edge():
    trending = [
        (99.0, 99.5, 99.0, 99.2),
        (99.2, 99.9, 99.1, 99.8),
        (99.8, 100.5, 99.7, 100.4),
        (100.4, 101.0, 100.3, 100.8),     # trend ratio (100.8 - 99.2) / 2 = 0.8
    ]
    rows = trending + [(100.7, 100.95, 100.6, 100.9)] + QUIET
    assert 4 in run(rows).entry_index.tolist()
    assert len(run(rows, max_trend=0.5)) == 0


def test_an_order_priced_through_the_market_is_not_placed():
    """If the bar opens at or below the long level, a limit there crosses the
    book and pays taker. Orders only rest."""
    rows = RANGE + [(99.2, 99.3, 99.0, 99.1)] + QUIET
    assert 4 not in run(rows, optimistic=True).entry_index.tolist()


# ---------------------------------------------------------------------------
# Leverage
# ---------------------------------------------------------------------------


def test_liquidation_ratios_come_from_the_validated_formula():
    long, short = liquidation_ratios(100, mmr=Decimal("0.005"), fee_buffer_bps=Decimal(0))
    assert long == pytest.approx(0.99 / 0.995)
    assert short == pytest.approx(1.01 / 1.005)


def test_a_stop_beyond_liquidation_is_refused_not_traded():
    """At 100x the long liquidates near 98.76 (0.99/0.995 x 1.0006 x 99.2),
    above the 98.5 stop: the stop could never fire first."""
    rows = RANGE + [LONG_FILL, (99.3, 100.1, 99.25, 100.0)] + QUIET
    refused = run(rows, leverage=100.0)
    assert len(refused) == 0 and refused.refused_liquidation == 1
    assert len(run(rows, leverage=50.0)) == 1


def test_a_stop_that_gaps_through_liquidation_loses_the_margin():
    """At 50x liquidation is ~97.76; a gap to 97.0 passes it. Isolated margin
    is 1/50 of notional: -200 bps, plus the 1 bp entry fee."""
    rows = RANGE + [LONG_FILL, (97.0, 97.1, 96.9, 97.0)] + QUIET
    reason, _, net = only(run(rows, leverage=50.0))
    assert reason == LIQUIDATED
    assert net == pytest.approx(-201.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def kline_line(open_ms, o, h, l, c):
    return f"{open_ms},{o},{h},{l},{c},1.0,{open_ms + 59_999},100.0,5,0.5,50.0,0"


def write_zip(path, lines):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(path.stem + ".csv", "\n".join(lines) + "\n")
    return path


def test_klines_parse_the_same_with_or_without_a_header(tmp_path):
    rows = [kline_line(1_780_000_000_000, 1, 2, 0.5, 1.5)]
    header = ",".join(KLINE_COLUMNS + ["ignore"])
    with_header = read_klines(write_zip(tmp_path / "a.zip", [header] + rows))
    without = read_klines(write_zip(tmp_path / "b.zip", rows))
    assert with_header.tolist() == without.tolist() == [
        [1_780_000_059_999.0, 1.0, 2.0, 0.5, 1.5]]


def test_a_header_with_high_and_low_swapped_is_refused(tmp_path):
    columns = list(KLINE_COLUMNS)
    columns[2], columns[3] = columns[3], columns[2]
    path = write_zip(tmp_path / "c.zip",
                     [",".join(columns)] + [kline_line(1_780_000_000_000, 1, 2, 0.5, 1.5)])
    with pytest.raises(SchemaError):
        read_klines(path)


def test_overlapping_archives_load_once_and_in_order(tmp_path):
    t = 1_780_000_000_000
    first = write_zip(tmp_path / "m.zip", [kline_line(t, 1, 1, 1, 1),
                                           kline_line(t + 60_000, 2, 2, 2, 2)])
    second = write_zip(tmp_path / "d.zip", [kline_line(t + 120_000, 3, 3, 3, 3),
                                            kline_line(t + 60_000, 2, 2, 2, 2)])
    ohlc = load_ohlc([second, first])
    assert (ohlc.ts - t).tolist() == [59_999, 119_999, 179_999]
    assert ohlc.close.tolist() == [1.0, 2.0, 3.0]


def test_microsecond_stamps_are_normalised(tmp_path):
    t = 1_780_000_000_000
    line = f"{t * 1000},1,1,1,1,1.0,{(t + 59_999) * 1000},100.0,5,0.5,50.0,0"
    ohlc = load_ohlc([write_zip(tmp_path / "u.zip", [line])])
    assert ohlc.ts.tolist() == [t + 59_999]


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------


def random_bars(n, seed=5, start=T0 - 100 * 60_000):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    prev = np.concatenate(([100.0], close[:-1]))
    open_ = prev * np.exp(rng.normal(0, 0.0001, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.0005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.0005, n)))
    return make_ohlc(np.column_stack([open_, high, low, close]), start=start)


def test_the_control_keeps_each_days_net_move_and_bars():
    """Bars 0-99 fall on one UTC day and 100-299 on the next."""
    real = random_bars(300)
    control = shuffle_within_days(real, np.random.default_rng(0))

    for day_end in (99, 299):
        assert control.close[day_end] == pytest.approx(real.close[day_end], rel=1e-9)

    # Both series start from the REAL first open. The control's own open[0]
    # is rebuilt from whichever bar's gap was shuffled into first place, so it
    # is not the anchor its first return was measured from.
    anchor = real.open[0]

    def day_returns(ohlc, a, b):
        prev = np.concatenate(([anchor], ohlc.close[:-1]))
        return np.sort(np.log(ohlc.close / prev)[a:b])

    assert day_returns(control, 0, 100) == pytest.approx(day_returns(real, 0, 100))
    assert day_returns(control, 100, 300) == pytest.approx(day_returns(real, 100, 300))
    assert np.all(control.high >= np.maximum(control.open, control.close) - 1e-9)
    assert np.all(control.low <= np.minimum(control.open, control.close) + 1e-9)
    assert not np.allclose(control.close, real.close)     # it did shuffle


def test_the_control_loses_its_costs_on_a_random_walk():
    """No ordering to exploit, so the pessimistic bracket must not make money.
    If it does, the simulator is manufacturing the edge."""
    rng = np.random.default_rng(11)
    n = 60_000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    prev = np.concatenate(([100.0], close[:-1]))
    high = np.maximum(prev, close) * np.exp(np.abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(prev, close) * np.exp(-np.abs(rng.normal(0, 0.0003, n)))
    ohlc = make_ohlc(np.column_stack([prev, high, low, close]))
    params = RangeParams(lookback_minutes=240, min_width_bps=50.0)
    trades = simulate(ohlc, RollingRange.build(ohlc, 240), params,
                      Costs(maker_bps=0.6, taker_bps=5.0, slippage_bps=3.0),
                      optimistic=False, leverage=5.0)
    assert len(trades) > 200
    assert trades.net_bps.mean() < 0


# ---------------------------------------------------------------------------
# Statistics and splits
# ---------------------------------------------------------------------------


def trades_of(entry_ts, net, exit_ts=None):
    exit_ts = entry_ts if exit_ts is None else exit_ts
    rows = [(0, 0, e, x, 1, 1.0, 1.0, TARGET, v, 2.0)
            for e, x, v in zip(entry_ts, exit_ts, net)]
    return Trades.from_rows(rows)


def test_a_trade_crossing_the_split_belongs_to_neither_side():
    trades = trades_of([10, 20, 30], [1.0, 2.0, 3.0], exit_ts=[15, 35, 40])
    assert trades.between(-math.inf, 25).net_bps.tolist() == [1.0]
    assert trades.between(25, math.inf).net_bps.tolist() == [3.0]


def test_day_blocks_group_trades_by_calendar_day():
    trades = trades_of([DAY_MS * 5, DAY_MS * 5 + 1, DAY_MS * 6], [10.0, 20.0, -4.0])
    days, sums, counts = day_blocks(trades)
    assert days.tolist() == [5, 6]
    assert sums.tolist() == [30.0, -4.0]
    assert counts.tolist() == [2.0, 1.0]


def test_the_bootstrap_mean_is_per_trade_not_per_day():
    """Three trades of 10 on one day and one of -2 on another average 7,
    not the 4 a mean of daily means would give."""
    means = bootstrap_means(np.array([30.0, -2.0]), np.array([3.0, 1.0]),
                            np.random.default_rng(0), 500)
    assert 7.0 in np.round(means, 9)
    assert bootstrap_means(np.array([5.0]), np.array([1.0]),
                           np.random.default_rng(0), 10) is None


def test_the_paired_difference_is_zero_against_itself():
    trades = trades_of([DAY_MS * d for d in range(10)], list(range(10)))
    point, low, high = paired_difference(trades, trades, np.random.default_rng(0), 200)
    assert point == 0.0 and low == pytest.approx(0.0) and high == pytest.approx(0.0)


def test_leverage_compounds_and_a_liquidation_ends_the_account():
    final, drawdown = equity_path(np.array([100.0, -50.0]), 2.0)
    assert final == pytest.approx(1.02 * 0.99)
    assert drawdown == pytest.approx(0.01)
    final, drawdown = equity_path(np.array([100.0, -201.0]), 50.0)
    assert final == 0.0 and drawdown == 1.0


def test_sideways_now_ranks_a_flat_stretch_against_the_trend_before_it():
    closes = [100, 101, 102, 103, 104, 104, 104, 104]
    rows = [(c, c + 0.5, c - 0.5, c) for c in closes]
    ohlc = make_ohlc(rows, start=T0)
    ohlc.ts = T0 + DAY_MS * np.arange(len(rows), dtype=np.int64)   # one bar per day
    s = sideways_now(ohlc, 3)
    assert s["move"] == pytest.approx(0.0)
    assert s["trend"] == pytest.approx(0.0)
    assert s["windows"] == 5
    # the other four windows each travelled edge to edge (ratio 1.0)
    assert s["trend_pctile"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# The entry gate
# ---------------------------------------------------------------------------


def test_a_closed_gate_removes_the_entry():
    """A model that says WHEN to fade acts by declining to open, and it must
    not reach into the levels: same range, same bars, one fewer trade."""
    rows = RANGE + [LONG_FILL, (99.3, 100.1, 99.25, 100.0)] + QUIET
    ohlc = make_ohlc(rows)
    params = RangeParams(lookback_minutes=4, entry_frac=0.1, stop_frac=0.25,
                         target_frac=0.5, hold_minutes=10, min_width_bps=50.0)
    costs = Costs(maker_bps=1.0, taker_bps=5.0, slippage_bps=0.0, through_bps=1.0)
    rolling = RollingRange.build(ohlc, 4)

    shut = np.ones(len(ohlc), dtype=bool)
    shut[4] = False           # the bar the entry would have filled on
    assert len(simulate(ohlc, rolling, params, costs, optimistic=False,
                        leverage=5.0)) == 1
    assert len(simulate(ohlc, rolling, params, costs, optimistic=False,
                        leverage=5.0, allow=shut)) == 0


def test_a_gate_that_shuts_after_entry_does_not_strand_the_position():
    """Exits are never gated. A gate that could hold a position open past its
    stop would be a far more dangerous instrument than one that declines to
    open another."""
    rows = RANGE + [LONG_FILL, (99.0, 99.1, 98.4, 98.6)] + QUIET
    ohlc = make_ohlc(rows)
    params = RangeParams(lookback_minutes=4, entry_frac=0.1, stop_frac=0.25,
                         target_frac=0.5, hold_minutes=10, min_width_bps=50.0)
    costs = Costs(maker_bps=1.0, taker_bps=5.0, slippage_bps=0.0, through_bps=1.0)
    allow = np.zeros(len(ohlc), dtype=bool)
    allow[4] = True           # open on bar 4, shut for every bar after it

    trades = simulate(ohlc, RollingRange.build(ohlc, 4), params, costs,
                      optimistic=False, leverage=5.0, allow=allow)
    reason, exit_index, net = only(trades)
    assert reason == STOP and exit_index == 5
    assert net == pytest.approx(-76.56452, abs=1e-4)
