"""Tests for the cross-sectional panel importer.

Two things carry the risk here. The ranking must be a genuine cross-section -
computed across symbols at one instant, never across time - and the label must
have the market factor actually removed rather than approximately removed.
Both are silent when wrong. Nothing here touches the network.
"""

import numpy as np
import pytest

from analysis.check_features import panel_geometry
from analysis.cross_sectional_import import (
    DEFAULT_UNIVERSE,
    META_COLUMNS,
    SymbolPanel,
    align,
    cross_sectional_rank,
    output_columns,
    per_symbol_features,
    relative_forward,
)

MINUTE = 60_000
BASE_TS = 1_700_000_000_000


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_rank_is_computed_across_symbols_not_across_time():
    # Two timestamps. Symbol order flips between them, so a rank computed down
    # the time axis instead of the symbol axis would give identical rows.
    values = np.array([[1.0, 2.0, 3.0],
                       [3.0, 2.0, 1.0]], dtype=np.float32)
    ranked = cross_sectional_rank(values)
    assert ranked[0].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert ranked[1].tolist() == pytest.approx([1.0, 0.0, -1.0])


def test_rank_spans_minus_one_to_plus_one():
    rng = np.random.default_rng(0)
    ranked = cross_sectional_rank(rng.normal(size=(50, 10)).astype(np.float32))
    assert ranked.min() == pytest.approx(-1.0)
    assert ranked.max() == pytest.approx(1.0)
    # Every row is a permutation of the same ladder.
    for row in ranked:
        assert sorted(row) == pytest.approx(
            np.linspace(-1.0, 1.0, 10).tolist(), abs=1e-6
        )


def test_rank_is_invariant_to_monotonic_rescaling():
    # The point of ranks over z-scores: one coin having an outlier minute must
    # not move everyone else's feature value.
    values = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    blown_up = np.array([[1.0, 2.0, 3.0, 10_000.0]], dtype=np.float32)
    assert cross_sectional_rank(values)[0].tolist() == pytest.approx(
        cross_sectional_rank(blown_up)[0].tolist()
    )


def test_missing_symbols_stay_missing_and_do_not_take_a_rank():
    values = np.array([[1.0, np.nan, 3.0, 2.0]], dtype=np.float32)
    ranked = cross_sectional_rank(values)
    assert np.isnan(ranked[0, 1])
    present = ranked[0, [0, 2, 3]]
    assert present.tolist() == pytest.approx([-1.0, 1.0, 0.0])


def test_a_single_present_symbol_ranks_neutral_rather_than_exploding():
    values = np.array([[np.nan, 5.0, np.nan]], dtype=np.float32)
    ranked = cross_sectional_rank(values)
    assert ranked[0, 1] == 0.0
    assert np.isnan(ranked[0, 0]) and np.isnan(ranked[0, 2])


# ---------------------------------------------------------------------------
# The label
# ---------------------------------------------------------------------------


def test_relative_forward_removes_the_market_factor_exactly():
    forward = np.array([[[10.0], [20.0], [30.0]]], dtype=np.float32)
    relative = relative_forward(forward)
    assert relative[0, :, 0].tolist() == pytest.approx([-10.0, 0.0, 10.0])
    # By construction the cross-section sums to zero.
    assert float(relative[0, :, 0].sum()) == pytest.approx(0.0, abs=1e-4)


def test_a_uniform_market_move_produces_no_label_at_all():
    # Everything up 50bps together is exactly what this label must not reward.
    forward = np.full((4, 6, 1), 50.0, dtype=np.float32)
    assert np.allclose(relative_forward(forward), 0.0)


def test_relative_forward_ignores_absent_symbols_in_the_mean():
    forward = np.array([[[10.0], [np.nan], [30.0]]], dtype=np.float32)
    relative = relative_forward(forward)
    # Mean of the two present symbols is 20, not 13.3.
    assert relative[0, 0, 0] == pytest.approx(-10.0)
    assert relative[0, 2, 0] == pytest.approx(10.0)
    assert np.isnan(relative[0, 1, 0])


def test_relative_forward_handles_each_horizon_independently():
    forward = np.array([[[10.0, 100.0], [30.0, 100.0]]], dtype=np.float32)
    relative = relative_forward(forward)
    assert relative[0, :, 0].tolist() == pytest.approx([-10.0, 10.0])
    assert relative[0, :, 1].tolist() == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def _panel(timestamps, feature_value, forward_value, tags=("900s",)):
    ts = np.asarray(timestamps, dtype=np.int64)
    return SymbolPanel(
        ts,
        np.full((len(ts), 2), feature_value, dtype=np.float32),
        {tag: np.full(len(ts), forward_value, dtype=np.float32) for tag in tags},
        np.full(len(ts), 100.0, dtype=np.float64),
    )


def test_align_puts_each_symbol_on_its_own_timestamp_row():
    panels = {
        "AAA": _panel([BASE_TS, BASE_TS + MINUTE], 1.0, 10.0),
        "BBB": _panel([BASE_TS, BASE_TS + MINUTE], 2.0, 20.0),
    }
    grid, features, forward, mid, order = align(panels, 2, ["900s"])
    assert order == ["AAA", "BBB"]
    assert grid.tolist() == [BASE_TS, BASE_TS + MINUTE]
    assert features.shape == (2, 2, 2)
    assert features[0, 0, 0] == 1.0 and features[0, 1, 0] == 2.0
    assert forward[1, 1, 0] == 20.0


def test_align_leaves_nan_where_a_symbol_has_no_observation():
    panels = {
        "AAA": _panel([BASE_TS, BASE_TS + MINUTE], 1.0, 10.0),
        "BBB": _panel([BASE_TS + MINUTE], 2.0, 20.0),
    }
    grid, features, forward, mid, order = align(panels, 2, ["900s"])
    assert len(grid) == 2
    assert np.isnan(features[0, 1, 0])       # BBB missing at the first stamp
    assert np.isnan(mid[0, 1])
    assert not np.isnan(features[1, 1, 0])


def test_align_unions_timestamps_that_do_not_overlap():
    panels = {
        "AAA": _panel([BASE_TS], 1.0, 10.0),
        "BBB": _panel([BASE_TS + MINUTE], 2.0, 20.0),
    }
    grid, _, _, mid, _ = align(panels, 2, ["900s"])
    assert grid.tolist() == [BASE_TS, BASE_TS + MINUTE]
    assert np.isnan(mid[0, 1]) and np.isnan(mid[1, 0])


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_output_columns_cover_what_the_downstream_tools_read():
    columns = output_columns(per_symbol_features(), [300.0, 900.0])
    for required in ("ts", "mid", "is_valid", "history_seconds", "symbol"):
        assert required in columns
    assert "fwd_ret_bps_300s" in columns and "label_900s" in columns
    assert any(name.startswith("xs_") for name in columns)
    assert "universe_ret_60m" in columns


def test_symbol_is_excluded_from_the_feature_matrix():
    # It is a string. Without the exclusion it parses as NaN and drops every
    # row in the file, which looks like an empty dataset rather than a bug.
    from analysis.check_features import EXCLUDED_FEATURES
    assert "symbol" in EXCLUDED_FEATURES


def test_meta_columns_are_not_ranked_as_features():
    features = per_symbol_features()
    for name in META_COLUMNS:
        assert name not in features
    assert "ret_60m" in features and "oi_chg_60m" in features


def test_default_universe_is_wide_enough_for_a_meaningful_rank():
    assert len(DEFAULT_UNIVERSE) >= 8
    assert "BTCUSDT" in DEFAULT_UNIVERSE
    assert len(set(DEFAULT_UNIVERSE)) == len(DEFAULT_UNIVERSE)


# ---------------------------------------------------------------------------
# Panel geometry - what the gate and trainer read off these files
# ---------------------------------------------------------------------------


def test_panel_geometry_sees_the_cross_section():
    # Three timestamps, four symbols each, fifteen minutes apart.
    timestamps = np.repeat(
        [BASE_TS, BASE_TS + 15 * MINUTE, BASE_TS + 30 * MINUTE], 4
    )
    interval, rows_per_ts, unique = panel_geometry(timestamps)
    assert interval == pytest.approx(900.0)
    assert rows_per_ts == pytest.approx(4.0)
    assert unique == 3


def test_panel_geometry_reduces_to_the_obvious_thing_for_one_asset():
    timestamps = np.array([BASE_TS + i * 5 * MINUTE for i in range(10)])
    interval, rows_per_ts, unique = panel_geometry(timestamps)
    assert interval == pytest.approx(300.0)
    assert rows_per_ts == pytest.approx(1.0)
    assert unique == 10


def test_naive_interval_would_understate_the_purge_gap_tenfold():
    # The bug this guards: with ten rows per timestamp, span/(rows-1) collapses
    # toward zero, and a purge gap derived from it is ten times too short in
    # time - so training rows sit inside the test period's forward window.
    timestamps = np.repeat([BASE_TS + i * 15 * MINUTE for i in range(100)], 10)
    interval, rows_per_ts, _ = panel_geometry(timestamps)
    naive = (timestamps.max() - timestamps.min()) / 1000.0 / (len(timestamps) - 1)
    assert interval == pytest.approx(900.0)
    assert naive < interval / 9
    assert interval * rows_per_ts == pytest.approx(9000.0)
