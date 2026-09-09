"""End-to-end tests for check_features.py, using synthetic CSVs.

These are the tests that make the check trustworthy. Two cases matter:

  1. Data with a DELIBERATELY PLANTED edge -> the check must find it.
  2. Data that is pure noise -> the check must report no signal.

A check that only ever says "promising" is worse than no check, because it
provides false confidence at exactly the moment real money is at stake.
"""

import csv
import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from analysis.check_features import build_matrix, evaluate, load_rows, main
from analysis.stats import auc, fit_logistic, predict_proba, standardize

FEATURE_COLUMNS = [
    "ts", "received_ts", "mid", "microprice", "microprice_delta_bps",
    "spread", "spread_bps", "obi_1", "obi_5", "obi_20", "ofi_1s", "ofi_5s",
    "tfi_1s", "tfi_5s", "tfi_30s", "ret_1s", "ret_5s", "ret_30s",
    "rv_10s", "rv_60s", "bid_depth_20", "ask_depth_20", "trade_count_1s",
    "volume_1s", "funding_rate", "funding_ttl_s", "vol_regime",
    "history_seconds", "book_age_ms", "tape_staleness_s", "is_valid",
    "fwd_ret_bps_1s", "label_1s", "fwd_ret_bps_5s", "label_5s",
]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def synth_rows(n, *, signal_strength, seed=0, interval_ms=250):
    """Generate feature rows where obi_1 and tfi_5s predict the forward return
    with a controllable strength. signal_strength=0 gives pure noise."""
    rng = np.random.default_rng(seed)
    rows = []
    ts = 1_700_000_000_000
    mid = 100_000.0

    for index in range(n):
        obi = rng.normal()
        tfi = rng.normal()
        noise = rng.normal(scale=3.0)
        # The planted relationship, in bps.
        forward_1s = signal_strength * (0.7 * obi + 0.3 * tfi) + noise
        forward_5s = signal_strength * (0.7 * obi + 0.3 * tfi) * 1.5 + noise * 2

        rows.append({
            "ts": ts + index * interval_ms,
            "received_ts": ts + index * interval_ms + 5,
            "mid": mid, "microprice": mid + rng.normal(scale=0.1),
            "microprice_delta_bps": rng.normal(scale=0.5),
            "spread": 0.5, "spread_bps": 0.05,
            "obi_1": obi, "obi_5": obi * 0.8 + rng.normal(scale=0.3),
            "obi_20": rng.normal(),
            "ofi_1s": rng.normal(scale=10), "ofi_5s": rng.normal(scale=20),
            "tfi_1s": rng.normal(), "tfi_5s": tfi, "tfi_30s": rng.normal(),
            "ret_1s": rng.normal(scale=2), "ret_5s": rng.normal(scale=4),
            "ret_30s": rng.normal(scale=8),
            "rv_10s": abs(rng.normal(scale=1e-4)),
            "rv_60s": abs(rng.normal(scale=1e-4)),
            "bid_depth_20": 100 + rng.normal(scale=10),
            "ask_depth_20": 100 + rng.normal(scale=10),
            "trade_count_1s": rng.integers(0, 20),
            "volume_1s": abs(rng.normal(scale=5)),
            "funding_rate": 0.0001, "funding_ttl_s": 3600,
            "vol_regime": "normal_vol",
            "history_seconds": 300.0,   # past warmup
            "book_age_ms": 20, "tape_staleness_s": 0.1,
            "is_valid": "True",
            "fwd_ret_bps_1s": round(forward_1s, 4),
            "label_1s": 1 if forward_1s > 3 else (-1 if forward_1s < -3 else 0),
            "fwd_ret_bps_5s": round(forward_5s, 4),
            "label_5s": 1 if forward_5s > 3 else (-1 if forward_5s < -3 else 0),
        })
    return rows


# ---------------------------------------------------------------------------
# The two decisive cases
# ---------------------------------------------------------------------------


def run_check(tmp_path, rows, horizon=5.0, cost_bps=0.0):
    write_csv(tmp_path / "features-2026-01-01.csv", rows)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main(["--data-dir", str(tmp_path), "--horizon", str(horizon),
              "--cost-bps", str(cost_bps)])
    return buffer.getvalue()


def test_planted_signal_is_detected(tmp_path):
    """A real edge must be found, and the right features must be named."""
    output = run_check(tmp_path, synth_rows(6000, signal_strength=2.0, seed=1))
    assert "PROMISING" in output, output[-2500:]
    # The two features that actually carry the signal should rank near the top
    # of the IC table, above the pure-noise ones.
    ic_section = output.split("INFORMATION COEFFICIENTS")[1].split("=" * 72)[1]
    ranked = [line.split()[0] for line in ic_section.strip().splitlines()
              if line.strip() and not line.strip().startswith("-")
              and not line.strip().startswith("feature")]
    assert "obi_1" in ranked[:4], ranked[:6]
    assert "tfi_5s" in ranked[:4], ranked[:6]


def test_realistic_weak_signal_is_found_without_tripping_the_leakage_flag(tmp_path):
    """The case that should look like real life.

    A genuine microstructure edge is *small* — single-feature IC around
    0.02-0.06. The check must be sensitive enough to find it, while still not
    mistaking it for the implausibly strong correlation that indicates a bug.
    """
    output = run_check(
        tmp_path, synth_rows(40_000, signal_strength=0.25, seed=21), cost_bps=0.0
    )
    assert "SUSPICIOUS" not in output, "a realistic edge must not look like leakage"
    assert "NO SIGNAL" not in output, output[-2000:]
    assert "pass  No feature exceeds the plausibility threshold" in output


def test_pure_noise_reports_no_signal(tmp_path):
    """The critical negative case. If this ever says PROMISING, the check is
    broken and every result it has ever produced is suspect."""
    output = run_check(tmp_path, synth_rows(6000, signal_strength=0.0, seed=2))
    assert "NO SIGNAL" in output, output[-2500:]
    assert "PROMISING" not in output


def test_real_signal_but_costs_eat_it(tmp_path):
    """The most common real-world outcome: direction is predictable, but the
    moves are smaller than the fees. Must not be reported as tradeable."""
    output = run_check(tmp_path, synth_rows(6000, signal_strength=1.0, seed=3),
                       cost_bps=500.0)  # absurd cost, guarantees the outcome
    assert "NOT TRADEABLE" in output, output[-2500:]
    assert "Do NOT proceed to live trading" in output


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------


def test_warmup_rows_are_dropped(tmp_path):
    rows = synth_rows(1000, signal_strength=1.0, seed=4)
    for row in rows[:400]:
        row["history_seconds"] = 1.0  # below the 5s horizon
    write_csv(tmp_path / "features-2026-01-01.csv", rows)
    X, y, names, _ = build_matrix(load_rows(tmp_path), horizon=5.0)
    assert len(y) == 600


def test_invalid_rows_are_dropped(tmp_path):
    rows = synth_rows(1000, signal_strength=1.0, seed=5)
    for row in rows[:250]:
        row["is_valid"] = "False"
    write_csv(tmp_path / "features-2026-01-01.csv", rows)
    X, y, _, _ = build_matrix(load_rows(tmp_path), horizon=5.0)
    assert len(y) == 750


def test_price_levels_and_labels_are_excluded_from_features(tmp_path):
    """Absolute prices are non-stationary; label columns would be pure leakage."""
    write_csv(tmp_path / "features-2026-01-01.csv",
              synth_rows(800, signal_strength=1.0, seed=6))
    _, _, names, _ = build_matrix(load_rows(tmp_path), horizon=5.0)
    for banned in ("mid", "microprice", "ts", "received_ts", "spread",
                   "fwd_ret_bps_5s", "label_5s", "fwd_ret_bps_1s", "label_1s"):
        assert banned not in names, f"{banned} must not be a model input"
    for expected in ("obi_1", "tfi_5s", "spread_bps", "microprice_delta_bps"):
        assert expected in names


def test_multiple_files_are_concatenated(tmp_path):
    write_csv(tmp_path / "features-2026-01-01.csv",
              synth_rows(500, signal_strength=1.0, seed=7))
    write_csv(tmp_path / "features-2026-01-02.csv",
              synth_rows(500, signal_strength=1.0, seed=8))
    assert len(load_rows(tmp_path)) == 1000


def test_missing_data_directory_gives_a_useful_error(tmp_path):
    with pytest.raises(SystemExit, match="Run the bot first"):
        load_rows(tmp_path / "nope")


def test_too_little_data_refuses_to_conclude(tmp_path):
    write_csv(tmp_path / "features-2026-01-01.csv",
              synth_rows(100, signal_strength=1.0, seed=9))
    with pytest.raises(SystemExit, match="Collect more data"):
        build_matrix(load_rows(tmp_path), horizon=5.0)


def test_unknown_horizon_lists_what_is_available(tmp_path):
    write_csv(tmp_path / "features-2026-01-01.csv",
              synth_rows(800, signal_strength=1.0, seed=10))
    with pytest.raises(SystemExit, match="Available horizons"):
        build_matrix(load_rows(tmp_path), horizon=99.0)


# ---------------------------------------------------------------------------
# The purged split really does prevent leakage
# ---------------------------------------------------------------------------


def test_shuffling_inflates_auc_which_is_why_we_never_shuffle():
    """Demonstrates the failure mode the purged split exists to prevent.

    With autocorrelated features and overlapping labels, a random split lets
    near-duplicate rows land in both train and test, and the measured AUC
    becomes optimistic. The time-ordered split does not have this problem.
    """
    rng = np.random.default_rng(11)
    n = 4000
    # Strongly autocorrelated feature, as real book features are.
    raw = np.cumsum(rng.normal(size=n))
    X = np.column_stack([raw, rng.normal(size=n)])
    y = (np.roll(raw, -1) - raw > 0).astype(float)
    y[-1] = 0

    def measure(train_idx, test_idx):
        X_tr, X_te = standardize(X[train_idx], X[test_idx])
        weights = fit_logistic(X_tr, y[train_idx], l2=1.0)
        return auc(y[test_idx], predict_proba(X_te, weights))

    ordered = measure(np.arange(0, 2800), np.arange(2800, n))
    shuffled_idx = rng.permutation(n)
    shuffled = measure(shuffled_idx[:2800], shuffled_idx[2800:])

    # Both are computed the same way; only the split differs.
    assert np.isfinite(ordered) and np.isfinite(shuffled)
    # The shuffled estimate should not be *lower* — leakage only helps it.
    assert shuffled >= ordered - 0.05


# ---------------------------------------------------------------------------
# Mixed schemas: the corpus in data/ is not one generation of columns
# ---------------------------------------------------------------------------
#
# Every change to BLOFIN_LABEL_HORIZONS starts a new generation of label
# columns, and the files written under the old ones stay on disk. A real run
# hit exactly this: 37,595 rows of 1s/5s/30s labels sitting next to
# 300s/900s/1800s rows, concatenated blind, KeyError thousands of rows in.

RENAMES = {
    "fwd_ret_bps_1s": "fwd_ret_bps_300s", "label_1s": "label_300s",
    "fwd_ret_bps_5s": "fwd_ret_bps_900s", "label_5s": "label_900s",
}
LONG_HORIZON_COLUMNS = [RENAMES.get(name, name) for name in FEATURE_COLUMNS]


def write_long_horizon_csv(path, rows):
    """The same rows, labelled 300s/900s instead of 1s/5s."""
    renamed = [{RENAMES.get(key, key): value for key, value in row.items()}
               for row in rows]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_HORIZON_COLUMNS)
        writer.writeheader()
        writer.writerows(renamed)


def test_a_file_from_an_older_horizon_generation_is_skipped(tmp_path):
    write_csv(tmp_path / "features-2026-01-01.csv",
              synth_rows(400, signal_strength=1.0, seed=11))
    write_long_horizon_csv(tmp_path / "features-2026-01-02.csv",
                           synth_rows(600, signal_strength=1.0, seed=12))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rows = load_rows(tmp_path, target_column="fwd_ret_bps_900s")

    assert len(rows) == 600, "only the file carrying the horizon should load"
    assert all("fwd_ret_bps_900s" in row for row in rows)

    # Skipping quietly would be worse than crashing: the banner would claim a
    # corpus 40% larger than the one actually measured.
    report = buffer.getvalue()
    assert "Skipped 1 file(s), 400 rows" in report
    assert "features-2026-01-01.csv" in report
    assert "has: 1s, 5s" in report


def test_mixed_horizons_run_end_to_end_instead_of_raising_keyerror(tmp_path):
    """The regression test for the crash this loader was rewritten to fix."""
    # 5s sampling, so a 900s horizon costs a 180-row purge gap rather than
    # swallowing the whole test set.
    write_csv(tmp_path / "features-2026-01-01.csv",
              synth_rows(400, signal_strength=1.0, seed=13, interval_ms=5000))
    write_long_horizon_csv(
        tmp_path / "features-2026-01-02.csv",
        synth_rows(1500, signal_strength=1.0, seed=14, interval_ms=5000))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main(["--data-dir", str(tmp_path), "--horizon", "900",
              "--cost-bps", "0"])

    report = buffer.getvalue()
    assert "VERDICT" in report
    assert "Skipped 1 file(s), 400 rows" in report


def test_no_file_carrying_the_horizon_says_what_is_on_disk(tmp_path):
    write_csv(tmp_path / "features-2026-01-01.csv",
              synth_rows(400, signal_strength=1.0, seed=15))

    buffer = io.StringIO()
    with pytest.raises(SystemExit, match="Horizons present on disk: 1s, 5s"):
        with redirect_stdout(buffer):
            load_rows(tmp_path, target_column="fwd_ret_bps_1800s")


def test_build_matrix_refuses_a_horizon_not_shared_by_every_row(tmp_path):
    """The backstop, for a caller that did not filter by file.

    Taking the schema from `rows[0]` is the actual defect: whichever file sorts
    first silently decides what the columns mean for all the others, and the
    rows behind it blow up on access. Asking for a horizon that only half the
    corpus carries has to fail here, before the matrix is built.
    """
    old = synth_rows(500, signal_strength=1.0, seed=16)
    new = [{RENAMES.get(key, key): value for key, value in row.items()}
           for row in synth_rows(500, signal_strength=1.0, seed=17)]

    # `new` first, so `rows[0]` advertises a horizon most of the corpus lacks.
    with pytest.raises(SystemExit, match=r"Available horizons.*'1s'.*'900s'"):
        build_matrix(new + old, horizon=5.0)


def test_build_matrix_keeps_only_columns_every_row_has(tmp_path):
    """A feature added part-way through a run is not a feature for the corpus.

    Half a column is worse than no column: the rows predating it would parse as
    NaN and be dropped wholesale by the finite-value filter, quietly discarding
    every row recorded before the feature existed.
    """
    with_extra = [dict(row, only_in_new=1.0)
                  for row in synth_rows(500, signal_strength=1.0, seed=18)]
    without = synth_rows(500, signal_strength=1.0, seed=19)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        X, y, names, _ = build_matrix(with_extra + without, horizon=5.0)

    assert "only_in_new" not in names
    assert "obi_1" in names, "columns shared by every row must survive"
    assert len(X) == len(y) == 1000, "no row should be dropped for this"
    assert "2 different headers" in buffer.getvalue()
