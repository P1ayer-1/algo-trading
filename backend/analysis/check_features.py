"""Do these features predict anything? Run this before building any model.

    python backend/analysis/check_features.py
    python backend/analysis/check_features.py --horizon 900 --data-dir data

This is step 5 of the roadmap, and it is a gate, not a formality. If the
features carry no out-of-sample information, no amount of LightGBM, feature
engineering, or hyperparameter tuning will manufacture an edge — and finding
that out here costs an afternoon instead of a month.

What it does, in order:

  1. Loads the recorded CSVs and drops warmup rows (see `history_seconds`).
  2. Reports the data you actually have: span, sampling rate, label balance,
     and the EFFECTIVE sample size after accounting for label overlap.
  3. Information coefficients per feature (Pearson + Spearman), with
     t-statistics computed on the effective N, not the row count.
  4. Leakage checks. Loud, because a leak looks exactly like success.
  5. A time-ordered, purged train/test split — never shuffled.
  6. A logistic-regression baseline: AUC and accuracy vs the majority class.
  7. The economic test: conditional on the model firing, is the mean forward
     move bigger than the round-trip cost? This is the only test that decides
     whether anything here is tradeable.

A model can pass (6) and fail (7). Accuracy on its own tells you nothing
about profitability — a 55%-accurate model that predicts 1bps moves loses
money against 6bps of costs, every time.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.stats import (  # noqa: E402
    auc,
    correlation_tstat,
    decile_returns,
    effective_sample_size,
    fit_logistic,
    pearson,
    predict_proba,
    purged_split,
    spearman,
    standardize,
)

# Columns that must never be used as model inputs.
#   ts / received_ts  - timestamps; a model keying on them is memorising, and
#                       the relationship cannot persist out of sample.
#   mid / microprice  - absolute price LEVELS. Non-stationary: BTC at 100k vs
#                       60k makes any learned threshold meaningless. Their
#                       relative forms (microprice_delta_bps, ret_*) are fine.
#   spread            - absolute; spread_bps is the stationary version.
#   vol_regime        - categorical string, and derived from rv_60s anyway.
#   is_valid          - constant after filtering.
EXCLUDED_FEATURES = {
    "ts",
    "received_ts",
    # Panel identifier, not a feature. It is also a string, and without this it
    # would parse as NaN and silently drop every row in a cross-sectional file.
    "symbol",
    "mid",
    "microprice",
    "spread",
    "vol_regime",
    "is_valid",
    "history_seconds",
}

# An out-of-sample IC above this is almost never real at HFT horizons.
SUSPICIOUS_IC = 0.20


def panel_geometry(timestamps: np.ndarray) -> Tuple[float, float, int]:
    """(sample interval in seconds, rows per timestamp, distinct timestamps).

    A single-asset file has one row per timestamp and this reduces to the
    obvious thing. A cross-sectional file has one row per (timestamp, symbol),
    and every quantity derived from `len(rows)` is then wrong by the width of
    the panel: the naive interval collapses toward zero, which silently shrinks
    the purge gap by the same factor and lets training rows sit inside the test
    period's forward window.
    """
    unique = np.unique(timestamps)
    if len(unique) < 2:
        return 0.0, float(len(timestamps)), len(unique)
    span_seconds = float(unique.max() - unique.min()) / 1000.0
    interval = span_seconds / (len(unique) - 1)
    return interval, len(timestamps) / len(unique), len(unique)

# The longest BACKWARD-looking window any feature uses (rv_60s). A row younger
# than this reports padded zeros for its slowest features.
#
# This is NOT the forward label horizon, and conflating the two was a bug:
# rows used to be dropped when `history_seconds < horizon`, which silently
# discarded the entire dataset at any horizon beyond FeatureEngine's 300s of
# retained mid history. A label's forward window is resolved by the recorder's
# own mid buffer and needs no feature history at all.
FEATURE_WARMUP_SECONDS = 60.0

try:  # Single source of truth for fees; see backend/config.py.
    from config import COST_MAKER_MAKER_BPS, ROUND_TRIP_COST_BPS

    DEFAULT_COST_BPS = float(ROUND_TRIP_COST_BPS)
    MAKER_ONLY_COST_BPS = float(COST_MAKER_MAKER_BPS)
except Exception:  # pragma: no cover - keeps the analysis tools standalone
    DEFAULT_COST_BPS = 10.0
    MAKER_ONLY_COST_BPS = 1.2


def horizons_in(fieldnames: List[str]) -> List[str]:
    """The forward horizons a header actually carries, as `300s`-style tags."""
    return [name[len("fwd_ret_bps_"):] for name in fieldnames
            if name.startswith("fwd_ret_bps_")]


def load_rows(data_dir: Path, pattern: str = "features-*.csv",
              target_column: Optional[str] = None) -> List[dict]:
    """Concatenate the recorded CSVs, skipping files of the wrong vintage.

    The files in `data/` are NOT guaranteed to share a schema. Every change to
    `BLOFIN_LABEL_HORIZONS` starts a new generation of label columns, and the
    old files stay on disk carrying their old names. Concatenating them blind
    and then indexing the requested horizon raises `KeyError` thousands of rows
    into the run - and that is the benign failure. The dangerous one is silent:
    feature columns used to be read off `rows[0]`, so a schema change that only
    reordered or renamed inputs would build a matrix whose columns mean
    different things in different halves of the data.

    So each file is admitted on its own header. A file without the requested
    horizon is skipped, loudly and with its row count, because "your corpus is
    half the size the banner implies" is exactly the kind of thing that must
    not be discovered after the conclusion has been drawn.
    """
    files = sorted(glob.glob(str(data_dir / pattern)))
    if not files:
        raise SystemExit(
            f"No files matching {pattern} in {data_dir}.\n"
            "Run the bot first (python backend/live-chart.py) to record data."
        )

    rows: List[dict] = []
    kept: List[Tuple[str, int]] = []
    skipped: List[Tuple[str, int, List[str]]] = []
    for path in files:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            if target_column is not None and target_column not in fields:
                skipped.append((path, sum(1 for _ in reader), horizons_in(fields)))
                continue
            file_rows = list(reader)
        rows.extend(file_rows)
        kept.append((path, len(file_rows)))

    print(f"Loaded {len(rows):,} rows from {len(kept)} file(s):")
    for path, count in kept:
        print(f"  {os.path.basename(path):<28} {count:>9,} rows  "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")

    if skipped:
        total = sum(count for _, count, _ in skipped)
        print(f"\nSkipped {len(skipped)} file(s), {total:,} rows, with no "
              f"{target_column} column - recorded under other label horizons:")
        for path, count, available in skipped:
            have = ", ".join(available) if available else "no forward labels"
            print(f"  {os.path.basename(path):<28} {count:>9,} rows  has: {have}")
        print("  backend/analysis/replay.py can re-derive them from the raw "
              "archive at the\n  current horizons if you want them back.")

    if not rows:
        available = sorted({tag for _, _, tags in skipped for tag in tags})
        raise SystemExit(
            f"\nNo rows carry {target_column}. Horizons present on disk: "
            f"{', '.join(available) if available else 'none'}."
        )
    return rows


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def build_matrix(
    rows: List[dict], horizon: float
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Return (X, forward_returns, feature_names, timestamps), warmup dropped."""
    tag = f"{horizon:g}s".replace(".", "p")
    target_column = f"fwd_ret_bps_{tag}"

    # Schema off the INTERSECTION of every row, not off `rows[0]`. A corpus
    # spanning a horizon change holds more than one header, and taking the
    # first row's columns as the schema is how a `KeyError` ends up thousands
    # of rows into the run - or worse, how a renamed feature ends up silently
    # meaning two different things in two halves of the matrix. `load_rows`
    # already drops whole files of the wrong vintage; this is the backstop for
    # a caller that did not pass `target_column`.
    schemas = {frozenset(row) for row in rows}
    shared = frozenset.intersection(*schemas) if schemas else frozenset()
    if target_column not in shared:
        available = sorted({key[len("fwd_ret_bps_"):] for schema in schemas
                            for key in schema if key.startswith("fwd_ret_bps_")})
        raise SystemExit(
            f"No column {target_column}. Available horizons: {available}"
        )
    if len(schemas) > 1:
        print(f"Note: {len(schemas)} different headers in this corpus; using "
              f"the {len(shared)} columns common to all of them.")

    label_columns = {
        key for key in shared if key.startswith(("fwd_ret_bps_", "label_"))
    }
    # Ordered by the first row's layout so the report reads in file order,
    # restricted to what every row actually has.
    feature_names = [
        key
        for key in rows[0]
        if key in shared
        and key not in EXCLUDED_FEATURES
        and key not in label_columns
    ]

    kept: List[dict] = []
    dropped_invalid = dropped_warmup = 0
    for row in rows:
        if row.get("is_valid", "True") not in ("True", "true", "1"):
            dropped_invalid += 1
            continue
        # Warmup rows report 0.0 for FEATURE windows longer than the history
        # behind them. That is padding, not a measurement — training on it
        # teaches the model that "no history" means "no move". The bar is the
        # slowest feature (rv_60s), not the forward horizon; see
        # FEATURE_WARMUP_SECONDS.
        if to_float(row.get("history_seconds", "0")) < FEATURE_WARMUP_SECONDS:
            dropped_warmup += 1
            continue
        kept.append(row)

    if dropped_invalid or dropped_warmup:
        print(
            f"\nDropped {dropped_invalid:,} invalid and {dropped_warmup:,} warmup rows "
            f"-> {len(kept):,} usable."
        )
    if len(kept) < 500:
        raise SystemExit(
            f"Only {len(kept)} usable rows. Collect more data before drawing "
            "any conclusion - a few minutes of recording proves nothing."
        )

    X = np.array(
        [[to_float(row[name]) for name in feature_names] for row in kept], dtype=float
    )
    y = np.array([to_float(row[target_column]) for row in kept], dtype=float)
    timestamps = np.array([to_float(row["ts"]) for row in kept], dtype=float)

    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not finite.all():
        print(f"Dropped {(~finite).sum():,} rows containing non-finite values.")
    return X[finite], y[finite], feature_names, timestamps[finite]


def drop_constant_features(
    X: np.ndarray, names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """A zero-variance column carries no information and breaks the solver."""
    keep = X.std(axis=0) > 0
    if not keep.all():
        removed = [name for name, k in zip(names, keep) if not k]
        print(f"Dropped {len(removed)} constant feature(s): {', '.join(removed)}")
    return X[:, keep], [name for name, k in zip(names, keep) if k]


def describe(
    y: np.ndarray, timestamps: np.ndarray, horizon: float, threshold_bps: float
) -> Tuple[float, int]:
    span_seconds = (timestamps.max() - timestamps.min()) / 1000.0
    interval, rows_per_ts, n_unique = panel_geometry(timestamps)
    # Counted in distinct timestamps, not rows. For a cross-sectional panel
    # that treats one whole cross-section as a single observation, which
    # understates the true figure - the safe direction to be wrong in.
    eff_n = effective_sample_size(n_unique, horizon, interval)

    print("\n" + "=" * 72)
    print("DATA")
    print("=" * 72)
    print(f"  rows                  {len(y):,}")
    if rows_per_ts > 1.01:
        print(f"  cross-section         {n_unique:,} timestamps x "
              f"{rows_per_ts:.1f} rows each")
    print(f"  time span             {span_seconds / 3600:.2f} hours")
    print(f"  mean sample interval  {interval * 1000:.0f} ms")
    print(f"  forward horizon       {horizon:g}s")
    print(f"  EFFECTIVE sample size {eff_n:,}   <- use this, not the row count")
    print(
        f"     (adjacent rows share {max(0.0, horizon - interval):.2f}s of their "
        f"{horizon:g}s forward window, so they are not independent)"
    )

    up = (y >= threshold_bps).sum()
    down = (y <= -threshold_bps).sum()
    flat = len(y) - up - down
    print(f"\n  forward return (bps)  mean {y.mean():+.3f}   std {y.std():.3f}")
    print(f"  label balance         up {up:,} ({up/len(y):.1%})  "
          f"down {down:,} ({down/len(y):.1%})  flat {flat:,} ({flat/len(y):.1%})")

    if span_seconds < 6 * 3600:
        print(
            "\n  WARNING: under 6 hours of data. Results here are indicative at "
            "best -\n  a single market session is not evidence of a persistent edge."
        )
    return interval, eff_n


def information_coefficients(
    X: np.ndarray, y: np.ndarray, names: List[str], eff_n: int
) -> List[Tuple[str, float, float, float]]:
    print("\n" + "=" * 72)
    print("INFORMATION COEFFICIENTS  (feature vs forward return)")
    print("=" * 72)
    print(f"  {'feature':<24} {'pearson':>9} {'spearman':>9} {'t(eff)':>8}   note")
    print("  " + "-" * 66)

    results = []
    for index, name in enumerate(names):
        column = X[:, index]
        p = pearson(column, y)
        s = spearman(column, y)
        t = correlation_tstat(s, eff_n)
        results.append((name, p, s, t))

    for name, p, s, t in sorted(results, key=lambda r: -abs(r[2])):
        note = ""
        if abs(s) > SUSPICIOUS_IC:
            note = "<-- SUSPICIOUS, check for leakage"
        elif abs(t) > 3:
            note = "significant"
        elif abs(t) > 2:
            note = "weak"
        print(f"  {name:<24} {p:>+9.4f} {s:>+9.4f} {t:>+8.2f}   {note}")

    print(
        "\n  For reference: single-feature |IC| of 0.01-0.06 is a realistic, "
        "usable\n  signal at these horizons. |IC| > 0.20 is nearly always a bug."
    )
    return results


def leakage_checks(
    ics: List[Tuple[str, float, float, float]], y: np.ndarray, eff_n: int
) -> bool:
    print("\n" + "=" * 72)
    print("LEAKAGE CHECKS")
    print("=" * 72)
    clean = True

    suspicious = [(n, s) for n, _, s, _ in ics if abs(s) > SUSPICIOUS_IC]
    if suspicious:
        clean = False
        print("  FAIL  Implausibly high correlations:")
        for name, s in suspicious:
            print(f"          {name}: spearman {s:+.4f}")
        print("        A feature this predictive is usually the label in "
              "disguise.\n        Check it is not computed from data after ts.")
    else:
        print("  pass  No feature exceeds the plausibility threshold "
              f"(|IC| <= {SUSPICIOUS_IC}).")

    if eff_n < 1000:
        clean = False
        print(f"  FAIL  Effective sample size is only {eff_n:,}. Too small to "
              "conclude anything.")
    else:
        print(f"  pass  Effective sample size {eff_n:,} is adequate.")

    # Near-zero variance in the target means a dead or illiquid market.
    if y.std() < 1e-6:
        clean = False
        print("  FAIL  Forward returns are ~constant. Was the market open?")
    else:
        print(f"  pass  Forward returns vary (std {y.std():.3f} bps).")
    return clean


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    *,
    horizon: float,
    interval: float,
    cost_bps: float,
    rows_per_timestamp: float = 1.0,
) -> Dict[str, float]:
    # In ROWS, so it has to scale with the width of the panel.
    purge = max(1, int(math.ceil(
        horizon / max(interval, 1e-9) * max(1.0, rows_per_timestamp)
    )))
    train, test = purged_split(len(y), train_fraction=0.7, purge_rows=purge)

    print("\n" + "=" * 72)
    print("OUT-OF-SAMPLE TEST  (time-ordered, purged - never shuffled)")
    print("=" * 72)
    print(f"  train rows   {train.stop - train.start:,}")
    print(f"  purge gap    {purge:,} rows (~{horizon:g}s, so train labels cannot")
    print(f"               overlap test features)")
    print(f"  test rows    {test.stop - test.start:,}")

    if test.stop - test.start < 200:
        raise SystemExit("Test set too small to evaluate. Collect more data.")

    X_train, X_test = standardize(X[train], X[test])
    y_train_binary = (y[train] > 0).astype(float)
    y_test_binary = (y[test] > 0).astype(float)

    weights = fit_logistic(X_train, y_train_binary, l2=1.0)
    scores = predict_proba(X_test, weights)

    model_auc = auc(y_test_binary, scores)
    accuracy = ((scores > 0.5) == y_test_binary).mean()
    majority = max(y_test_binary.mean(), 1 - y_test_binary.mean())

    print("\n  Logistic regression (linear baseline)")
    print(f"    AUC                  {model_auc:.4f}   (0.5 = coin flip)")
    print(f"    accuracy             {accuracy:.4f}")
    print(f"    majority-class rate  {majority:.4f}   <- must beat this")
    print(f"    edge over majority   {accuracy - majority:+.4f}")

    print("\n  Largest standardised coefficients:")
    ranked = sorted(zip(names, weights[1:]), key=lambda kv: -abs(kv[1]))
    for name, weight in ranked[:8]:
        print(f"    {name:<24} {weight:+.4f}")

    return economic_test(scores, y[test], cost_bps, model_auc, accuracy, majority)


def economic_test(
    scores: np.ndarray,
    forward_returns: np.ndarray,
    cost_bps: float,
    model_auc: float,
    accuracy: float,
    majority: float,
) -> Dict[str, float]:
    """The test that actually decides whether to keep going.

    Accuracy is not profitability. What matters is the mean forward move
    conditional on the model firing, measured against what a round trip costs.
    """
    print("\n" + "=" * 72)
    print("ECONOMIC TEST  (does the edge survive costs?)")
    print("=" * 72)

    deciles = decile_returns(scores, forward_returns)
    print("  Mean forward return (bps) by predicted-probability decile:")
    for index, value in enumerate(deciles):
        bar = "#" * int(min(abs(value) * 12, 40))
        print(f"    d{index + 1:<2} {value:>+8.3f}  {bar}")

    monotonic = float(pearson(np.arange(len(deciles), dtype=float), deciles))
    print(f"\n  decile monotonicity  {monotonic:+.3f}   "
          "(want strongly positive; a top decile with no")
    print("                                     structure beneath it is usually outliers)")

    top = forward_returns[scores >= np.quantile(scores, 0.9)]
    bottom = forward_returns[scores <= np.quantile(scores, 0.1)]
    long_edge = float(top.mean())
    short_edge = float(-bottom.mean())

    print(f"\n  top decile mean move      {long_edge:+.3f} bps  (would go long)")
    print(f"  bottom decile mean move   {short_edge:+.3f} bps  (would go short)")
    print(f"  round-trip cost           {cost_bps:.3f} bps")
    print(f"  net long edge             {long_edge - cost_bps:+.3f} bps")
    print(f"  net short edge            {short_edge - cost_bps:+.3f} bps")

    tradeable = max(long_edge, short_edge) > cost_bps

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if model_auc < 0.505:
        print("  NO SIGNAL. AUC is indistinguishable from a coin flip.")
        print("  Do not build a model on this. Options: collect more data across")
        print("  varied conditions, try a shorter horizon, or add features (queue")
        print("  position, book slope, cross-venue) - the current set is not enough.")
    elif not tradeable:
        print("  STATISTICAL SIGNAL, BUT NOT TRADEABLE.")
        print(f"  The model predicts direction (AUC {model_auc:.3f}) but the moves it")
        print(f"  finds ({max(long_edge, short_edge):.3f} bps) are smaller than costs")
        print(f"  ({cost_bps:.3f} bps). This is the most common outcome, and it is a")
        print("  real result - the edge exists but the fee schedule eats it.")
        print("  Next: maker-only execution to cut costs, or a longer horizon where")
        print("  moves are larger. Do NOT proceed to live trading.")
    else:
        print(f"  PROMISING. AUC {model_auc:.3f}, net edge "
              f"{max(long_edge, short_edge) - cost_bps:+.3f} bps after costs.")
        print("  Worth building a proper model. Before believing it:")
        print("    - confirm it holds on a DIFFERENT day, not just this test split")
        print("    - re-run with realistic fills, not mid-price (you will not get mid)")
        print("    - check the edge is not concentrated in a few minutes of one event")

    print(
        "\n  Remember this is a linear model on mid-price moves with no slippage,\n"
        "  no queue position, and no adverse selection. Live results will be\n"
        "  worse than this - the only question is by how much."
    )
    return {
        "auc": model_auc,
        "accuracy": accuracy,
        "majority": majority,
        "long_edge_bps": long_edge,
        "short_edge_bps": short_edge,
        "monotonicity": monotonic,
        "tradeable": float(tradeable),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--horizon", type=float, default=900.0,
                        help="Forward horizon in seconds (must exist in the CSV).")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help=f"Round-trip cost in bps, from backend/config.py "
                             f"(default {DEFAULT_COST_BPS:g} = taker both sides). "
                             f"Pass {MAKER_ONLY_COST_BPS:g} for the maker-only "
                             f"case, but only once a passive execution engine "
                             f"exists and its fill rate has been measured.")
    parser.add_argument("--threshold-bps", type=float, default=DEFAULT_COST_BPS,
                        help="Move size counted as up/down for the balance report.")
    args = parser.parse_args(argv)

    tag = f"{args.horizon:g}s".replace(".", "p")
    rows = load_rows(args.data_dir, target_column=f"fwd_ret_bps_{tag}")
    X, y, names, timestamps = build_matrix(rows, args.horizon)
    X, names = drop_constant_features(X, names)

    interval, eff_n = describe(y, timestamps, args.horizon, args.threshold_bps)
    _, rows_per_ts, _ = panel_geometry(timestamps)
    ics = information_coefficients(X, y, names, eff_n)
    clean = leakage_checks(ics, y, eff_n)

    if not clean:
        print("\nLeakage checks failed. Fix those before trusting anything below.\n")

    evaluate(X, y, names, horizon=args.horizon, interval=interval,
             cost_bps=args.cost_bps, rows_per_timestamp=rows_per_ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
