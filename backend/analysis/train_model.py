"""Roadmap step 7: a gradient-boosted model, honestly evaluated.

    python backend\\analysis\\train_model.py --data-dir data\\bars\\BTCUSDT-... --horizon 900
    python backend\\analysis\\train_model.py --data-dir data\\bars\\... --horizon 1800 --seed 7

`check_features.py` fits a logistic regression. That is the right baseline and
the wrong ceiling: it cannot represent an interaction, and every feature in the
bar set is plausibly conditional on another (open-interest change means one
thing in a low-volatility regime and the opposite in a high one). This fits
LightGBM on the same matrix, with the same purged split discipline, and asks
one question: **does the non-linear model beat the linear one by more than
noise?**

What this does that a naive `lgb.train()` does not
--------------------------------------------------
1. **Three-way time-ordered split, purged twice.** Train / validation / test,
   with a horizon-sized gap either side of the validation block. Early stopping
   reads the validation set, which makes validation a *used* set - it has seen
   the model. Only the final block is untouched, and only its numbers are
   reported as out-of-sample.

2. **The test set is decimated to non-overlapping rows** before any economic
   number is computed. At a 900s horizon sampled every 300s, three consecutive
   rows share most of their forward window; averaging over them produces a
   confident-looking mean built from a third as many real observations.

3. **A bootstrap interval on the top-decile move.** A point estimate of
   "+1.4 bps" is not a result. The interval is what says whether it differs
   from zero, and it is computed on the decimated rows so resampling is valid.

4. **A shuffled-label control, run several times.** The identical pipeline is
   retrained on shuffled labels. Whatever that produces is the noise floor of
   this procedure on this data, and a real model has to clear its own control
   rather than merely clear zero.

   It is run `--runs` times, not once, because one control is one draw from the
   noise distribution rather than the distribution itself. Measured on this
   repo's own data at a 900s horizon, four control seeds produced top deciles
   of -0.69, +1.03, +1.55 and +1.12 bps - and at one seed the control *beat*
   the real model. A single control had made the same model look like a result.

5. **The model is run several times too**, and its seed spread is reported next
   to its score. Predictions are ensembled across seeds, which is both standard
   practice and the only version of the number that is stable enough to act on.

6. **The train/test gap is reported.** A large one means the trees memorised
   the training period, and the honest response is fewer leaves, not more
   rounds.

Reading the output
------------------
The comparison table is the whole point. If LightGBM's top decile does not
clearly beat both the logistic baseline and the shuffled control, the answer is
that the non-linearity was not there - which is a real finding, and cheaper to
learn here than after building an execution engine around it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.check_features import (  # noqa: E402
    DEFAULT_COST_BPS,
    MAKER_ONLY_COST_BPS,
    build_matrix,
    drop_constant_features,
    load_rows,
)
from analysis.stats import (  # noqa: E402
    auc,
    decile_returns,
    fit_logistic,
    predict_proba,
    standardize,
)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - exercised only without the dependency
    raise SystemExit(
        "LightGBM is not installed in this environment.\n"
        "  python -m pip install lightgbm\n"
        "It is the model the roadmap names, and the only extra dependency this "
        "script needs beyond numpy."
    )


# Deliberately conservative. The signal in this data is a fraction of a basis
# point against a 24bps standard deviation; a model with the capacity to fit it
# also has the capacity to fit the noise around it, and will prefer to.
DEFAULT_PARAMS = {
    "objective": "huber",       # 900s BTC returns are fat-tailed; L2 would
                                # spend most of its capacity on a few moves.
    "metric": "l2",
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbosity": -1,
}


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def three_way_split(
    n_rows: int, *, train_fraction: float, valid_fraction: float, purge_rows: int
) -> Tuple[slice, slice, slice]:
    """Time-ordered train / validation / test, purged at both seams.

    The validation block is not out-of-sample in any useful sense: early
    stopping picks the iteration count that suits it, so it has influenced the
    model. Keeping a third block that nothing has read is the only way the
    final number means what it says.
    """
    train_end = int(n_rows * train_fraction)
    valid_start = train_end + purge_rows
    valid_end = valid_start + int(n_rows * valid_fraction)
    test_start = valid_end + purge_rows
    if test_start >= n_rows:
        raise SystemExit(
            f"Not enough rows ({n_rows:,}) for a three-way split with a "
            f"{purge_rows:,}-row purge gap. Import a longer date range."
        )
    return (
        slice(0, train_end),
        slice(valid_start, valid_end),
        slice(test_start, n_rows),
    )


def decimate(count: int, stride: int) -> np.ndarray:
    """Indices of non-overlapping rows: every `stride`-th one."""
    return np.arange(0, count, max(1, stride))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def bootstrap_top_decile(
    scores: np.ndarray,
    returns: np.ndarray,
    *,
    draws: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean top-decile forward return with a 95% bootstrap interval.

    Rows must already be decimated to non-overlapping forward windows, or the
    resampling assumption is false and the interval comes out too narrow.
    """
    rng = np.random.default_rng(seed)
    count = len(returns)
    cut = max(1, count // 10)
    point = float(np.sort(returns[np.argsort(scores)][-cut:]).mean())

    means = np.empty(draws)
    for draw in range(draws):
        pick = rng.integers(0, count, count)
        sample_scores, sample_returns = scores[pick], returns[pick]
        order = np.argsort(sample_scores)
        means[draw] = sample_returns[order][-cut:].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return point, float(low), float(high)


def paired_top_decile_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    returns: np.ndarray,
    *,
    draws: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap `top_decile(a) - top_decile(b)` on the SAME resampled rows.

    Why paired rather than comparing two separate intervals: both models are
    scored on one test set, so most of the uncertainty in each is the same
    uncertainty - which fortnight the test block happened to land on, which
    handful of large moves fell in the top decile. Resampling them jointly
    cancels that shared noise and leaves the question actually being asked,
    which is whether A ranks these particular rows better than B does.

    Comparing A's point estimate against B's interval instead answers a
    different and much harsher question, and would reject a real difference
    whenever the test set is small - exactly when this matters most.
    """
    rng = np.random.default_rng(seed)
    count = len(returns)
    cut = max(1, count // 10)

    def top(scores: np.ndarray, values: np.ndarray) -> float:
        return float(values[np.argsort(scores)][-cut:].mean())

    point = top(scores_a, returns) - top(scores_b, returns)
    differences = np.empty(draws)
    for draw in range(draws):
        pick = rng.integers(0, count, count)
        sampled = returns[pick]
        differences[draw] = (top(scores_a[pick], sampled)
                             - top(scores_b[pick], sampled))
    low, high = np.percentile(differences, [2.5, 97.5])
    return point, float(low), float(high)


def score_report(
    name: str,
    scores: np.ndarray,
    returns: np.ndarray,
    *,
    seed: int,
) -> Dict[str, float]:
    binary = (returns > 0).astype(float)
    point, low, high = bootstrap_top_decile(scores, returns, seed=seed)
    deciles = decile_returns(scores, returns)
    monotonic = float(
        np.corrcoef(np.arange(len(deciles)), deciles)[0, 1]
    ) if len(deciles) > 1 else 0.0
    return {
        "name": name,
        "auc": float(auc(binary, scores)),
        "top_decile_bps": point,
        "ci_low_bps": low,
        "ci_high_bps": high,
        "monotonicity": monotonic,
        "deciles": [float(value) for value in deciles],
    }


def print_comparison(results: List[Dict[str, float]], costs: Dict[str, float],
                     runs: int) -> None:
    print("\n" + "=" * 78)
    print(f"OUT-OF-SAMPLE COMPARISON  (final block, non-overlapping rows, "
          f"{runs}-seed ensembles)")
    print("=" * 78)
    header = f"  {'model':<22}{'AUC':>8}{'top decile':>14}{'95% interval':>22}"
    for label in costs:
        header += f"{'net @ ' + label:>16}"
    print(header)
    print("  " + "-" * 74)
    for result in results:
        interval = ("[" + format(result["ci_low_bps"], "+.3f") + ", "
                    + format(result["ci_high_bps"], "+.3f") + "]")
        line = (f"  {result['name']:<22}{result['auc']:>8.4f}"
                f"{result['top_decile_bps']:>+13.3f} {interval:>22}")
        for cost in costs.values():
            line += f"{result['top_decile_bps'] - cost:>+16.3f}"
        print(line)

    print("\n  per-seed top decile (bps) - a wide spread means the number is "
          "seed noise:")
    for result in results:
        spread = result.get("per_seed")
        if not spread:
            print(f"    {result['name']:<22} deterministic")
            continue
        cells = " ".join(f"{value:+6.2f}" for value in spread)
        print(f"    {result['name']:<22} {cells}"
              f"   (min {min(spread):+.2f}, max {max(spread):+.2f})")

    print("\n  decile means (bps), lowest model score first:")
    for result in results:
        cells = " ".join(f"{value:+6.2f}" for value in result["deciles"])
        print(f"    {result['name']:<22} {cells}")
        print(f"    {'':<22} monotonicity {result['monotonicity']:+.3f}")


def print_paired(pairs: List[Tuple[str, Tuple[float, float, float]]]) -> None:
    print("\n  paired comparisons (same resampled rows, so shared test-set "
          "noise cancels):")
    for label, (point, low, high) in pairs:
        verdict_word = "significant" if low > 0 else "not significant"
        print(f"    {label:<34}{point:>+8.3f} bps  "
              f"[{low:+.3f}, {high:+.3f}]   {verdict_word}")


def verdict(
    model: Dict[str, float],
    baseline: Dict[str, float],
    control: Dict[str, float],
    costs: Dict[str, float],
    vs_control: Tuple[float, float, float],
    vs_baseline: Tuple[float, float, float],
) -> None:
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    beats_zero = model["ci_low_bps"] > 0
    # The paired lower bound, not a point estimate against an interval.
    beats_control = vs_control[1] > 0
    beats_baseline = vs_baseline[1] > 0
    control_ceiling = max(
        control.get("per_seed") or [control.get("top_decile_bps", 0.0)]
    )
    cheapest = min(costs.values())
    cheapest_label = min(costs, key=costs.get)

    if not beats_zero:
        print("  NOT DISTINGUISHABLE FROM ZERO.")
        print(f"  The top-decile interval [{model['ci_low_bps']:+.3f}, "
              f"{model['ci_high_bps']:+.3f}] bps includes zero, so this model's")
        print("  edge is not measurable on the data it was given, before costs")
        print("  are even considered. More rounds will not fix that; more data,")
        print("  or different features, might.")
    elif not beats_control:
        print("  NOT SEPARABLE FROM SHUFFLED LABELS.")
        print(f"  The model scores {model['top_decile_bps']:+.3f} bps and the "
              f"shuffled control")
        print(f"  {control['top_decile_bps']:+.3f} bps, and the paired "
              f"difference is {vs_control[0]:+.3f} bps with a")
        print(f"  95% interval of [{vs_control[1]:+.3f}, {vs_control[2]:+.3f}] "
              f"- which includes zero.")
        print(f"  The best single control seed reached {control_ceiling:+.3f} "
              f"bps on labels")
        print("  with the signal shuffled out of them. The model's number is")
        print("  not yet evidence of anything: change the features, not the")
        print("  hyperparameters.")
    elif model["ci_low_bps"] <= cheapest:
        print("  REAL BUT NOT TRADEABLE.")
        print(f"  The edge clears zero and clears the shuffled control, but the")
        print(f"  lower bound {model['ci_low_bps']:+.3f} bps does not clear the "
              f"cheapest")
        print(f"  round trip ({cheapest_label}, {cheapest:g} bps). Cheaper "
              f"execution or a")
        print("  longer horizon, not a bigger model.")
    else:
        print("  PROMISING.")
        print(f"  Clears zero, clears the shuffled control, and the lower bound")
        print(f"  {model['ci_low_bps']:+.3f} bps clears the {cheapest_label} round "
              f"trip ({cheapest:g} bps).")
        print("  Confirm on a different date range before believing it, then")
        print("  paper-trade it. Do NOT size this from the backtest.")

    print(f"\n  vs the linear baseline: "
          f"{'better' if beats_baseline else 'no better'} "
          f"(paired {vs_baseline[0]:+.3f} bps, "
          f"[{vs_baseline[1]:+.3f}, {vs_baseline[2]:+.3f}])")
    if not beats_baseline:
        print("  The non-linearity is not measurably there. That is a finding,")
        print("  not a failure - it says the feature set, not the model class,")
        print("  is the constraint.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_booster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    names: List[str],
    params: Dict[str, object],
    *,
    rounds: int,
    patience: int,
    quiet: bool = False,
) -> "lgb.Booster":
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=names)
    valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=names,
                            reference=train_set)
    callbacks = [lgb.early_stopping(patience, verbose=not quiet)]
    if not quiet:
        callbacks.append(lgb.log_evaluation(period=200))
    return lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=callbacks,
    )


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--horizon", type=float, default=900.0)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--maker-cost-bps", type=float, default=MAKER_ONLY_COST_BPS)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--min-data-in-leaf", type=int, default=None)
    parser.add_argument("--objective", default=None,
                        help="LightGBM objective (default huber).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5,
                        help="Seeds to train and ensemble, for the model and "
                             "the control alike. One run of either is a draw "
                             "from a distribution, not a measurement.")
    parser.add_argument("--no-control", action="store_true",
                        help="Skip the shuffled-label control. Don't.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Directory for the saved model and metadata.")
    args = parser.parse_args(argv)

    rows = load_rows(args.data_dir)
    X, y, names, timestamps = build_matrix(rows, args.horizon)
    X, names = drop_constant_features(X, names)
    del rows

    intervals = np.diff(timestamps)
    interval_s = float(np.median(intervals)) / 1000.0 if len(intervals) else 1.0
    stride = max(1, math.ceil(args.horizon / max(interval_s, 1e-9)))

    params = dict(DEFAULT_PARAMS)
    params["seed"] = args.seed
    params["bagging_seed"] = args.seed
    params["feature_fraction_seed"] = args.seed
    for key, value in (("learning_rate", args.learning_rate),
                       ("num_leaves", args.num_leaves),
                       ("min_data_in_leaf", args.min_data_in_leaf),
                       ("objective", args.objective)):
        if value is not None:
            params[key] = value

    train, valid, test = three_way_split(
        len(y), train_fraction=args.train_fraction,
        valid_fraction=args.valid_fraction, purge_rows=stride,
    )

    print("\n" + "=" * 78)
    print("DATA AND SPLIT")
    print("=" * 78)
    print(f"  rows                 {len(y):,}")
    print(f"  features             {len(names)}")
    print(f"  horizon              {args.horizon:g}s")
    print(f"  sample interval      {interval_s:g}s")
    print(f"  overlap stride       {stride} rows share a forward window")
    print(f"  train                {train.stop - train.start:,} rows")
    print(f"  purge gap            {stride:,} rows")
    print(f"  validation           {valid.stop - valid.start:,} rows "
          f"(early stopping reads this)")
    print(f"  purge gap            {stride:,} rows")
    print(f"  test                 {test.stop - test.start:,} rows "
          f"(untouched until the end)")
    keep = decimate(test.stop - test.start, stride)
    print(f"  test, decimated      {len(keep):,} non-overlapping rows")
    if len(keep) < 200:
        raise SystemExit(
            f"Only {len(keep)} independent test observations. Any number "
            "computed on that is noise; import a longer date range."
        )

    # LightGBM handles raw scales, but the logistic baseline does not, and both
    # must see the same information - so standardise once, from train only.
    X_train, X_valid, X_test = standardize(X[train], X[valid], X[test])
    y_train, y_valid, y_test = y[train], y[valid], y[test]

    runs = max(1, args.runs)
    print("\n" + "=" * 78)
    print(f"TRAINING  ({runs} seed{'s' if runs > 1 else ''})")
    print("=" * 78)
    started = time.time()

    boosters = []
    train_score_runs = []
    test_score_runs = []
    for offset in range(runs):
        seeded = dict(params)
        for key in ("seed", "bagging_seed", "feature_fraction_seed"):
            seeded[key] = args.seed + offset
        booster = train_booster(
            X_train, y_train, X_valid, y_valid, names, seeded,
            rounds=args.rounds, patience=args.patience, quiet=offset > 0,
        )
        boosters.append(booster)
        train_score_runs.append(
            booster.predict(X_train, num_iteration=booster.best_iteration)
        )
        test_score_runs.append(
            booster.predict(X_test, num_iteration=booster.best_iteration)
        )
        print(f"  seed {args.seed + offset}: best iteration "
              f"{booster.best_iteration}")
    print(f"  trained in           {time.time() - started:.1f}s")

    booster = boosters[0]
    train_scores = np.mean(train_score_runs, axis=0)
    test_scores = np.mean(test_score_runs, axis=0)
    train_auc = auc((y_train > 0).astype(float), train_scores)
    test_auc_full = auc((y_test > 0).astype(float), test_scores)
    print(f"  AUC train / test     {train_auc:.4f} / {test_auc_full:.4f}"
          f"   (gap {train_auc - test_auc_full:+.4f})")
    if train_auc - test_auc_full > 0.05:
        print("  NOTE: a gap this wide means the trees memorised the training")
        print("        period. Fewer leaves or more min_data_in_leaf, not more"
              " rounds.")

    print("\n  Top features by gain (averaged over seeds):")
    gains = np.mean(
        [each.feature_importance(importance_type="gain") for each in boosters],
        axis=0,
    )
    for name, gain in sorted(zip(names, gains), key=lambda kv: -kv[1])[:12]:
        share = gain / max(gains.sum(), 1e-12) * 100.0
        print(f"    {name:<24} {share:5.1f}%")

    # Everything below is measured on non-overlapping test rows only.
    y_eval = y_test[keep]
    results = []

    weights = fit_logistic(X_train, (y_train > 0).astype(float), l2=1.0)
    results.append(score_report(
        "logistic baseline", predict_proba(X_test, weights)[keep], y_eval,
        seed=args.seed,
    ))
    model_result = score_report(
        "lightgbm", test_scores[keep], y_eval, seed=args.seed,
    )
    model_result["per_seed"] = [
        float(score_report("run", run[keep], y_eval, seed=args.seed)
              ["top_decile_bps"])
        for run in test_score_runs
    ]
    results.append(model_result)

    control_runs: List[np.ndarray] = []
    if args.no_control:
        control = {"name": "shuffled control", "top_decile_bps": 0.0,
                   "ci_high_bps": 0.0, "ci_low_bps": 0.0, "auc": 0.5,
                   "monotonicity": 0.0, "deciles": [], "per_seed": []}
    else:
        print(f"\n  Training {runs} shuffled-label control(s)...")
        for offset in range(runs):
            rng = np.random.default_rng(args.seed + 1000 + offset)
            seeded = dict(params)
            for key in ("seed", "bagging_seed", "feature_fraction_seed"):
                seeded[key] = args.seed + 1000 + offset
            control_booster = train_booster(
                X_train, rng.permutation(y_train),
                X_valid, rng.permutation(y_valid), names, seeded,
                rounds=args.rounds, patience=args.patience, quiet=True,
            )
            control_runs.append(control_booster.predict(
                X_test, num_iteration=control_booster.best_iteration
            ))
        control = score_report(
            "shuffled control", np.mean(control_runs, axis=0)[keep], y_eval,
            seed=args.seed,
        )
        control["per_seed"] = [
            float(score_report("run", run[keep], y_eval, seed=args.seed)
                  ["top_decile_bps"])
            for run in control_runs
        ]
        results.append(control)

    costs = {f"maker {args.maker_cost_bps:g}": args.maker_cost_bps,
             f"taker {args.cost_bps:g}": args.cost_bps}
    print_comparison(results, costs, runs)

    baseline_scores = predict_proba(X_test, weights)[keep]
    model_scores = test_scores[keep]
    control_scores = (np.mean(control_runs, axis=0)[keep]
                      if not args.no_control else model_scores)
    vs_control = paired_top_decile_difference(
        model_scores, control_scores, y_eval, seed=args.seed
    )
    vs_baseline = paired_top_decile_difference(
        model_scores, baseline_scores, y_eval, seed=args.seed
    )
    print_paired([
        ("lightgbm - shuffled control", vs_control),
        ("lightgbm - logistic baseline", vs_baseline),
    ])
    verdict(results[1], results[0], control, costs, vs_control, vs_baseline)

    out_dir = args.out or (repo_root / "models"
                           / f"lgbm-{args.horizon:g}s-seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for offset, each in enumerate(boosters):
        each.save_model(str(out_dir / f"model-seed{args.seed + offset}.txt"),
                        num_iteration=each.best_iteration)
    metadata = {
        "horizon_s": args.horizon,
        "features": names,
        "params": {key: value for key, value in params.items()},
        "runs": runs,
        "best_iterations": [each.best_iteration for each in boosters],
        "sample_interval_s": interval_s,
        "overlap_stride": stride,
        "rows": {"train": train.stop - train.start,
                 "valid": valid.stop - valid.start,
                 "test": test.stop - test.start,
                 "test_decimated": int(len(keep))},
        "train_auc": float(train_auc),
        "test_auc": float(test_auc_full),
        "results": results,
        "paired_vs_control_bps": list(vs_control),
        "paired_vs_baseline_bps": list(vs_baseline),
        "costs_bps": costs,
        "data_dir": str(args.data_dir),
        "seed": args.seed,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\n  Saved model and metadata to {out_dir}")
    print("  The metadata records the feature list and their order. Scoring a")
    print("  live row with the columns in a different order is silent garbage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
