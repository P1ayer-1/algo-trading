"""A scoreboard for range forecasts, so model classes get measured, not argued about.

    python backend\\analysis\\range_harness.py --target centre --model ridge
    python backend\\analysis\\range_harness.py --target contained --model ridge
    python backend\\analysis\\range_harness.py --dump data\\range-harness
    python backend\\analysis\\range_harness.py --predictions data\\range-harness\\preds

Why this exists
---------------
`range_information.py` measured what a forecast range is worth (2026-09-11):
perfect knowledge of the next 24h WIDTH is worth -15.9 bps per trade, perfect
knowledge of where the range SITS is worth +102.1, and break-even sits at a
centre IC of 0.177. So one number decides whether any model - ridge,
LightGBM, an attention network, an LLM reading news, an RL policy - can rescue
the range fade, and this scores that number the same way for all of them.

The bar, and why each piece of it is here
-----------------------------------------
* **Purged, time-ordered split.** Never shuffled, and train stops one forward
  horizon before test starts, because a training row's target reaches into the
  test window.
* **Effective sample size.** Rows sampled hourly with a 24h horizon share 23/24
  of their window. A year is ~363 independent windows per symbol, not 8,712
  rows, and a t-statistic on the row count overstates significance ~4.9x.
* **A shuffled-label control, several seeds.** Whatever the same pipeline
  produces on shuffled targets is this procedure's noise floor. One control is
  one draw - step 7 found a seed where the control beat the model.
* **Sign agreement across symbols.** The majors are correlated, so this is
  weaker than N independent tests, and still the cheapest way to tell an edge
  from a week.
* **A money threshold, not a p-value.** A centre IC converts to bps through the
  measured table, so the verdict reads "clears the cost of trading" rather than
  "statistically significant".

Plugging in a model
-------------------
A model here is anything with `fit(X, y)` and `predict(X)` - structural, no base
class, the same convention as the strategy Protocols. For anything that cannot
live in this process (PyTorch, an LLM pipeline, an RL policy), `--dump` writes
features, targets and the exact train/test rows per symbol as .npz, and
`--predictions` scores files written back in that shape. Alignment is CHECKED
against the timestamps rather than trusted, because a prediction file one row
out of step scores like a signal.

The three targets
-----------------
* `centre`   log(geometric centre of the next H hours / price now). The one that
             pays, and the one a symmetric forecast cannot have skill at.
* `width`    log(high/low) over the next H hours. Predictable, measured worth
             below zero - kept so a model claiming range skill has to say which
             half it means.
* `contained` did price stay inside the trailing range for H hours. Not a level
             forecast at all: it gates when the fade runs, which is the part of
             the sideways-market idea that has never been tested.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.range_backtest import Ohlc, contiguous, trailing_max, trailing_min  # noqa: E402
from analysis.range_information import (  # noqa: E402
    breakeven_centre_skill,
    future_max,
    future_min,
    future_valid,
    load_cached,
    value_of_centre_skill,
)
from analysis.stats import (  # noqa: E402
    correlation_tstat,
    effective_sample_size,
    purged_split,
    spearman,
)

MINUTE_MS = 60_000
FEATURES = ("log_width", "rv", "parkinson", "vol_ratio", "ret",
            "pos_in_range", "dist_high", "dist_low", "trend_ratio", "width_ratio")
TARGETS = ("centre", "width", "contained")
DEFAULT_SYMBOLS = ("BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,"
                   "DOGEUSDT,ADAUSDT,LINKUSDT,LTCUSDT,AVAXUSDT")
# The long window a couple of features look back over, in lookbacks.
LONG_MULTIPLE = 4


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------


def trailing_sum(values: np.ndarray, window: int) -> np.ndarray:
    """out[i] = sum(values[i-window:i]) - the PRIOR window, never bar i."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan)
    if window < 1 or n <= window:
        return out
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    out[window:] = cumulative[window:n] - cumulative[0:n - window]
    return out


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    return trailing_sum(values, window) / window


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """Features known at `ts`, targets measured strictly after it."""

    symbol: str
    ts: np.ndarray
    X: np.ndarray
    targets: Dict[str, np.ndarray]
    price: np.ndarray
    horizon_minutes: int
    lookback_minutes: int
    sample_minutes: int
    feature_names: Tuple[str, ...] = FEATURES

    def __len__(self) -> int:
        return len(self.ts)


def build_dataset(ohlc: Ohlc, *, horizon_minutes: int, lookback_minutes: int,
                  sample_minutes: int = 60, symbol: str = "") -> Dataset:
    """One row per `sample_minutes`, features backwards only, targets forwards only."""
    n = len(ohlc)
    horizon, lookback = horizon_minutes, lookback_minutes
    long_window = LONG_MULTIPLE * lookback
    empty = Dataset(symbol, np.empty(0, dtype=np.int64), np.empty((0, len(FEATURES))),
                    {name: np.empty(0) for name in TARGETS}, np.empty(0),
                    horizon, lookback, sample_minutes)
    if n <= long_window + horizon + 1:
        return empty

    high = trailing_max(ohlc.high, lookback)
    low = trailing_min(ohlc.low, lookback)
    high_long = trailing_max(ohlc.high, long_window)
    low_long = trailing_min(ohlc.low, long_window)
    future_high = future_max(ohlc.high, horizon)
    future_low = future_min(ohlc.low, horizon)

    close = ohlc.close
    returns = np.zeros(n)
    returns[1:] = np.log(close[1:] / close[:-1])
    squared = returns ** 2
    short = max(1, lookback // 6)
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = np.sqrt(np.maximum(trailing_mean(squared, lookback), 0.0))
        rv_short = np.sqrt(np.maximum(trailing_mean(squared, short), 0.0))
        parkinson = np.sqrt(
            np.maximum(trailing_mean(np.log(ohlc.high / ohlc.low) ** 2, lookback), 0.0)
            / (4 * np.log(2)))
        previous = np.full(n, np.nan)
        previous[lookback:] = close[:n - lookback]

        log_width = np.log(high / low)
        ret = np.log(close / previous)
        columns = {
            "log_width": log_width,
            "rv": np.log(rv + 1e-12),
            "parkinson": np.log(parkinson + 1e-12),
            "vol_ratio": np.log((rv_short + 1e-12) / (rv + 1e-12)),
            "ret": ret,
            "pos_in_range": (close - low) / (high - low),
            "dist_high": np.log(high / close),
            "dist_low": np.log(close / low),
            "trend_ratio": np.abs(ret) / log_width,
            "width_ratio": np.log(log_width / np.log(high_long / low_long)),
        }
        centre = np.log(np.sqrt(future_high * future_low) / close)
        width = np.log(future_high / future_low)
    contained = ((future_high <= high) & (future_low >= low)).astype(float)

    usable = (contiguous(ohlc.ts, long_window) & future_valid(ohlc.ts, horizon)
              & np.isfinite(centre) & np.isfinite(width))
    stacked = np.column_stack([columns[name] for name in FEATURES])
    usable &= np.all(np.isfinite(stacked), axis=1)

    rows = np.arange(long_window, n - horizon, max(1, sample_minutes))
    rows = rows[usable[rows]]
    return Dataset(symbol=symbol, ts=ohlc.ts[rows], X=stacked[rows],
                   targets={"centre": centre[rows], "width": width[rows],
                            "contained": contained[rows]},
                   price=close[rows], horizon_minutes=horizon,
                   lookback_minutes=lookback, sample_minutes=sample_minutes)


# ---------------------------------------------------------------------------
# Models - structural, inherited from never
# ---------------------------------------------------------------------------


@runtime_checkable
class Model(Protocol):
    """Anything that learns from (X, y) and scores rows. No base class."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class Ridge:
    """L2 least squares. The baseline that can only find linear structure.

    Deliberately the first thing on the board: if it finds nothing, the honest
    reading is usually that there is little to find, not that the model was too
    small. A bigger model has to beat this before capacity is the explanation.
    """

    l2: float = 1.0
    name: str = "ridge"
    weights: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        design = np.hstack([np.ones((len(X), 1)), X])
        penalty = np.eye(design.shape[1]) * self.l2
        penalty[0, 0] = 0.0            # never penalise the intercept
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("fit before predict")
        return np.hstack([np.ones((len(X), 1)), X]) @ self.weights


@dataclass
class Persistence:
    """Predict one feature directly - tomorrow looks like today.

    The right baseline for width, where it is strong (IC ~0.4), and the honest
    null for centre, where the feature is momentum.
    """

    index: int
    name: str = "persistence"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.index]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def information_coefficient(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Spearman, and 0.0 when either side is constant rather than a divide by zero."""
    if len(prediction) < 3 or np.std(prediction) == 0 or np.std(truth) == 0:
        return 0.0
    return spearman(np.asarray(prediction, dtype=float), np.asarray(truth, dtype=float))


@dataclass
class SymbolScore:
    symbol: str
    rows: int
    effective_n: int
    ic: float
    t_stat: float


@dataclass
class Report:
    target: str
    model: str
    per_symbol: List[SymbolScore] = field(default_factory=list)
    rows: int = 0
    effective_n: int = 0
    pooled_ic: float = float("nan")
    pooled_t: float = float("nan")
    control_ics: List[float] = field(default_factory=list)
    bps: float = float("nan")
    required_ic: Optional[float] = None
    split_ts: int = 0

    @property
    def positive_symbols(self) -> int:
        return sum(1 for score in self.per_symbol if score.ic > 0)

    @property
    def control_ceiling(self) -> float:
        return max((abs(ic) for ic in self.control_ics), default=float("nan"))


def split_timestamp(datasets: Sequence[Dataset], train_fraction: float) -> int:
    """One calendar split for every symbol, so no symbol trains on another's future."""
    stamps = np.sort(np.concatenate([dataset.ts for dataset in datasets]))
    train_rows, _ = purged_split(len(stamps), train_fraction=train_fraction)
    return int(stamps[min(train_rows.stop, len(stamps) - 1)])


def evaluate(datasets: Sequence[Dataset], make_model: Callable[[], Model], target: str,
             *, train_fraction: float = 0.7, control_seeds: int = 3,
             seed: int = 7) -> Report:
    """Fit pooled across symbols, score per symbol, against a shuffled control."""
    datasets = [dataset for dataset in datasets if len(dataset)]
    if not datasets:
        raise SystemExit("No dataset has any rows.")
    if target not in TARGETS:
        raise SystemExit(f"Unknown target {target!r}. Known: {', '.join(TARGETS)}")

    horizon_ms = datasets[0].horizon_minutes * MINUTE_MS
    split = split_timestamp(datasets, train_fraction)
    # Purge: a training row's target reaches one horizon forward, so the last
    # horizon before the split cannot be trained on.
    train_cutoff = split - horizon_ms

    train_X, train_y = [], []
    for dataset in datasets:
        mask = dataset.ts < train_cutoff
        train_X.append(dataset.X[mask])
        train_y.append(dataset.targets[target][mask])
    X = np.vstack(train_X)
    y = np.concatenate(train_y)
    if len(X) < len(FEATURES) + 2:
        raise SystemExit(
            f"Only {len(X)} training rows for {len(FEATURES)} features. Use more "
            "days, more symbols, or a smaller --sample-minutes.")

    # Scale on TRAIN only - fitting the scaler on everything leaks the test set.
    mean, deviation = X.mean(axis=0), X.std(axis=0)
    deviation[deviation == 0] = 1.0

    model = make_model()
    model.fit((X - mean) / deviation, y)

    report = Report(target=target, model=getattr(model, "name", type(model).__name__),
                    split_ts=split)
    predictions, truths = [], []
    for dataset in datasets:
        mask = dataset.ts >= split
        if not mask.any():
            continue
        prediction = model.predict((dataset.X[mask] - mean) / deviation)
        truth = dataset.targets[target][mask]
        effective = effective_sample_size(int(mask.sum()),
                                          dataset.horizon_minutes * 60,
                                          dataset.sample_minutes * 60)
        ic = information_coefficient(prediction, truth)
        report.per_symbol.append(SymbolScore(dataset.symbol, int(mask.sum()), effective,
                                             ic, correlation_tstat(ic, effective)))
        predictions.append(prediction)
        truths.append(truth)

    if predictions:
        pooled_prediction = np.concatenate(predictions)
        pooled_truth = np.concatenate(truths)
        report.rows = len(pooled_prediction)
        report.effective_n = sum(score.effective_n for score in report.per_symbol)
        report.pooled_ic = information_coefficient(pooled_prediction, pooled_truth)
        report.pooled_t = correlation_tstat(report.pooled_ic, report.effective_n)

    rng = np.random.default_rng(seed)
    for _ in range(max(0, control_seeds)):
        control = make_model()
        control.fit((X - mean) / deviation, rng.permutation(y))
        control_predictions, control_truths = [], []
        for dataset in datasets:
            mask = dataset.ts >= split
            if not mask.any():
                continue
            control_predictions.append(control.predict((dataset.X[mask] - mean) / deviation))
            control_truths.append(dataset.targets[target][mask])
        if control_predictions:
            report.control_ics.append(information_coefficient(
                np.concatenate(control_predictions), np.concatenate(control_truths)))

    if target == "centre":
        report.required_ic = breakeven_centre_skill()
        report.bps = value_of_centre_skill(max(report.pooled_ic, 0.0))
    return report


def print_report(report: Report) -> None:
    width = 78
    print("\n" + "=" * width)
    print(f"TARGET {report.target}   MODEL {report.model}")
    print("=" * width)
    print(f"  {'symbol':<12}{'rows':>8}{'eff N':>8}{'IC':>9}{'t':>8}")
    print("  " + "-" * 43)
    for score in report.per_symbol:
        print(f"  {score.symbol:<12}{score.rows:>8,}{score.effective_n:>8}"
              f"{score.ic:>+9.3f}{score.t_stat:>8.2f}")
    print("  " + "-" * 43)
    print(f"  {'pooled':<12}{report.rows:>8,}{report.effective_n:>8}"
          f"{report.pooled_ic:>+9.3f}{report.pooled_t:>8.2f}")
    print(f"\n  sign agreement      {report.positive_symbols}/{len(report.per_symbol)} "
          "symbols positive (the majors are correlated - weaker than it looks)")
    if report.control_ics:
        controls = ", ".join(f"{ic:+.3f}" for ic in report.control_ics)
        print(f"  shuffled control    {controls}   (this procedure's noise floor)")
    split = datetime.fromtimestamp(report.split_ts / 1000, tz=timezone.utc)
    print(f"  test starts         {split:%Y-%m-%d}, train purged one horizon before it")


def verdict(report: Report) -> None:
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if report.target != "centre":
        print(f"  IC {report.pooled_ic:+.3f} on `{report.target}`, control ceiling "
              f"{report.control_ceiling:.3f}.")
        if report.target == "width":
            print("  Width skill was measured at -15.9 bps per trade even when "
                  "PERFECT\n  (range_information.py, 2026-09-11). Skill here is not "
                  "money; see `centre`.")
        else:
            print("  `contained` gates WHEN the fade runs rather than where its "
                  "levels sit.\n  Converting this into money needs the gated "
                  "backtest, not this IC alone.")
        return

    required = report.required_ic
    beats_control = report.pooled_ic > report.control_ceiling
    print(f"  centre IC {report.pooled_ic:+.3f} (t {report.pooled_t:.2f} on "
          f"{report.effective_n} independent windows)")
    print(f"  worth about {report.bps:+.1f} bps per trade through the measured table")
    if required is not None:
        print(f"  break-even needs IC {required:.3f}")
    if required is not None and report.pooled_ic >= required and beats_control:
        print("\n  CLEARS THE BAR. Confirm on a later period and on symbols outside "
              "this set\n  before sizing anything: this is the first thing in the "
              "repo to get here.")
    elif not beats_control:
        print(f"\n  DOES NOT BEAT ITS OWN SHUFFLED CONTROL ({report.control_ceiling:.3f}). "
              "Whatever the\n  model found, the same pipeline finds as much in "
              "permuted targets.")
    else:
        print(f"\n  BELOW BREAK-EVEN. A model needs roughly "
              f"{required / max(report.pooled_ic, 1e-9):.1f}x this skill to pay "
              f"for the\n  round trip. Capacity is not obviously the binding "
              "constraint - the sample is:\n  "
              f"{report.effective_n} independent windows resolve an IC of about "
              f"{2 / max(report.effective_n, 1) ** 0.5:.3f} at two standard errors.")


# ---------------------------------------------------------------------------
# The external-model contract
# ---------------------------------------------------------------------------


def dump(datasets: Sequence[Dataset], directory: Path, *, train_fraction: float,
         target: str) -> None:
    """Write features, targets and the exact split per symbol, for other runtimes."""
    directory.mkdir(parents=True, exist_ok=True)
    split = split_timestamp(datasets, train_fraction)
    horizon_ms = datasets[0].horizon_minutes * MINUTE_MS
    for dataset in datasets:
        train = dataset.ts < split - horizon_ms
        test = dataset.ts >= split
        np.savez(directory / f"{dataset.symbol}.npz", ts=dataset.ts, X=dataset.X,
                 price=dataset.price, train=train, test=test,
                 feature_names=np.array(dataset.feature_names),
                 **{f"target_{name}": values for name, values in dataset.targets.items()})
        print(f"  {dataset.symbol:<12}{int(train.sum()):>8,} train "
              f"{int(test.sum()):>8,} test -> {dataset.symbol}.npz")
    print(f"\n  Write predictions back as <SYMBOL>.npz holding `ts` and `prediction` "
          f"for the\n  TEST rows, then score them:\n"
          f"    python backend\\analysis\\range_harness.py --predictions <dir> "
          f"--target {target}")


def score_predictions(datasets: Sequence[Dataset], directory: Path, target: str, *,
                      train_fraction: float) -> Report:
    """Score files written by another runtime. Alignment is checked, not trusted."""
    split = split_timestamp(datasets, train_fraction)
    report = Report(target=target, model=f"external:{directory.name}", split_ts=split)
    predictions, truths, problems = [], [], []
    for dataset in datasets:
        path = directory / f"{dataset.symbol}.npz"
        if not path.exists():
            problems.append(f"{dataset.symbol}: no {path.name}")
            continue
        payload = np.load(path, allow_pickle=False)
        for key in ("ts", "prediction"):
            if key not in payload:
                problems.append(f"{path.name}: no `{key}` array")
        if problems and problems[-1].startswith(path.name):
            continue
        mask = dataset.ts >= split
        expected = dataset.ts[mask]
        if len(payload["ts"]) != len(expected) or not np.array_equal(payload["ts"], expected):
            problems.append(
                f"{path.name}: {len(payload['ts'])} rows against {len(expected)} test "
                "rows, or timestamps out of step - a prediction one row off scores "
                "like a signal")
            continue
        truth = dataset.targets[target][mask]
        ic = information_coefficient(payload["prediction"], truth)
        effective = effective_sample_size(int(mask.sum()), dataset.horizon_minutes * 60,
                                          dataset.sample_minutes * 60)
        report.per_symbol.append(SymbolScore(dataset.symbol, int(mask.sum()), effective,
                                             ic, correlation_tstat(ic, effective)))
        predictions.append(np.asarray(payload["prediction"], dtype=float))
        truths.append(truth)
    if problems:
        raise SystemExit("Refusing to score:\n  - " + "\n  - ".join(problems))
    pooled_prediction = np.concatenate(predictions)
    pooled_truth = np.concatenate(truths)
    report.rows = len(pooled_prediction)
    report.effective_n = sum(score.effective_n for score in report.per_symbol)
    report.pooled_ic = information_coefficient(pooled_prediction, pooled_truth)
    report.pooled_t = correlation_tstat(report.pooled_ic, report.effective_n)
    if target == "centre":
        report.required_ic = breakeven_centre_skill()
        report.bps = value_of_centre_skill(max(report.pooled_ic, 0.0))
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cache", type=Path, default=repo_root / "data" / "cache")
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--sample-minutes", type=int, default=60)
    parser.add_argument("--target", default="centre", choices=TARGETS)
    parser.add_argument("--model", default="ridge", choices=("ridge", "persistence"))
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--control-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dump", type=Path, default=None,
                        help="Write the dataset and split for an external model.")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Score <SYMBOL>.npz files written by an external model.")
    args = parser.parse_args(argv)

    reasons = []
    if not 0 < args.train_fraction < 1:
        reasons.append("--train-fraction must be between 0 and 1")
    if args.sample_minutes < 1:
        reasons.append("--sample-minutes must be at least 1")
    if args.horizon_hours <= 0 or args.lookback_hours <= 0:
        reasons.append("--horizon-hours and --lookback-hours must be positive")
    if args.dump and args.predictions:
        reasons.append("--dump writes a dataset and --predictions scores one; "
                       "do one at a time")
    if reasons:
        raise SystemExit("Refusing to run:\n  - " + "\n  - ".join(reasons))

    end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    start = end - timedelta(days=args.days - 1)
    horizon = int(round(args.horizon_hours * 60))
    lookback = int(round(args.lookback_hours * 60))

    datasets: List[Dataset] = []
    for symbol in [part.strip().upper() for part in args.symbols.split(",") if part.strip()]:
        ohlc = load_cached(symbol, start, end, args.cache)
        if len(ohlc) == 0:
            print(f"  {symbol:<12} nothing cached - skipped")
            continue
        dataset = build_dataset(ohlc, horizon_minutes=horizon, lookback_minutes=lookback,
                                sample_minutes=args.sample_minutes, symbol=symbol)
        print(f"  {symbol:<12}{len(dataset):>8,} rows from {len(ohlc):,} bars")
        if len(dataset):
            datasets.append(dataset)
    if not datasets:
        raise SystemExit(
            "No symbol produced rows.\nFetch bars first: python "
            "backend\\analysis\\fetch_klines.py --symbols <SYMBOLS>")

    if args.dump:
        dump(datasets, args.dump, train_fraction=args.train_fraction, target=args.target)
        return 0

    if args.predictions:
        report = score_predictions(datasets, args.predictions, args.target,
                                   train_fraction=args.train_fraction)
    else:
        if args.model == "ridge":
            def make_model() -> Model:
                return Ridge(l2=args.l2)
        else:
            index = FEATURES.index("log_width" if args.target == "width" else "ret")

            def make_model() -> Model:
                return Persistence(index=index)
        report = evaluate(datasets, make_model, args.target,
                          train_fraction=args.train_fraction,
                          control_seeds=args.control_seeds, seed=args.seed)

    print_report(report)
    verdict(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
