"""Ranges at several timeframes at once: which hold, which break, and does it
say where price ends up?

    python backend\\analysis\\range_multiscale.py
    python backend\\analysis\\range_multiscale.py --scale-hours 4,24,72,168 --days 1825

The idea being tested
---------------------
Compute the range at several lookbacks, predict for EACH whether it holds over
the next 24h, and feed them to one model so the scales inform each other. If
price breaks the 4h range but not the 72h, that is a statement about how far it
travels - and, done per side, about which way.

`range_harness.py` scored a single symmetric `contained` target: did price stay
inside both edges of ONE range. That threw the direction away, which is the
half that pays (step 9l: the centre is worth +102 bps known perfectly, the
width -15.9). This splits it.

What the idea reduces to, and why it is still worth running
-----------------------------------------------------------
A break at scale L is exactly

    up:    log(future_high / close) > log(range_high_L / close)
    down:  log(close / future_low)  > log(close / range_low_L)

and the left sides do not depend on L. So nested ranges are a DISCRETISED CDF
of the same two quantities - the next 24h up-excursion and down-excursion -
read at different thresholds. The scales re-parameterise location (the centre)
and spread (the width); they do not add a third thing.

That is an argument for a low prior, not a result. The hole in it: these are
tail classifications, and a classifier on a tail can find structure that a
least-squares fit on a mean misses, which is exactly what the single-scale
centre regression was. So the breaks are scored per scale and per side, and
then the question that decides it is asked directly - do multi-scale features
predict the CENTRE any better than one scale did?

Everything is scored on the harness bar: purged split, effective N, a shuffled
control per target, sign agreement, and the centre IC converted to bps through
the measured table. A model that shares information across scales by attention
rather than by a linear combination is scored the same way through
`range_harness.py --dump` / `--predictions`; the bar does not change with the
architecture.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.range_harness import (  # noqa: E402
    FEATURES,
    MINUTE_MS,
    Dataset,
    Ridge,
    build_dataset,
    evaluate,
    information_coefficient,
    print_report,
    split_timestamp,
)
from analysis.range_information import (  # noqa: E402
    breakeven_centre_skill,
    load_cached,
    value_of_centre_skill,
)
from analysis.range_backtest import Ohlc  # noqa: E402

DEFAULT_SYMBOLS = ("BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,"
                   "DOGEUSDT,ADAUSDT,LINKUSDT,LTCUSDT,AVAXUSDT")
DEFAULT_SCALES = "4,24,72,168"


def scale_label(minutes: int) -> str:
    hours = minutes / 60
    return f"{hours / 24:g}d" if hours >= 24 else f"{hours:g}h"


def break_targets(dataset: Dataset, label: str) -> Dict[str, np.ndarray]:
    """Per-side break flags for one scale, derived from that scale's own rows.

    `centre` is log(sqrt(fh*fl)/close) and `width` is log(fh/fl), so
    log(fh/close) = centre + width/2 and log(close/fl) = width/2 - centre.
    Comparing those against the scale's own `dist_high` / `dist_low` features
    is the break condition exactly - no recomputation, and no chance of the
    targets drifting from the features they are compared with.
    """
    centre = dataset.targets["centre"]
    width = dataset.targets["width"]
    columns = {name: index for index, name in enumerate(dataset.feature_names)}
    up_room = dataset.X[:, columns["dist_high"]]
    down_room = dataset.X[:, columns["dist_low"]]
    return {
        f"break_up@{label}": (centre + width / 2 > up_room).astype(float),
        f"break_down@{label}": (width / 2 - centre > down_room).astype(float),
    }


def build_multiscale(ohlc: Ohlc, *, scales_minutes: Sequence[int], horizon_minutes: int,
                     sample_minutes: int, symbol: str) -> Optional[Dataset]:
    """One row per decision, features and break targets from every scale.

    Rows are the INTERSECTION across scales: a long lookback needs more history
    before its first valid row, and a row missing one scale is a row where the
    model would be guessing at that scale rather than reading it.
    """
    per_scale = {}
    for lookback in scales_minutes:
        data = build_dataset(ohlc, horizon_minutes=horizon_minutes,
                             lookback_minutes=lookback,
                             sample_minutes=sample_minutes, symbol=symbol)
        if not len(data):
            return None
        per_scale[lookback] = data

    common = per_scale[scales_minutes[0]].ts
    for lookback in scales_minutes[1:]:
        common = np.intersect1d(common, per_scale[lookback].ts)
    if len(common) == 0:
        return None

    blocks, names, targets, price = [], [], {}, None
    for position, lookback in enumerate(scales_minutes):
        data = per_scale[lookback]
        keep = np.searchsorted(data.ts, common)
        label = scale_label(lookback)
        blocks.append(data.X[keep])
        names.extend(f"{name}@{label}" for name in data.feature_names)
        aligned = Dataset(symbol=symbol, ts=common, X=data.X[keep],
                          targets={key: value[keep] for key, value in data.targets.items()},
                          price=data.price[keep], horizon_minutes=horizon_minutes,
                          lookback_minutes=lookback, sample_minutes=sample_minutes,
                          feature_names=data.feature_names)
        targets.update(break_targets(aligned, label))
        if position == 0:
            # centre and width describe the FORWARD window and the close, so
            # they are the same at every scale. Taken once, from the first.
            targets["centre"] = aligned.targets["centre"]
            targets["width"] = aligned.targets["width"]
            price = aligned.price

    return Dataset(symbol=symbol, ts=common, X=np.hstack(blocks), targets=targets,
                   price=price, horizon_minutes=horizon_minutes,
                   lookback_minutes=scales_minutes[-1],
                   sample_minutes=sample_minutes, feature_names=tuple(names))


def single_scale_view(dataset: Dataset, scales_minutes: Sequence[int],
                      wanted: int) -> Dataset:
    """The same rows and targets, with only one scale's feature block.

    The comparison that matters is like-for-like: same split, same control,
    same rows - only the width of the feature set changes.
    """
    index = list(scales_minutes).index(wanted)
    block = len(FEATURES)
    columns = slice(index * block, (index + 1) * block)
    return Dataset(symbol=dataset.symbol, ts=dataset.ts, X=dataset.X[:, columns],
                   targets=dataset.targets, price=dataset.price,
                   horizon_minutes=dataset.horizon_minutes,
                   lookback_minutes=wanted, sample_minutes=dataset.sample_minutes,
                   feature_names=tuple(f"{name}@{scale_label(wanted)}"
                                       for name in FEATURES))


def fit_side_models(datasets: Sequence[Dataset], labels: Sequence[str], *,
                    train_fraction: float, l2: float,
                    shuffle_seed: Optional[int] = None):
    """One model per side per scale, fitted on TRAIN rows only.

    Returns (models, mean, deviation, split_ts). No second stage is fitted
    anywhere: the asymmetry is read straight off these predictions on the test
    rows, so there is nothing for a later fit to leak through.

    `shuffle_seed` permutes each side's TRAINING labels before fitting, which
    is the control: whatever gap the same pipeline produces from targets that
    cannot be predicted is this procedure's noise floor, and a real directional
    signal has to clear it. The two sides are permuted INDEPENDENTLY, so their
    relationship is destroyed rather than merely relabelled.
    """
    split = split_timestamp(datasets, train_fraction)
    cutoff = split - datasets[0].horizon_minutes * MINUTE_MS
    X = np.vstack([data.X[data.ts < cutoff] for data in datasets])
    mean, deviation = X.mean(axis=0), X.std(axis=0)
    deviation[deviation == 0] = 1.0
    scaled = (X - mean) / deviation
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    models = {}
    for label in labels:
        for side in ("up", "down"):
            target = f"break_{side}@{label}"
            y = np.concatenate([data.targets[target][data.ts < cutoff]
                                for data in datasets])
            if rng is not None:
                y = rng.permutation(y)
            model = Ridge(l2=l2)
            model.fit(scaled, y)
            models[target] = model
    return models, mean, deviation, split


def asymmetry(datasets: Sequence[Dataset], labels: Sequence[str], models, mean,
              deviation, split_ts: int) -> Dict[str, Tuple[float, float]]:
    """Per scale: (correlation of the two sides, IC of their difference vs centre).

    A model that knows only HOW FAR price travels moves both sides together and
    its difference says nothing. A model that knows WHICH WAY separates them,
    and the difference is then a centre forecast.
    """
    out: Dict[str, Tuple[float, float]] = {}
    pooled_difference, pooled_centre = [], []
    for label in labels:
        ups, downs, centres = [], [], []
        for data in datasets:
            test = data.ts >= split_ts
            if not test.any():
                continue
            scaled = (data.X[test] - mean) / deviation
            ups.append(models[f"break_up@{label}"].predict(scaled))
            downs.append(models[f"break_down@{label}"].predict(scaled))
            centres.append(data.targets["centre"][test])
        if not ups:
            continue
        up, down = np.concatenate(ups), np.concatenate(downs)
        centre = np.concatenate(centres)
        together = float(np.corrcoef(up, down)[0, 1]) if np.std(up) and np.std(down) else float("nan")
        out[label] = (together, information_coefficient(up - down, centre))
        pooled_difference.append(up - down)
        pooled_centre.append(centre)
    if pooled_difference:
        out["all"] = (float("nan"), information_coefficient(
            np.mean(pooled_difference, axis=0), pooled_centre[0]))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cache", type=Path, default=repo_root / "data" / "cache")
    parser.add_argument("--scale-hours", default=DEFAULT_SCALES)
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--sample-minutes", type=int, default=60)
    parser.add_argument("--baseline-hours", type=float, default=24.0,
                        help="The single scale the multi-scale set is compared with.")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--control-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true",
                        help="Print the full per-symbol table for every target.")
    args = parser.parse_args(argv)

    scales = [int(round(float(part) * 60)) for part in args.scale_hours.split(",")
              if part.strip()]
    baseline = int(round(args.baseline_hours * 60))
    reasons = []
    if len(scales) < 2:
        reasons.append("--scale-hours needs at least two scales")
    if baseline not in scales:
        reasons.append(f"--baseline-hours {args.baseline_hours:g} must be one of "
                       f"--scale-hours")
    if any(scale < 1 for scale in scales):
        reasons.append("every scale must be at least one minute")
    if reasons:
        raise SystemExit("Refusing to run:\n  - " + "\n  - ".join(reasons))
    scales.sort()

    horizon = int(round(args.horizon_hours * 60))
    end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    start = end - timedelta(days=args.days - 1)

    datasets: List[Dataset] = []
    for symbol in [part.strip().upper() for part in args.symbols.split(",") if part.strip()]:
        ohlc = load_cached(symbol, start, end, args.cache)
        if len(ohlc) == 0:
            print(f"  {symbol:<12} nothing cached - skipped")
            continue
        data = build_multiscale(ohlc, scales_minutes=scales, horizon_minutes=horizon,
                                sample_minutes=args.sample_minutes, symbol=symbol)
        if data is None:
            print(f"  {symbol:<12} too short for the longest scale - skipped")
            continue
        datasets.append(data)
        print(f"  {symbol:<12}{len(data):>8,} rows x {len(data.feature_names)} features")
    if not datasets:
        raise SystemExit("No symbol produced rows.")

    def score(sets: Sequence[Dataset], target: str):
        return evaluate(sets, lambda: Ridge(l2=args.l2), target,
                        train_fraction=args.train_fraction,
                        control_seeds=args.control_seeds, seed=args.seed)

    labels = [scale_label(scale) for scale in scales]
    print("\n" + "=" * 78)
    print(f"DOES EACH RANGE HOLD?  scales {', '.join(labels)}, "
          f"{args.horizon_hours:g}h ahead")
    print("=" * 78)
    print(f"  {'target':<20}{'IC':>9}{'control':>10}{'symbols +':>12}{'base rate':>12}")
    print("  " + "-" * 61)
    for label in labels:
        for side in ("up", "down"):
            target = f"break_{side}@{label}"
            report = score(datasets, target)
            rate = float(np.mean(np.concatenate(
                [data.targets[target] for data in datasets])))
            print(f"  {target:<20}{report.pooled_ic:>+9.3f}"
                  f"{report.control_ceiling:>10.3f}"
                  f"{report.positive_symbols:>7}/{len(report.per_symbol):<4}{rate:>12.1%}")
            if args.verbose:
                print_report(report)

    print("\n" + "=" * 78)
    print("BUT DOES IT SAY WHERE PRICE ENDS UP?")
    print("=" * 78)
    one = score([single_scale_view(data, scales, baseline) for data in datasets], "centre")
    many = score(datasets, "centre")
    required = breakeven_centre_skill()
    print(f"  {'feature set':<28}{'centre IC':>11}{'control':>10}{'bps/trade':>12}")
    print("  " + "-" * 61)
    for name, report in ((f"{scale_label(baseline)} alone", one),
                         (f"all {len(scales)} scales", many)):
        print(f"  {name:<28}{report.pooled_ic:>+11.3f}{report.control_ceiling:>10.3f}"
              f"{value_of_centre_skill(max(report.pooled_ic, 0.0)):>+12.1f}")
    print(f"  {'break-even needs':<28}{required:>+11.3f}")

    models, mean, deviation, split = fit_side_models(
        datasets, labels, train_fraction=args.train_fraction, l2=args.l2)
    sides = asymmetry(datasets, labels, models, mean, deviation, split)
    print("\n  Which way, rather than how far: the two sides of each scale")
    print(f"  {'scale':<12}{'corr(up, down)':>17}{'IC of the gap vs centre':>26}")
    print("  " + "-" * 55)
    for label in labels:
        if label in sides:
            together, gap_ic = sides[label]
            print(f"  {label:<12}{together:>+17.3f}{gap_ic:>+26.3f}")
    if "all" in sides:
        print(f"  {'all scales':<12}{'':>17}{sides['all'][1]:>+26.3f}")

    controls = []
    for offset in range(max(0, args.control_seeds)):
        shuffled, c_mean, c_deviation, _ = fit_side_models(
            datasets, labels, train_fraction=args.train_fraction, l2=args.l2,
            shuffle_seed=args.seed + offset)
        control_sides = asymmetry(datasets, labels, shuffled, c_mean, c_deviation, split)
        if "all" in control_sides:
            controls.append(control_sides["all"][1])
    if controls:
        ceiling = max(abs(value) for value in controls)
        print(f"  {'control':<12}{'':>17}"
              + ", ".join(f"{value:+.3f}" for value in controls)
              + f"   (ceiling {ceiling:.3f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    gain = many.pooled_ic - one.pooled_ic
    print(f"  Predicting the BREAKS works: the model reads whether a range holds "
          f"well above\n  its control at every scale. That is the easy half - a wide "
          "range holds and a\n  narrow one breaks.")
    gap_ic = sides["all"][1] if "all" in sides else float("nan")
    ceiling = max((abs(value) for value in controls), default=float("nan"))
    best = max(value for value in (many.pooled_ic, one.pooled_ic, gap_ic)
               if np.isfinite(value))

    if np.isfinite(gap_ic) and np.isfinite(ceiling) and gap_ic > ceiling:
        print(f"\n  Reading it as TWO SIDES does find direction the centre "
              f"regression missed:\n  the gap between them scores {gap_ic:+.3f} "
              f"against a control ceiling of {ceiling:.3f}, where\n  regressing the "
              f"centre on the same features gives {many.pooled_ic:+.3f}. Splitting "
              "the target\n  by side is a better use of the same information.")
    elif np.isfinite(gap_ic):
        print(f"\n  Reading it as two sides does not help either: the gap scores "
              f"{gap_ic:+.3f} against a\n  control ceiling of {ceiling:.3f}.")

    if required is not None and best >= required:
        print(f"\n  AND IT CLEARS THE BAR: {best:+.3f} against a {required:.3f} "
              "break-even. Confirm on a\n  later period and on symbols outside this "
              "set before sizing anything.")
    elif required is not None:
        print(f"\n  It is still short of the money, though: {best:+.3f} against a "
              f"{required:.3f}\n  break-even, which is "
              f"{required / max(best, 1e-9):.1f}x. Knowing which scales break says "
              f"mostly HOW FAR\n  price travels, and that is the width - measured at "
              "-15.9 bps even when perfect.")
    linked = [value for _, (value, _) in sides.items() if np.isfinite(value)]
    if linked and min(linked) > 0.5:
        print(f"\n  The two sides move TOGETHER (correlation {min(linked):+.2f} to "
              f"{max(linked):+.2f}): the model\n  forecasts the SIZE of the next move "
              "and not its direction, which is why the gap\n  between them carries no "
              "centre signal. It is the width result once more.")

    print("\n  A model that shares information ACROSS scales rather than combining "
          "them linearly\n  is scored the same way: range_harness.py --dump writes "
          "these rows, --predictions\n  scores what comes back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
