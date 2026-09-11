"""Does knowing WHEN the range holds make the fade pay? The money test for `contained`.

    python backend\\analysis\\range_gated.py
    python backend\\analysis\\range_gated.py --keep 0.3 --symbols BTCUSDT,ETHUSDT

Where this comes from
---------------------
`range_harness.py` scored three targets on 35 symbols over five years
(2026-09-11). Two collapsed and one held:

    centre      IC +0.011, under its own shuffled control  -> nothing there
    width       IC +0.600, measured worth -15.9 bps        -> skill, not money
    contained   IC +0.267, 35/35 symbols positive          -> the live one

`contained` is not a level forecast. It says whether the trailing range will
still hold over the next 24 hours, which gates WHEN the fade runs rather than
where its orders sit - the tradeable version of "use it in a sideways market",
and the part of that idea never actually tested.

An IC is not money, and this repo has a long list of statistically real effects
that died on execution cost. So this runs the fade twice over the same
out-of-sample bars, once ungated and once with entries allowed only when the
model says the range holds, and reports the difference.

The two ways a gate can fool you
--------------------------------
**It can just trade less.** Fewer trades at the same edge per trade is not an
improvement, it is a smaller position in the same losing strategy. So the
report leads with mean bps per TRADE, and prints the trade count beside it.

**It can be a lucky subset.** A gate that keeps half the bars will, by chance
alone, sometimes keep the better half. The control keeps the same NUMBER of
bars at shuffled times: same reduction in trading, none of the timing. A gate
worth having has to beat that, not just beat trading everything.

The threshold comes from the TRAIN predictions only (`--keep` of them), never
from the test distribution - choosing the cut on the data being scored is how a
gate is made to look decisive after the fact.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.range_backtest import (  # noqa: E402
    STOP,
    TARGET,
    Costs,
    Ohlc,
    RollingRange,
    Trades,
    paired_difference,
    simulate,
    summarise,
)
from analysis.range_harness import (  # noqa: E402
    Dataset,
    Ridge,
    build_dataset,
    split_timestamp,
)
from analysis.range_information import load_cached  # noqa: E402
from trading.strategies.range_trade.levels import RangeParams  # noqa: E402

MINUTE_MS = 60_000
DEFAULT_SYMBOLS = ("BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,"
                   "DOGEUSDT,ADAUSDT,LINKUSDT,LTCUSDT,AVAXUSDT")


def allow_mask(bar_ts: np.ndarray, row_ts: np.ndarray, keep: np.ndarray,
               sample_minutes: int) -> np.ndarray:
    """Per-bar entry permission from per-row decisions.

    A row decided at time T used only bars before T, so it governs the bars
    from T until the next decision - never the bars before it, which is where
    a gate would otherwise quietly read the future.
    """
    allow = np.zeros(len(bar_ts), dtype=bool)
    starts = np.searchsorted(bar_ts, row_ts)
    for start, permitted in zip(starts, keep):
        if permitted:
            allow[start:start + sample_minutes] = True
    return allow


@dataclass
class SymbolRun:
    symbol: str
    ungated: Trades
    gated: Trades
    control: Trades
    allowed_share: float = float("nan")


def fit_gate(datasets: Sequence[Dataset], *, train_fraction: float, keep: float,
             l2: float = 1.0) -> Tuple[Ridge, np.ndarray, np.ndarray, float, int]:
    """(model, mean, deviation, threshold, split_ts) - all from TRAIN rows only."""
    split = split_timestamp(datasets, train_fraction)
    horizon_ms = datasets[0].horizon_minutes * MINUTE_MS
    cutoff = split - horizon_ms

    X = np.vstack([dataset.X[dataset.ts < cutoff] for dataset in datasets])
    y = np.concatenate([dataset.targets["contained"][dataset.ts < cutoff]
                        for dataset in datasets])
    if len(X) < 50:
        raise SystemExit(
            f"Only {len(X)} training rows. Use more days or more symbols.")
    mean, deviation = X.mean(axis=0), X.std(axis=0)
    deviation[deviation == 0] = 1.0

    model = Ridge(l2=l2)
    model.fit((X - mean) / deviation, y)
    # The cut is a quantile of what the model said in TRAINING. Taking it from
    # the test predictions would tune the gate on the data it is scored on.
    threshold = float(np.quantile(model.predict((X - mean) / deviation), 1.0 - keep))
    return model, mean, deviation, threshold, split


def run_symbol(ohlc: Ohlc, dataset: Dataset, rolling: RollingRange,
               params: RangeParams, costs: Costs, *, model: Ridge,
               mean: np.ndarray, deviation: np.ndarray, threshold: float,
               split_ts: int, leverage: float, optimistic: bool,
               rng: np.random.Generator) -> SymbolRun:
    test = dataset.ts >= split_ts
    start = int(np.searchsorted(ohlc.ts, split_ts))
    predictions = model.predict((dataset.X[test] - mean) / deviation)
    keep = predictions >= threshold

    gate = allow_mask(ohlc.ts, dataset.ts[test], keep, dataset.sample_minutes)
    # Same number of permitted rows, shuffled in time: the gate that trades as
    # little without knowing when.
    control_gate = allow_mask(ohlc.ts, dataset.ts[test], rng.permutation(keep),
                              dataset.sample_minutes)

    def run(allow: Optional[np.ndarray]) -> Trades:
        return simulate(ohlc, rolling, params, costs, optimistic=optimistic,
                        leverage=leverage, start=start, allow=allow)

    return SymbolRun(symbol=dataset.symbol, ungated=run(None), gated=run(gate),
                     control=run(control_gate),
                     allowed_share=float(np.mean(keep)) if len(keep) else float("nan"))


def report(runs: Sequence[SymbolRun], *, keep: float, optimistic: bool,
           draws: int, seed: int, split_ts: int) -> None:
    rng = np.random.default_rng(seed + 3)
    width = 92
    print("\n" + "=" * width)
    print(f"GATED RANGE FADE  entries allowed on the top {keep:.0%} of "
          f"`contained` predictions")
    print("=" * width)
    bracket = "optimistic" if optimistic else "pessimistic"
    split = datetime.fromtimestamp(split_ts / 1000, tz=timezone.utc)
    print(f"  out of sample from {split:%Y-%m-%d}, {bracket} fills, "
          f"mean bps per TRADE\n")
    print(f"  {'symbol':<12}{'ungated n':>10}{'ungated':>9}{'gated n':>9}"
          f"{'gated':>9}{'ctrl n':>8}{'ctrl':>9}{'gated-ungated':>15}")
    print("  " + "-" * (width - 4))
    for run in runs:
        def mean(trades: Trades) -> float:
            return float(trades.net_bps.mean()) if len(trades) else float("nan")
        difference = mean(run.gated) - mean(run.ungated)
        print(f"  {run.symbol:<12}{len(run.ungated):>10,}{mean(run.ungated):>+9.1f}"
              f"{len(run.gated):>9,}{mean(run.gated):>+9.1f}"
              f"{len(run.control):>8,}{mean(run.control):>+9.1f}"
              f"{difference:>+15.1f}")

    pooled = {name: Trades.concat([getattr(run, name) for run in runs])
              for name in ("ungated", "gated", "control")}
    summaries = {name: summarise(trades, rng, draws) for name, trades in pooled.items()}
    print("\n  POOLED, 95% interval from resampling whole days across all symbols")
    for name in ("ungated", "gated", "control"):
        summary = summaries[name]
        print(f"    {name:<9}{summary.mean_bps:>+8.1f} "
              f"[{summary.low_bps:>+7.1f}, {summary.high_bps:>+7.1f}]   "
              f"{summary.trades:,} trades, win {summary.win_rate:.0%}, "
              f"stops {summary.shares[STOP]:.0%}")

    versus_ungated = paired_difference(pooled["gated"], pooled["ungated"], rng, draws)
    versus_control = paired_difference(pooled["gated"], pooled["control"], rng, draws)
    print(f"\n    gated - ungated  {versus_ungated[0]:>+8.1f} "
          f"[{versus_ungated[1]:>+7.1f}, {versus_ungated[2]:>+7.1f}]   "
          "<- is the timing worth anything")
    print(f"    gated - control  {versus_control[0]:>+8.1f} "
          f"[{versus_control[1]:>+7.1f}, {versus_control[2]:>+7.1f}]   "
          "<- against trading as little at shuffled times")

    kept = float(np.nanmean([run.allowed_share for run in runs]))
    reduction = 1 - summaries["gated"].trades / max(summaries["ungated"].trades, 1)
    print(f"\n    the gate permitted {kept:.0%} of decisions and removed "
          f"{reduction:.0%} of the trades")

    print("\n" + "=" * width)
    print("VERDICT")
    print("=" * width)
    gated, ungated = summaries["gated"], summaries["ungated"]
    if gated.trades == 0:
        print("  The gate allowed nothing. Raise --keep.")
        return
    if np.isfinite(gated.low_bps) and gated.low_bps > 0:
        print(f"  THE GATED FADE MAKES MONEY OUT OF SAMPLE: {gated.mean_bps:+.1f} bps "
              f"per trade\n  [{gated.low_bps:+.1f}, {gated.high_bps:+.1f}], against "
              f"{ungated.mean_bps:+.1f} ungated.")
        if np.isfinite(versus_control[1]) and versus_control[1] > 0:
            print("  It also beats a gate that trades as little at shuffled times, so "
                  "the timing\n  is doing the work rather than the reduction in "
                  "trading. Confirm on a later\n  period before sizing anything.")
        else:
            print("  But it does NOT separate from a gate that trades as little at "
                  "shuffled times,\n  so what looks like timing may be the smaller "
                  "trade count.")
    elif np.isfinite(versus_ungated[1]) and versus_ungated[1] > 0:
        print(f"  The gate HELPS but does not rescue it: {versus_ungated[0]:+.1f} bps "
              f"per trade\n  [{versus_ungated[1]:+.1f}, {versus_ungated[2]:+.1f}] "
              f"better than ungated, and still {gated.mean_bps:+.1f} overall.")
    else:
        print(f"  THE GATE DOES NOT PAY FOR THE FADE. {gated.mean_bps:+.1f} bps per "
              f"trade gated against\n  {ungated.mean_bps:+.1f} ungated, a difference "
              f"of {versus_ungated[0]:+.1f} "
              f"[{versus_ungated[1]:+.1f}, {versus_ungated[2]:+.1f}].")
        print("  A real forecast of whether the range holds is not the same thing as\n"
              "  an edge in fading it - the fade still pays the same spread and the\n"
              "  same stops on the trades the gate keeps.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cache", type=Path, default=repo_root / "data" / "cache")
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--sample-minutes", type=int, default=60)
    parser.add_argument("--entry-frac", type=float, default=0.25)
    parser.add_argument("--stop-frac", type=float, default=0.25)
    parser.add_argument("--target-frac", type=float, default=0.5)
    parser.add_argument("--min-width-bps", type=float, default=50.0)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--through-bps", type=float, default=1.0)
    parser.add_argument("--keep", type=float, default=0.5,
                        help="Share of decisions the gate permits, cut on TRAIN.")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--optimistic", action="store_true")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    reasons = []
    if not 0 < args.keep <= 1:
        reasons.append("--keep must be in (0, 1]")
    if not 0 < args.train_fraction < 1:
        reasons.append("--train-fraction must be between 0 and 1")
    if reasons:
        raise SystemExit("Refusing to run:\n  - " + "\n  - ".join(reasons))

    from config import MAKER_FEE_BPS, TAKER_FEE_BPS, VIP_TIER

    costs = Costs(maker_bps=float(MAKER_FEE_BPS), taker_bps=float(TAKER_FEE_BPS),
                  slippage_bps=args.slippage_bps, through_bps=args.through_bps)
    end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    start = end - timedelta(days=args.days - 1)
    lookback = int(round(args.lookback_hours * 60))
    params = RangeParams(lookback_minutes=lookback, entry_frac=args.entry_frac,
                         stop_frac=args.stop_frac, target_frac=args.target_frac,
                         hold_minutes=lookback, min_width_bps=args.min_width_bps)
    problems = params.problems()
    if problems:
        raise SystemExit("Refusing to run:\n  - " + "\n  - ".join(problems))

    series: Dict[str, Ohlc] = {}
    datasets: List[Dataset] = []
    for symbol in [part.strip().upper() for part in args.symbols.split(",") if part.strip()]:
        ohlc = load_cached(symbol, start, end, args.cache)
        if len(ohlc) == 0:
            print(f"  {symbol:<12} nothing cached - skipped")
            continue
        dataset = build_dataset(ohlc, horizon_minutes=lookback,
                                lookback_minutes=lookback,
                                sample_minutes=args.sample_minutes, symbol=symbol)
        if not len(dataset):
            continue
        series[symbol] = ohlc
        datasets.append(dataset)
        print(f"  {symbol:<12}{len(dataset):>8,} decisions from {len(ohlc):,} bars")
    if not datasets:
        raise SystemExit("No symbol produced rows.")

    model, mean, deviation, threshold, split_ts = fit_gate(
        datasets, train_fraction=args.train_fraction, keep=args.keep)
    print(f"\n  gate trained on rows before {datetime.fromtimestamp(split_ts / 1000, tz=timezone.utc):%Y-%m-%d}, "
          f"threshold {threshold:.3f} (top {args.keep:.0%} of train predictions)")

    rng = np.random.default_rng(args.seed)
    runs = []
    for dataset in datasets:
        ohlc = series[dataset.symbol]
        rolling = RollingRange.build(ohlc, lookback)
        runs.append(run_symbol(ohlc, dataset, rolling, params, costs, model=model,
                               mean=mean, deviation=deviation, threshold=threshold,
                               split_ts=split_ts, leverage=args.leverage,
                               optimistic=args.optimistic, rng=rng))

    report(runs, keep=args.keep, optimistic=args.optimistic, draws=args.draws,
           seed=args.seed, split_ts=split_ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
