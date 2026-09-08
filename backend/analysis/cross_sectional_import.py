"""Cross-sectional panel: predict which coin outperforms, not where BTC goes.

    python backend\\analysis\\cross_sectional_import.py
    python backend\\analysis\\cross_sectional_import.py --start 2023-01-01 --end 2026-09-06
    python backend\\analysis\\cross_sectional_import.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

Then the same gate and the same trainer, unchanged:

    python backend\\analysis\\check_features.py --data-dir data\\cross\\... --horizon 900
    python backend\\analysis\\train_model.py    --data-dir data\\cross\\... --horizon 900

Why this exists
---------------
`bars_import.py` asks "will BTC be higher in 15 minutes?" A year of data and
two model classes said no, and said it with intervals tight enough to believe.
That is the expected answer: BTC-USDT perp is among the most arbitraged
instruments in existence, and its direction is the hardest thing in this market
to forecast.

This asks a different question: **will SOL outperform LINK over the next 15
minutes?** The label is each symbol's forward return minus the cross-sectional
mean, so the market factor - "crypto went up" - is subtracted out rather than
predicted. What remains is dispersion, which is both larger than BTC's own
drift and less efficiently priced.

The features are the same ones, converted to cross-sectional ranks. A raw
`ret_60m` of +40bps means nothing on its own; being the *strongest* of ten
majors over the last hour is a statement about relative positioning, and it is
scale-free, regime-robust and comparable across a four-year sample.

What changes downstream
-----------------------
Rows become one per (timestamp, symbol), which breaks two assumptions that were
safe for a single asset. Both are fixed in `check_features.panel_geometry`:

  * The purge gap is measured in ROWS. With ten symbols per timestamp a naive
    gap is ten times too short in time, and training rows end up inside the
    test period's forward window.
  * Bootstrap resampling must draw whole timestamps, not rows. Ten symbols
    observed at one instant are ten correlated measurements of one moment;
    resampling them independently reports an interval about sqrt(10) too
    narrow.

The cost of being right
-----------------------
A cross-sectional position is two legs, so it pays two round trips. At VIP 1
that is 2.4bps maker or 20bps taker, and `--threshold-bps` defaults to the
latter accordingly. The edge has to clear double what a directional trade
needed - which is the price of not having to know where the market is going.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bars_import import (  # noqa: E402
    FIRST_DATE,
    Fetcher,
    build_rows,
    daily_urls,
    daterange,
    feature_columns,
    horizon_tag,
    kline_urls,
    load_bars,
    load_metrics,
    load_premium,
    parse_date,
)
from analysis.importer_core import ensure_clean_output  # noqa: E402

# Ten most liquid USDT-M perpetuals with the full klines / metrics /
# premiumIndex set, confirmed against the bucket listing on 2026-09-07.
DEFAULT_UNIVERSE = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
)

# Columns `bars_import` writes that describe the row rather than the market.
META_COLUMNS = ("ts", "received_ts", "mid", "is_valid", "history_seconds",
                "vol_regime")

# Raw features whose cross-sectional MEAN is kept as a universe-level column.
# Identical across symbols at a timestamp, so they carry no cross-sectional
# information alone - but a tree can condition on them, which is the point:
# "strongest of ten" means something different in a falling market.
UNIVERSE_FEATURES = ("ret_60m", "rv_60m")


def per_symbol_features() -> List[str]:
    return [name for name in feature_columns(with_depth=False)
            if name not in META_COLUMNS]


# ---------------------------------------------------------------------------
# Per-symbol build
# ---------------------------------------------------------------------------


class SymbolPanel:
    """One symbol's rows, as arrays rather than dicts.

    Dicts are convenient and far too expensive here: ten symbols of a year at
    five-minute sampling is over a million rows, and a million 40-key dicts is
    several gigabytes before anything has been computed.
    """

    __slots__ = ("ts", "features", "forward", "mid")

    def __init__(self, ts, features, forward, mid):
        self.ts: np.ndarray = ts
        self.features: np.ndarray = features
        self.forward: Dict[str, np.ndarray] = forward
        self.mid: np.ndarray = mid


def build_symbol(
    symbol: str,
    days: Sequence[date],
    fetcher: Fetcher,
    *,
    horizons: Tuple[float, ...],
    sample_minutes: int,
    feature_names: Sequence[str],
) -> Optional[SymbolPanel]:
    """Download and featurise one symbol through the existing bar pipeline."""
    print(f"\n{symbol}")
    kline_paths = fetcher.fetch_all(kline_urls(symbol, days, "klines", "1m"))
    premium_paths = fetcher.fetch_all(
        kline_urls(symbol, days, "premiumIndexKlines", "1h")
    )
    metric_paths = fetcher.fetch_all(daily_urls(symbol, days, "metrics"))

    bars = load_bars(kline_paths)
    if len(bars) < 1500:
        print(f"  only {len(bars)} bars; skipping {symbol}")
        return None
    premium_ts, premium_values = load_premium(premium_paths)
    metric_ts, metric_rows = load_metrics(metric_paths)

    dropped = {"warmup": 0, "bar_gap": 0, "metrics": 0, "premium": 0,
               "depth": 0, "no_future": 0}
    timestamps: List[int] = []
    mids: List[float] = []
    feature_rows: List[List[float]] = []
    forward: Dict[str, List[float]] = {horizon_tag(h): [] for h in horizons}

    for row in build_rows(
        bars, metric_ts, metric_rows, premium_ts, premium_values, [], [],
        horizons=horizons, threshold_bps=0.0, sample_minutes=sample_minutes,
        with_depth=False, dropped=dropped,
    ):
        timestamps.append(int(row["ts"]))
        mids.append(float(row["mid"]))
        feature_rows.append([float(row[name]) for name in feature_names])
        for tag in forward:
            forward[tag].append(float(row[f"fwd_ret_bps_{tag}"]))

    if not timestamps:
        print(f"  no usable rows for {symbol}: {dropped}")
        return None
    print(f"  {len(timestamps):,} rows   dropped {dropped}")
    return SymbolPanel(
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(feature_rows, dtype=np.float32),
        {tag: np.asarray(values, dtype=np.float32)
         for tag, values in forward.items()},
        np.asarray(mids, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Cross-sectional transforms
# ---------------------------------------------------------------------------


def align(
    panels: Dict[str, SymbolPanel], n_features: int, tags: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Stack symbols onto one timestamp grid.

    Returns (timestamps, features, forward, mid, symbol order) where features
    is [n_timestamps, n_symbols, n_features] and missing observations are NaN.
    """
    symbols = sorted(panels)
    grid = np.unique(np.concatenate([panels[s].ts for s in symbols]))
    shape = (len(grid), len(symbols))

    features = np.full(shape + (n_features,), np.nan, dtype=np.float32)
    forward = np.full(shape + (len(tags),), np.nan, dtype=np.float32)
    mid = np.full(shape, np.nan, dtype=np.float64)

    for column, symbol in enumerate(symbols):
        panel = panels[symbol]
        rows = np.searchsorted(grid, panel.ts)
        features[rows, column, :] = panel.features
        mid[rows, column] = panel.mid
        for index, tag in enumerate(tags):
            forward[rows, column, index] = panel.forward[tag]
    return grid, features, forward, mid, symbols


def cross_sectional_rank(values: np.ndarray) -> np.ndarray:
    """Rank along the symbol axis, scaled to [-1, +1]; NaN stays NaN.

    Ranks rather than z-scores: a z-score is dominated by whichever coin had an
    outlier that minute, and this data has an outlier most minutes. A rank says
    "third strongest of ten", which is the statement being made and is stable
    across a four-year sample that spans very different volatility regimes.
    """
    present = ~np.isnan(values)
    count = present.sum(axis=1, keepdims=True).astype(np.float32)

    # NaNs sort last, so they never displace a real observation's rank.
    filled = np.where(present, values, np.inf)
    order = np.argsort(filled, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order,
        np.broadcast_to(np.arange(values.shape[1]), values.shape).copy(),
        axis=1,
    )

    scaled = np.where(
        count > 1,
        2.0 * ranks.astype(np.float32) / np.maximum(count - 1.0, 1.0) - 1.0,
        0.0,
    )
    return np.where(present, scaled, np.nan).astype(np.float32)


def relative_forward(forward: np.ndarray) -> np.ndarray:
    """Forward return minus the equal-weighted cross-sectional mean.

    This subtraction is the whole idea. It removes the market factor from the
    label, so the model is never rewarded for knowing that everything went up
    together - only for knowing which names went up more.
    """
    with np.errstate(invalid="ignore"):
        market = np.nanmean(forward, axis=1, keepdims=True)
    return forward - market


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def output_columns(feature_names: Sequence[str],
                   horizons: Sequence[float]) -> List[str]:
    columns = ["ts", "received_ts", "symbol", "mid", "is_valid",
               "history_seconds", "vol_regime", "n_symbols"]
    columns += [f"xs_{name}" for name in feature_names]
    columns += [f"universe_{name}" for name in UNIVERSE_FEATURES]
    for horizon in horizons:
        tag = horizon_tag(horizon)
        columns += [f"fwd_ret_bps_{tag}", f"label_{tag}"]
    return columns


def write_panel(
    path: Path,
    grid: np.ndarray,
    ranked: np.ndarray,
    universe: np.ndarray,
    relative: np.ndarray,
    mid: np.ndarray,
    symbols: Sequence[str],
    feature_names: Sequence[str],
    horizons: Sequence[float],
    *,
    min_symbols: int,
    threshold_bps: float,
) -> Tuple[int, int]:
    columns = output_columns(feature_names, horizons)
    tags = [horizon_tag(h) for h in horizons]
    present = ~np.isnan(mid)
    counts = present.sum(axis=1)
    usable = counts >= min_symbols

    written = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for index in np.flatnonzero(usable):
            timestamp = int(grid[index])
            width = int(counts[index])
            for column, symbol in enumerate(symbols):
                if not present[index, column]:
                    continue
                values = ranked[index, column, :]
                forwards = relative[index, column, :]
                if not np.all(np.isfinite(values)) or not np.all(
                    np.isfinite(forwards)
                ):
                    continue
                row: Dict[str, object] = {
                    "ts": timestamp,
                    "received_ts": timestamp,
                    "symbol": symbol,
                    "mid": float(mid[index, column]),
                    "is_valid": True,
                    # The panel's warmup is the per-symbol builder's warmup,
                    # which already required a full 1440-minute lookback.
                    "history_seconds": 86_400.0,
                    "vol_regime": "unknown",
                    "n_symbols": width,
                }
                for position, name in enumerate(feature_names):
                    row[f"xs_{name}"] = round(float(values[position]), 6)
                for position, name in enumerate(UNIVERSE_FEATURES):
                    row[f"universe_{name}"] = round(
                        float(universe[index, position]), 4
                    )
                for position, tag in enumerate(tags):
                    value = float(forwards[position])
                    row[f"fwd_ret_bps_{tag}"] = round(value, 4)
                    row[f"label_{tag}"] = (
                        1 if value >= threshold_bps
                        else -1 if value <= -threshold_bps else 0
                    )
                writer.writerow(row)
                written += 1
    return written, int(usable.sum())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_end = datetime.now(timezone.utc).date() - timedelta(days=1)
    default_start = default_end - timedelta(days=364)

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--start", type=parse_date, default=default_start)
    parser.add_argument("--end", type=parse_date, default=default_end)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=repo_root / "data" / "cache")
    parser.add_argument("--sample-minutes", type=int, default=15,
                        help="One row per symbol every N minutes (default 15). "
                             "Ten symbols multiply the row count, and every "
                             "downstream tool loads the file into memory.")
    parser.add_argument("--horizons", default="300,900,1800")
    parser.add_argument("--threshold-bps", type=float, default=20.0,
                        help="Move counted as up/down. Defaults to TWO taker "
                             "round trips, because a cross-sectional position "
                             "is two legs.")
    parser.add_argument("--min-symbols", type=int, default=8,
                        help="Drop timestamps with a thinner cross-section "
                             "than this; a rank out of three means little.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if len(symbols) < 3:
        raise SystemExit("A cross-section needs at least three symbols.")
    if args.end < args.start:
        raise SystemExit("--end is before --start.")

    horizons = tuple(float(p) for p in args.horizons.split(",") if p.strip())
    tags = [horizon_tag(h) for h in horizons]
    feature_names = per_symbol_features()

    earliest = max(FIRST_DATE["klines"], FIRST_DATE["metrics"],
                   FIRST_DATE["premiumIndexKlines"])
    if args.start < earliest:
        print(f"--start {args.start} precedes the archive; clamping to {earliest}.")
        args.start = earliest

    out_dir = args.out or (
        repo_root / "data" / "cross"
        / f"{len(symbols)}sym-{args.start.isoformat()}-{args.end.isoformat()}"
    )
    ensure_clean_output(out_dir)

    days = daterange(args.start, args.end)
    print(f"{len(symbols)} symbols  {args.start} .. {args.end}  "
          f"({len(days)} days)\n  {', '.join(symbols)}")

    panels: Dict[str, SymbolPanel] = {}
    for symbol in symbols:
        fetcher = Fetcher(args.cache / symbol, force=args.force,
                          workers=args.workers)
        panel = build_symbol(
            symbol, days, fetcher, horizons=horizons,
            sample_minutes=max(1, args.sample_minutes),
            feature_names=feature_names,
        )
        if panel is not None:
            panels[symbol] = panel

    if len(panels) < args.min_symbols:
        raise SystemExit(
            f"Only {len(panels)} symbol(s) produced rows, below "
            f"--min-symbols {args.min_symbols}."
        )

    print("\nAligning and ranking...")
    grid, features, forward, mid, order = align(panels, len(feature_names), tags)
    del panels

    universe_index = [feature_names.index(name) for name in UNIVERSE_FEATURES]
    with np.errstate(invalid="ignore"):
        universe = np.nanmean(features[:, :, universe_index], axis=1)

    ranked = np.empty_like(features)
    for position in range(features.shape[2]):
        ranked[:, :, position] = cross_sectional_rank(features[:, :, position])
    del features

    relative = relative_forward(forward)
    del forward

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (f"features-{args.start.isoformat()}_"
                      f"{args.end.isoformat()}.csv")
    written, timestamps_kept = write_panel(
        path, grid, ranked, universe, relative, mid, order,
        feature_names, horizons,
        min_symbols=args.min_symbols, threshold_bps=args.threshold_bps,
    )
    if not written:
        path.unlink(missing_ok=True)
        raise SystemExit("No rows survived alignment.")

    span_s = float(grid[-1] - grid[0]) / 1000.0
    print(f"\nWrote {written:,} rows across {timestamps_kept:,} timestamps")
    print(f"  symbols         {len(order)}  ({', '.join(order)})")
    print(f"  span            {span_s / 86400:.1f} days")
    print(f"  file            {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    for horizon in horizons:
        print(f"  independent cross-sections at {horizon:g}s: "
              f"{span_s / horizon:,.0f}")
    print(f"\nRun the gate:\n  python backend\\analysis\\check_features.py "
          f"--data-dir {out_dir} --horizon {horizons[len(horizons) // 2]:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
