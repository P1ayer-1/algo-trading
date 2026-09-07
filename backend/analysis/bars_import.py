"""Build a training set from Binance's free bar-level archive.

    python backend\\analysis\\bars_import.py
    python backend\\analysis\\bars_import.py --start 2024-01-01 --end 2026-09-06
    python backend\\analysis\\bars_import.py --symbol ETHUSDT --sample-minutes 15
    python backend\\analysis\\bars_import.py --with-depth

Then the usual gate, unchanged:

    python backend\\analysis\\check_features.py --data-dir data\\bars\\BTCUSDT-... --horizon 900

Why this exists
---------------
`binance_import.py` and `tardis_import.py` reconstruct an order book from tick
data, because the features they feed operate on a horizon of seconds. That is
the right tool for a 5-second label and the wrong one for a 30-minute label,
and it is also the expensive one: tick-level L2 history is a $1k/month product,
and the one free source of it (Binance `bookTicker`) stopped being published on
2024-03-30.

At the horizons this project actually has to trade — see the fee arithmetic in
`backend/config.py` — none of that is needed. The information that matters over
15 to 30 minutes is in bars, open interest, positioning and carry, all of which
Binance publishes free, in bulk, with no API key, current to yesterday:

    klines/1m             2,442 daily files   OHLCV, trade count, taker buy volume
    metrics               2,197 daily files   open interest, long/short ratios
    premiumIndexKlines    2,443 daily files   premium / basis, i.e. funding
    bookDepth             1,342 daily files   depth in bands around mid (opt-in)

That is roughly 830 MB for the full history. The binding constraint on how far
back a run can go is `metrics` (from 2020-09-01) or, with --with-depth,
`bookDepth` (from 2023-01-01).

What it produces
----------------
A feature CSV in exactly the format `FeatureRecorder` writes, so
`check_features.py`, `stats.py` and `compact.py` all work on it unchanged. The
columns are different — these are bar features, not microstructure features —
but the contract is the same: `ts`, `mid`, `is_valid`, `history_seconds`, a
block of stationary features, and `fwd_ret_bps_<h>` / `label_<h>` per horizon.

Lookahead discipline
--------------------
This is the one thing here that must not be wrong, and bar data makes it easy
to get wrong in a way that looks like success.

  * Rows are anchored on a kline's **close_time**, never its open_time. A bar's
    close is not known until the bar closes; anchoring on open_time hands the
    model the next 60 seconds of price action for free.
  * Every other dataset is joined **as-of, backwards**: the most recent
    snapshot at or before the anchor, never the next one. Their staleness is
    recorded as a feature rather than hidden.
  * The `metrics` dataset is lagged a further five minutes on top of that,
    because its rows describe the window *after* their own timestamp. This was
    not documented anywhere; it was measured, and it is the single easiest way
    to build a spectacular backtest out of this data. See METRICS_WINDOW_MS.
  * Lookback windows are resolved by timestamp, not by index arithmetic, and a
    window with bars missing inside it drops the row instead of quietly
    measuring a different horizon than its name claims.
  * `vol_regime` is a percentile against strictly prior observations, so it
    cannot encode the future distribution.
  * Forward returns use the first bar closing at or after the target time, and
    a row whose forward window runs past the end of the data is dropped, not
    padded.

A note on bookDepth
-------------------
It is opt-in, and it is opt-in for a reason. Its rows are cumulative depth in
percentage bands either side of mid (-5, -4, -3, -2, -1, -0.2, +0.2, +1, +2,
+3, +4, +5), roughly every 30 seconds. On the day sampled while writing this
(2026-09-06), the implied average price of the positive-percentage side
(notional/depth) came out *below* the contemporaneous mid, which cannot be true
of resting asks above the mid. Either the side convention is not what it
appears, or those bands measure something else.

So `--with-depth` prints that diagnostic on every run and the features carry a
neutral name (`depth_imb_*` = negative-percentage side minus positive-
percentage side). Read the diagnostic before believing anything the depth
features tell you.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from bisect import bisect_left, insort
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.binance_import import read_zip_rows  # noqa: E402
from analysis.importer_core import (  # noqa: E402
    check_epoch_ms,
    ensure_clean_output,
    normalise_timestamp,
)

BASE = "https://data.binance.vision/data/futures/um"

# Earliest date each dataset exists for, from the bucket listing on 2026-09-07.
FIRST_DATE = {
    "klines": date(2019, 12, 31),
    "metrics": date(2020, 9, 1),
    "premiumIndexKlines": date(2019, 12, 24),
    "bookDepth": date(2023, 1, 1),
}

# Binance appends an "ignore" column to kline files; we name the first eleven.
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
]
METRICS_COLUMNS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]
DEPTH_COLUMNS = ["timestamp", "percentage", "depth", "notional"]

MINUTE_MS = 60_000
# A bar's close_time may sit a millisecond short of the round minute, and a
# venue may drop the odd bar. One bar of slack on any timestamp match.
MATCH_TOLERANCE_MS = 90_000

# Depth bands kept as features, by absolute percentage.
DEPTH_BANDS = (0.2, 1.0, 2.0, 5.0)

# A metrics row timestamped T describes the window [T, T+5min), NOT the window
# ending at T. Measured on 5,758 paired samples over 2026-07-01..2026-07-20:
# sum_taker_long_short_vol_ratio has a Spearman of +0.44 with the price move
# over the NEXT five minutes and +0.25 with the previous five. It is reporting
# flow that has not happened yet at time T.
#
# So an as-of join at T leaks the future, and it leaks it in the most
# convincing possible way: check_features flagged the resulting feature at
# IC +0.25 as "suspicious", which is exactly what a real edge would look like
# to someone who wanted one.
#
# A row is therefore only usable once its whole window has elapsed, so every
# metrics lookup is taken as-of `anchor - METRICS_WINDOW_MS`. The lag is
# applied to the entire dataset, not just the one field it was proved on: all
# eight columns share a create_time, and the semantics of the others cannot be
# verified independently.
METRICS_WINDOW_MS = 300_000


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


class Fetcher:
    """Cached, parallel, 404-tolerant downloader.

    Deliberately not `importer_core.download()`: that one narrates every file
    and exits the process on a 404, which is right when fetching the two files
    a tick import needs and wrong when fetching two thousand, where a missing
    day is a fact about the archive rather than an error.
    """

    def __init__(self, cache: Path, *, force: bool = False, workers: int = 8):
        self.cache = cache
        self.force = force
        self.workers = workers
        self.downloaded = 0
        self.cached = 0
        self.missing: List[str] = []
        self._done = 0
        self._total = 0

    def _one(self, url: str) -> Optional[Path]:
        destination = self.cache / url.rsplit("/", 1)[-1]
        if destination.exists() and destination.stat().st_size > 0 and not self.force:
            self.cached += 1
            self._progress()
            return destination

        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                partial.parent.mkdir(parents=True, exist_ok=True)
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
            partial.replace(destination)
            self.downloaded += 1
            self._progress()
            return destination
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            if exc.code == 404:
                self.missing.append(url.rsplit("/", 1)[-1])
                self._progress()
                return None
            raise
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    def _progress(self) -> None:
        with _print_lock:
            self._done += 1
            print(f"\r  {self._done:,} / {self._total:,} files", end="", flush=True)

    def fetch_all(self, urls: Sequence[str]) -> List[Path]:
        self._total = len(urls)
        self._done = 0
        self.cache.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            paths = list(pool.map(self._one, urls))
        print()
        return [path for path in paths if path is not None]


def daterange(start: date, end: date) -> List[date]:
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def kline_urls(symbol: str, days: Sequence[date], dataset: str, interval: str) -> List[str]:
    """Monthly archives where a whole calendar month is covered, else daily.

    Monthly files exist only for klines and premiumIndexKlines, and only for
    complete past months, so the edges of any range still need daily files.
    Using them where possible turns 2,442 requests into 80.
    """
    wanted = set(days)
    urls: List[str] = []
    covered: set = set()
    for year, month in sorted({(day.year, day.month) for day in days}):
        first = date(year, month, 1)
        following = date(year + (month == 12), month % 12 + 1, 1)
        month_days = daterange(first, following - timedelta(days=1))
        if all(day in wanted for day in month_days):
            urls.append(
                f"{BASE}/monthly/{dataset}/{symbol}/{interval}/"
                f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
            )
            covered.update(month_days)
    for day in days:
        if day not in covered:
            urls.append(
                f"{BASE}/daily/{dataset}/{symbol}/{interval}/"
                f"{symbol}-{interval}-{day.isoformat()}.zip"
            )
    return urls


def daily_urls(symbol: str, days: Sequence[date], dataset: str) -> List[str]:
    return [
        f"{BASE}/daily/{dataset}/{symbol}/{symbol}-{dataset}-{day.isoformat()}.zip"
        for day in days
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_datetime(value: str) -> int:
    """'2026-09-06 01:25:00' (UTC) -> epoch milliseconds."""
    stamp = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
    return int(stamp.replace(tzinfo=timezone.utc).timestamp() * 1000)


class Bars:
    """1-minute klines, sorted by close time. Close time is the anchor."""

    __slots__ = ("ts", "close", "high", "low", "volume", "count", "taker_buy")

    def __init__(self) -> None:
        self.ts: List[int] = []
        self.close: List[float] = []
        self.high: List[float] = []
        self.low: List[float] = []
        self.volume: List[float] = []
        self.count: List[float] = []
        self.taker_buy: List[float] = []

    def __len__(self) -> int:
        return len(self.ts)


def load_bars(paths: Sequence[Path]) -> Bars:
    rows: List[Tuple[int, float, float, float, float, float, float]] = []
    for path in paths:
        for row in read_zip_rows(path, KLINE_COLUMNS):
            try:
                ts = normalise_timestamp(int(float(row["close_time"])))
                rows.append((
                    ts,
                    float(row["close"]), float(row["high"]), float(row["low"]),
                    float(row["volume"]), float(row["count"]),
                    float(row["taker_buy_volume"]),
                ))
            except (TypeError, ValueError):
                continue

    rows.sort(key=lambda item: item[0])
    bars = Bars()
    previous = -1
    for ts, close, high, low, volume, count, taker in rows:
        if ts == previous:      # monthly and daily archives overlap at the edges
            continue
        previous = ts
        bars.ts.append(ts)
        bars.close.append(close)
        bars.high.append(high)
        bars.low.append(low)
        bars.volume.append(volume)
        bars.count.append(count)
        bars.taker_buy.append(taker)
    if bars.ts:
        check_epoch_ms(bars.ts[0], "klines close_time")
    return bars


def load_premium(paths: Sequence[Path]) -> Tuple[List[int], List[float]]:
    """Premium index closes -> (timestamps, premium as a fraction)."""
    rows: List[Tuple[int, float]] = []
    for path in paths:
        for row in read_zip_rows(path, KLINE_COLUMNS):
            try:
                rows.append((
                    normalise_timestamp(int(float(row["close_time"]))),
                    float(row["close"]),
                ))
            except (TypeError, ValueError):
                continue
    rows.sort(key=lambda item: item[0])
    ts = [item[0] for item in rows]
    values = [item[1] for item in rows]
    return ts, values


def load_metrics(paths: Sequence[Path]) -> Tuple[List[int], List[Dict[str, float]]]:
    rows: List[Tuple[int, Dict[str, float]]] = []
    for path in paths:
        for row in read_zip_rows(path, METRICS_COLUMNS):
            try:
                ts = parse_datetime(row["create_time"])
                rows.append((ts, {
                    "oi": float(row["sum_open_interest"]),
                    "toptrader_pos_ls": float(row["sum_toptrader_long_short_ratio"]),
                    "toptrader_acct_ls": float(row["count_toptrader_long_short_ratio"]),
                    "global_acct_ls": float(row["count_long_short_ratio"]),
                    "taker_ls_ratio": float(row["sum_taker_long_short_vol_ratio"]),
                }))
            except (TypeError, ValueError):
                continue
    rows.sort(key=lambda item: item[0])
    return [item[0] for item in rows], [item[1] for item in rows]


def load_depth(paths: Sequence[Path]) -> Tuple[List[int], List[Dict[str, float]]]:
    """Collapse each depth snapshot to per-band notional on each side."""
    snapshots: Dict[int, Dict[str, float]] = {}
    for path in paths:
        for row in read_zip_rows(path, DEPTH_COLUMNS):
            try:
                ts = parse_datetime(row["timestamp"])
                percentage = float(row["percentage"])
                notional = float(row["notional"])
                depth = float(row["depth"])
            except (TypeError, ValueError):
                continue
            band = snapshots.setdefault(ts, {})
            side = "neg" if percentage < 0 else "pos"
            band[f"{side}_{abs(percentage):g}"] = notional
            band[f"{side}_{abs(percentage):g}_qty"] = depth

    ts_sorted = sorted(snapshots)
    return ts_sorted, [snapshots[ts] for ts in ts_sorted]


# ---------------------------------------------------------------------------
# As-of joins and window resolution
# ---------------------------------------------------------------------------


def as_of(timestamps: Sequence[int], anchor_ts: int) -> Optional[int]:
    """Index of the most recent observation at or before `anchor_ts`.

    Backwards only. An as-of join that reaches forward by even one snapshot
    hands the model information it could not have had, and the resulting
    backtest is worthless in a way that is very hard to see.
    """
    position = bisect_left(timestamps, anchor_ts + 1) - 1
    return position if position >= 0 else None


def window_start(bars: Bars, index: int, minutes: int) -> Optional[int]:
    """Index of the bar `minutes` before `index`, or None if it is not there.

    Resolved by timestamp rather than `index - minutes`, so a gap in the
    archive drops the row instead of silently relabelling a 240-minute return
    as whatever span those 240 rows happen to cover.
    """
    target = bars.ts[index] - minutes * MINUTE_MS
    start = bisect_left(bars.ts, target)
    if start >= index:
        return None
    if abs(bars.ts[start] - target) > MATCH_TOLERANCE_MS:
        return None
    if index - start != minutes:        # bars missing inside the window
        return None
    return start


def forward_index(bars: Bars, index: int, horizon_s: float) -> Optional[int]:
    """First bar closing at or after `horizon_s` past this one."""
    target = bars.ts[index] + int(horizon_s * 1000)
    position = bisect_left(bars.ts, target)
    if position >= len(bars.ts):
        return None
    if bars.ts[position] - target > MATCH_TOLERANCE_MS:
        return None
    return position


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

RETURN_LOOKBACKS = (5, 15, 60, 240, 1440)
MA_LOOKBACKS = (60, 240, 1440)
RV_LOOKBACKS = (15, 60, 240)
TFI_LOOKBACKS = (5, 15, 60)
RANGE_LOOKBACKS = (240, 1440)
OI_LOOKBACKS = (15, 60, 240)
MAX_LOOKBACK_MINUTES = 1440

# How many strictly-prior observations the vol_regime percentile is measured
# against. Bounded for two reasons: an unbounded sorted list makes every
# insertion O(n) against a list the size of the whole dataset, and a regime
# label is more useful measured against the recent past than against 2019.
REGIME_WINDOW = 2000


def feature_columns(with_depth: bool) -> List[str]:
    columns = [
        "ts", "received_ts", "mid", "is_valid", "history_seconds", "vol_regime",
    ]
    columns += [f"ret_{n}m" for n in RETURN_LOOKBACKS]
    columns += [f"ma_dist_{n}m" for n in MA_LOOKBACKS]
    columns += [f"rv_{n}m" for n in RV_LOOKBACKS]
    columns += ["atr_60m"]
    columns += [f"range_pos_{n}m" for n in RANGE_LOOKBACKS]
    columns += [f"tfi_{n}m" for n in TFI_LOOKBACKS]
    columns += ["vol_ratio_15m", "count_ratio_15m"]
    columns += [f"oi_chg_{n}m" for n in OI_LOOKBACKS]
    columns += [
        "toptrader_pos_ls", "toptrader_acct_ls", "global_acct_ls",
        "taker_ls_ratio", "metrics_age_s",
        "premium_bps", "premium_age_s",
    ]
    if with_depth:
        columns += [f"depth_imb_{band:g}".replace(".", "p") for band in DEPTH_BANDS]
        columns += ["depth_total_ln", "depth_age_s"]
    return columns


def _log_bps(numerator: float, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return math.log(numerator / denominator) * 10_000.0


def bar_features(bars: Bars, index: int) -> Optional[Dict[str, float]]:
    """Kline-derived features, or None if any window is incomplete."""
    close = bars.close[index]
    if close <= 0:
        return None
    row: Dict[str, float] = {}

    starts: Dict[int, int] = {}
    for minutes in set(RETURN_LOOKBACKS + MA_LOOKBACKS + RV_LOOKBACKS
                       + TFI_LOOKBACKS + RANGE_LOOKBACKS + (15,)):
        start = window_start(bars, index, minutes)
        if start is None:
            return None
        starts[minutes] = start

    for minutes in RETURN_LOOKBACKS:
        row[f"ret_{minutes}m"] = _log_bps(close, bars.close[starts[minutes]])

    for minutes in MA_LOOKBACKS:
        window = bars.close[starts[minutes]:index + 1]
        average = sum(window) / len(window)
        row[f"ma_dist_{minutes}m"] = _log_bps(close, average)

    for minutes in RV_LOOKBACKS:
        window = bars.close[starts[minutes]:index + 1]
        steps = [
            math.log(window[i] / window[i - 1]) * 10_000.0
            for i in range(1, len(window))
            if window[i] > 0 and window[i - 1] > 0
        ]
        row[f"rv_{minutes}m"] = statistics.pstdev(steps) if len(steps) > 1 else 0.0

    start_60 = starts[60] if 60 in starts else window_start(bars, index, 60)
    if start_60 is None:
        return None
    true_ranges = [
        (bars.high[i] - bars.low[i]) / bars.close[i] * 10_000.0
        for i in range(start_60 + 1, index + 1)
        if bars.close[i] > 0
    ]
    row["atr_60m"] = sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    for minutes in RANGE_LOOKBACKS:
        window_high = max(bars.high[starts[minutes]:index + 1])
        window_low = min(bars.low[starts[minutes]:index + 1])
        span = window_high - window_low
        row[f"range_pos_{minutes}m"] = (
            (close - window_low) / span if span > 0 else 0.5
        )

    for minutes in TFI_LOOKBACKS:
        volume = sum(bars.volume[starts[minutes]:index + 1])
        buys = sum(bars.taker_buy[starts[minutes]:index + 1])
        row[f"tfi_{minutes}m"] = (
            (2.0 * buys - volume) / volume if volume > 0 else 0.0
        )

    recent = slice(starts[15], index + 1)
    baseline = slice(starts[1440], index + 1)
    row["vol_ratio_15m"] = _log_bps(
        sum(bars.volume[recent]) / 15.0,
        max(sum(bars.volume[baseline]) / 1440.0, 1e-12),
    ) / 10_000.0
    row["count_ratio_15m"] = _log_bps(
        sum(bars.count[recent]) / 15.0,
        max(sum(bars.count[baseline]) / 1440.0, 1e-12),
    ) / 10_000.0
    return row


def metrics_features(
    metric_ts: Sequence[int],
    metric_rows: Sequence[Dict[str, float]],
    anchor_ts: int,
) -> Optional[Dict[str, float]]:
    """Positioning features, lagged by one full metrics window.

    See METRICS_WINDOW_MS: a row timestamped T summarises the five minutes
    *after* T, so the newest row fully in the past at `anchor_ts` is the last
    one timestamped at or before `anchor_ts - METRICS_WINDOW_MS`.
    """
    observable_ts = anchor_ts - METRICS_WINDOW_MS
    position = as_of(metric_ts, observable_ts)
    if position is None:
        return None
    current = metric_rows[position]
    row: Dict[str, float] = {
        "toptrader_pos_ls": current["toptrader_pos_ls"],
        "toptrader_acct_ls": current["toptrader_acct_ls"],
        "global_acct_ls": current["global_acct_ls"],
        "taker_ls_ratio": current["taker_ls_ratio"],
        # Measured from the anchor, so it includes the deliberate lag: this is
        # genuinely how stale the newest usable observation is.
        "metrics_age_s": (anchor_ts - metric_ts[position]) / 1000.0,
    }
    for minutes in OI_LOOKBACKS:
        past = as_of(metric_ts, observable_ts - minutes * MINUTE_MS)
        if past is None:
            return None
        row[f"oi_chg_{minutes}m"] = _log_bps(current["oi"], metric_rows[past]["oi"])
    return row


def depth_features(
    depth_ts: Sequence[int],
    depth_rows: Sequence[Dict[str, float]],
    anchor_ts: int,
    baseline_ln: Optional[float],
) -> Optional[Dict[str, float]]:
    position = as_of(depth_ts, anchor_ts)
    if position is None:
        return None
    snapshot = depth_rows[position]
    row: Dict[str, float] = {"depth_age_s": (anchor_ts - depth_ts[position]) / 1000.0}
    total = 0.0
    for band in DEPTH_BANDS:
        negative = snapshot.get(f"neg_{band:g}")
        positive = snapshot.get(f"pos_{band:g}")
        if negative is None or positive is None:
            return None
        combined = negative + positive
        name = f"depth_imb_{band:g}".replace(".", "p")
        row[name] = (negative - positive) / combined if combined > 0 else 0.0
        total += combined
    if total <= 0:
        return None
    row["depth_total_ln"] = (
        math.log(total) - baseline_ln if baseline_ln is not None else 0.0
    )
    return row


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def horizon_tag(horizon: float) -> str:
    """Matches FeatureRecorder._tag, so the columns line up exactly."""
    return f"{horizon:g}s".replace(".", "p")


def build_rows(
    bars: Bars,
    metric_ts: Sequence[int],
    metric_rows: Sequence[Dict[str, float]],
    premium_ts: Sequence[int],
    premium_values: Sequence[float],
    depth_ts: Sequence[int],
    depth_rows: Sequence[Dict[str, float]],
    *,
    horizons: Tuple[float, ...],
    threshold_bps: float,
    sample_minutes: int,
    with_depth: bool,
    dropped: Dict[str, int],
) -> Iterator[Dict[str, object]]:
    """Yield finished rows in time order.

    A generator rather than a list: a multi-year run at one row a minute is
    millions of dicts, and there is no reason for more than one of them to
    exist at a time.
    """
    # Strictly-prior realized volatilities, for a leak-free regime percentile.
    seen_rv: List[float] = []
    recent_rv: deque = deque()
    depth_baseline: deque = deque(maxlen=60)

    for index in range(len(bars)):
        anchor_ts = bars.ts[index]
        if index % sample_minutes:
            continue
        if index < MAX_LOOKBACK_MINUTES:
            dropped["warmup"] += 1
            continue

        features = bar_features(bars, index)
        if features is None:
            dropped["bar_gap"] += 1
            continue

        metrics = metrics_features(metric_ts, metric_rows, anchor_ts)
        if metrics is None:
            dropped["metrics"] += 1
            continue
        features.update(metrics)

        premium_at = as_of(premium_ts, anchor_ts)
        if premium_at is None:
            dropped["premium"] += 1
            continue
        features["premium_bps"] = premium_values[premium_at] * 10_000.0
        features["premium_age_s"] = (anchor_ts - premium_ts[premium_at]) / 1000.0

        if with_depth:
            baseline = (
                statistics.median(depth_baseline[-60:]) if depth_baseline else None
            )
            depth = depth_features(depth_ts, depth_rows, anchor_ts, baseline)
            if depth is None:
                dropped["depth"] += 1
                continue
            features.update(depth)

        labels: Dict[str, object] = {}
        incomplete = False
        close = bars.close[index]
        for horizon in horizons:
            ahead = forward_index(bars, index, horizon)
            if ahead is None:
                incomplete = True
                break
            # Simple return, matching FeatureRecorder._label exactly, so a row
            # from here and a row from the live recorder mean the same thing.
            forward_bps = (bars.close[ahead] / close - 1.0) * 10_000.0
            tag = horizon_tag(horizon)
            labels[f"fwd_ret_bps_{tag}"] = round(forward_bps, 4)
            labels[f"label_{tag}"] = (
                1 if forward_bps >= threshold_bps
                else -1 if forward_bps <= -threshold_bps
                else 0
            )
        if incomplete:
            dropped["no_future"] += 1
            continue

        rv = features["rv_60m"]
        if len(seen_rv) < 200:
            regime = "unknown"
        else:
            percentile = bisect_left(seen_rv, rv) / len(seen_rv)
            regime = ("low" if percentile < 1 / 3
                      else "high" if percentile > 2 / 3 else "normal")
        insort(seen_rv, rv)
        recent_rv.append(rv)
        if len(recent_rv) > REGIME_WINDOW:
            oldest = recent_rv.popleft()
            position = bisect_left(seen_rv, oldest)
            if position < len(seen_rv) and seen_rv[position] == oldest:
                del seen_rv[position]

        if with_depth:
            snapshot_at = as_of(depth_ts, anchor_ts)
            if snapshot_at is not None:
                total = sum(
                    depth_rows[snapshot_at].get(f"neg_{band:g}", 0.0)
                    + depth_rows[snapshot_at].get(f"pos_{band:g}", 0.0)
                    for band in DEPTH_BANDS
                )
                if total > 0:
                    depth_baseline.append(math.log(total))

        row: Dict[str, object] = {
            "ts": anchor_ts,
            "received_ts": anchor_ts,
            "mid": close,
            "is_valid": True,
            "history_seconds": (anchor_ts - bars.ts[0]) / 1000.0,
            "vol_regime": regime,
        }
        row.update(features)
        row.update(labels)
        yield row


def write_csv(
    rows: Iterator[Dict[str, object]],
    out_dir: Path,
    name: str,
    horizons: Tuple[float, ...],
    with_depth: bool,
) -> Tuple[Path, int, Optional[int], Optional[int]]:
    """Write the stream to CSV. Returns (path, count, first ts, last ts)."""
    columns = feature_columns(with_depth)
    for horizon in horizons:
        tag = horizon_tag(horizon)
        columns += [f"fwd_ret_bps_{tag}", f"label_{tag}"]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    count = 0
    first_ts: Optional[int] = None
    last_ts: Optional[int] = None
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
            if first_ts is None:
                first_ts = int(row["ts"])
            last_ts = int(row["ts"])
    return path, count, first_ts, last_ts


def depth_diagnostic(
    depth_ts: Sequence[int],
    depth_rows: Sequence[Dict[str, float]],
    bars: Bars,
) -> None:
    """Report the implied average price of each side against the real price.

    notional/depth is the average price of the orders in a band. For bands
    below mid it must come out below the mid, and for bands above it, above.
    If the positive side prices below the mid, that side is not what its name
    suggests and no feature built on it should be trusted.
    """
    if not depth_ts or not bars.ts:
        return
    position = min(len(depth_ts) - 1, len(depth_ts) // 2)
    ts = depth_ts[position]
    snapshot = depth_rows[position]
    bar_at = as_of(bars.ts, ts)
    if bar_at is None:
        return
    reference = bars.close[bar_at]

    print("\n  bookDepth sanity check (implied average price per band):")
    print(f"    reference close      {reference:12,.2f}")
    for band in DEPTH_BANDS:
        for side, label in (("neg", "below mid"), ("pos", "above mid")):
            notional = snapshot.get(f"{side}_{band:g}")
            quantity = snapshot.get(f"{side}_{band:g}_qty")
            if not notional or not quantity:
                continue
            implied = notional / quantity
            offset = (implied / reference - 1.0) * 100.0
            flag = ""
            if side == "pos" and implied < reference:
                flag = "  <-- above-mid band prices BELOW mid"
            print(f"    {band:>4}% {label:9s}  {implied:12,.2f}  "
                  f"({offset:+.2f}%){flag}")
    print("    If any line is flagged, treat depth_imb_* as unverified.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_end = datetime.now(timezone.utc).date() - timedelta(days=1)
    default_start = default_end - timedelta(days=364)

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", type=parse_date, default=default_start,
                        help="YYYY-MM-DD (default: 365 days before --end).")
    parser.add_argument("--end", type=parse_date, default=default_end,
                        help="YYYY-MM-DD (default: yesterday UTC).")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=repo_root / "data" / "cache")
    parser.add_argument("--sample-minutes", type=int, default=5,
                        help="Emit one row every N minutes (default 5). Finer "
                             "sampling adds overlapping rows, not information.")
    parser.add_argument("--horizons", default="300,900,1800",
                        help="Forward label horizons in seconds.")
    parser.add_argument("--threshold-bps", type=float, default=10.0,
                        help="Move counted as up/down. Defaults to the taker "
                             "round-trip cost.")
    parser.add_argument("--with-depth", action="store_true",
                        help="Include bookDepth features. Read the module "
                             "docstring first: the side convention did not "
                             "validate against price.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true",
                        help="Re-download files already in the cache.")
    args = parser.parse_args(argv)

    if args.end < args.start:
        raise SystemExit("--end is before --start.")

    horizons = tuple(
        float(part) for part in args.horizons.split(",") if part.strip()
    )
    if not horizons:
        raise SystemExit("--horizons is empty.")

    earliest = max(
        FIRST_DATE["klines"], FIRST_DATE["metrics"], FIRST_DATE["premiumIndexKlines"],
        *([FIRST_DATE["bookDepth"]] if args.with_depth else []),
    )
    if args.start < earliest:
        print(f"--start {args.start} is before the archive begins; "
              f"clamping to {earliest}.")
        args.start = earliest

    out_dir = args.out or (
        repo_root / "data" / "bars"
        / f"{args.symbol}-{args.start.isoformat()}-{args.end.isoformat()}"
    )
    ensure_clean_output(out_dir)

    days = daterange(args.start, args.end)
    print(f"{args.symbol}  {args.start} .. {args.end}  ({len(days)} days)")

    cache = args.cache / args.symbol
    fetcher = Fetcher(cache, force=args.force, workers=args.workers)

    print("\nklines 1m")
    kline_paths = fetcher.fetch_all(kline_urls(args.symbol, days, "klines", "1m"))
    print("premiumIndexKlines 1h")
    premium_paths = fetcher.fetch_all(
        kline_urls(args.symbol, days, "premiumIndexKlines", "1h")
    )
    print("metrics")
    metric_paths = fetcher.fetch_all(daily_urls(args.symbol, days, "metrics"))
    depth_paths: List[Path] = []
    if args.with_depth:
        print("bookDepth")
        depth_paths = fetcher.fetch_all(daily_urls(args.symbol, days, "bookDepth"))

    if fetcher.missing:
        print(f"\n  {len(fetcher.missing)} file(s) not in the archive "
              f"(e.g. {fetcher.missing[0]}) - those spans are skipped.")

    print("\nParsing...")
    bars = load_bars(kline_paths)
    if len(bars) < MAX_LOOKBACK_MINUTES + 10:
        raise SystemExit(
            f"Only {len(bars)} bars parsed; need more than "
            f"{MAX_LOOKBACK_MINUTES} for the longest lookback."
        )
    premium_ts, premium_values = load_premium(premium_paths)
    metric_ts, metric_rows = load_metrics(metric_paths)
    depth_ts, depth_rows = load_depth(depth_paths) if args.with_depth else ([], [])
    print(f"  bars {len(bars):,}   metrics {len(metric_ts):,}   "
          f"premium {len(premium_ts):,}   depth {len(depth_ts):,}")

    if args.with_depth:
        depth_diagnostic(depth_ts, depth_rows, bars)

    print("\nBuilding features...")
    started = time.time()
    dropped = {"warmup": 0, "bar_gap": 0, "metrics": 0, "premium": 0,
               "depth": 0, "no_future": 0}
    stream = build_rows(
        bars, metric_ts, metric_rows, premium_ts, premium_values,
        depth_ts, depth_rows,
        horizons=horizons, threshold_bps=args.threshold_bps,
        sample_minutes=max(1, args.sample_minutes), with_depth=args.with_depth,
        dropped=dropped,
    )
    name = f"features-{args.start.isoformat()}_{args.end.isoformat()}.csv"
    path, written, first_ts, last_ts = write_csv(
        stream, out_dir, name, horizons, args.with_depth
    )
    if not written:
        path.unlink(missing_ok=True)
        raise SystemExit(f"No rows survived. Dropped: {dropped}")

    span_s = (last_ts - first_ts) / 1000.0
    print(f"\nWrote {written:,} rows in {time.time() - started:.1f}s")
    print(f"  span            {span_s / 86400:.1f} days")
    print(f"  dropped         {dropped}")
    print(f"  file            {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    for horizon in horizons:
        print(f"  independent observations at {horizon:g}s: "
              f"{span_s / horizon:,.0f}")
    print(f"\nRun the gate:\n  python backend\\analysis\\check_features.py "
          f"--data-dir {out_dir} --horizon {horizons[len(horizons) // 2]:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
