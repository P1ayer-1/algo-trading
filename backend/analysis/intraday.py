"""A `(period, symbol)` grid at any bar from 15 minutes up, funding aligned.

    from analysis.intraday import build_grid
    grid = build_grid(bar_minutes=480)        # one row per funding interval
    panel = grid.to_panel()                   # factor_panel.run_factor reads it unchanged

Why a grid rather than per-symbol arrays
----------------------------------------
Every cross-sectional question needs the same instant across symbols: the
cross-section's own mean return in a period (the market term every label here
subtracts), a rank of symbols at a settlement, a book held from one bar close
to another. A dense grid indexed by bar end makes "the same instant" an array
index rather than a timestamp join, and NaN where a symbol had no bar.

Alignment rules, the same two `panel_daily.py` was built around
---------------------------------------------------------------
A bar ending at `T` contains only 1m bars that closed at or before `T`.
Funding in the row ending `T` is what ACCRUED during `(T - bar, T]`: the
settlement nominally stamped `T` pays for the interval that just ended, so it
sits in the row that ends at `T` and never in an earlier one. Settlements
print a few tens of milliseconds late and are rounded to the nominal minute
first, exactly as the daily panel does - a raw `T + 21ms` would land in the
NEXT row, which hands every position a settlement it could not have known.

`complete` requires every minute of the bar to be present. A partial bar has a
close that is not a close, and a return across it spans more than one bar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from analysis.panel_intraday import OUT as INTRADAY_DIR, load_bars, panel_symbols
from analysis.factor_panel import Panel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "data" / "cache"
MINUTE_MS = 60_000


def _stamp(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).strftime("%Y-%m-%dT%H:%M")


def load_funding(symbol: str, cache: Path = CACHE) -> Dict[str, np.ndarray]:
    """Per-settlement funding in bps, stamped at the NOMINAL minute."""
    path = cache / symbol / "funding.json"
    if not path.exists():
        return {"ts": np.zeros(0, dtype=np.int64), "bps": np.zeros(0)}
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    ts = np.array(sorted(int(k) for k in raw), dtype=np.int64)
    bps = np.array([float(raw[str(k)]) for k in ts])
    nominal = ((ts + MINUTE_MS // 2) // MINUTE_MS) * MINUTE_MS
    return {"ts": nominal, "bps": bps}


@dataclass
class Grid:
    bar_ms: int
    ts_end: np.ndarray            # (periods,) int64, bar ends
    symbols: List[str]
    open: np.ndarray              # (periods, symbols), NaN where absent
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray            # quote volume in the bar
    taker: np.ndarray             # taker-buy share of quote volume
    trades: np.ndarray
    minutes: np.ndarray           # int16, 1m bars present
    rv: np.ndarray                # sqrt(sum sq 1m log returns) in bps
    funding: np.ndarray           # bps accrued in the bar, NaN where none settled
    funding_periods: np.ndarray   # settlements in the bar

    @property
    def shape(self):
        return self.close.shape

    @property
    def complete(self) -> np.ndarray:
        return self.minutes >= self.bar_ms // MINUTE_MS

    @property
    def dates(self) -> List[str]:
        return [_stamp(int(t)) for t in self.ts_end]

    def hours(self) -> np.ndarray:
        """UTC hour at which each bar ENDS."""
        return ((self.ts_end // 3_600_000) % 24).astype(int)

    def weekday(self) -> np.ndarray:
        """0 = Monday, of the bar's END instant."""
        return (((self.ts_end // 86_400_000) + 3) % 7).astype(int)

    def to_panel(self) -> Panel:
        """The shape `factor_panel.run_factor` expects. `hold_days` there means bars."""
        funding = np.where(self.funding_periods > 0, self.funding, np.nan)
        return Panel(self.dates, list(self.symbols), self.close, self.volume, funding,
                     self.rv, self.high, self.low, self.taker, self.complete)


def _resample(bars: Dict[str, np.ndarray], bar_ms: int) -> Dict[str, np.ndarray]:
    ts = bars["ts_end"]
    key = ((ts - 1) // bar_ms + 1) * bar_ms
    ends, start = np.unique(key, return_index=True)
    stop = np.append(start[1:], len(ts))
    quote = np.add.reduceat(bars["quote_volume"], start)
    taker_q = np.add.reduceat(bars["taker_buy_quote"], start)
    # Squared steps inside the coarse bar: the 15m bars' own inside-steps plus
    # the steps between consecutive 15m closes within the same coarse bar.
    close = bars["close"]
    between = np.zeros(len(close))
    between[1:] = np.log(close[1:] / close[:-1])
    between[start] = 0.0
    ss = np.add.reduceat(bars["ss_logret"] + between * between, start)
    with np.errstate(divide="ignore", invalid="ignore"):
        taker = np.where(quote > 0, taker_q / quote, np.nan)
    return {
        "ts_end": ends,
        "open": bars["open"][start],
        "high": np.maximum.reduceat(bars["high"], start),
        "low": np.minimum.reduceat(bars["low"], start),
        "close": close[stop - 1],
        "volume": quote,
        "taker": taker,
        "trades": np.add.reduceat(bars["trades"], start),
        "minutes": np.add.reduceat(bars["minutes"].astype(np.int64), start),
        "rv": np.sqrt(ss) * 10_000.0,
    }


ALL_FIELDS = ("open", "high", "low", "close", "volume", "taker", "trades", "rv", "funding")


def build_grid(bar_minutes: int = 480, symbols: Optional[Sequence[str]] = None,
               *, intraday_dir: Path = INTRADAY_DIR, cache: Path = CACHE,
               start_ms: Optional[int] = None, end_ms: Optional[int] = None,
               fields: Optional[Sequence[str]] = None) -> Grid:
    """Build the grid. `fields` limits which float arrays are filled.

    At 15 minutes a full grid is nine arrays of 175k x 108 float64, 1.4 GB,
    and two of them running beside each other on a 32 GB machine were
    measured paging (2026-09-12: 90 CPU-seconds in ten minutes each). An
    event study needs `close`, `volume` and `funding`; everything not asked
    for is left as an empty array rather than allocated and ignored.
    """
    if bar_minutes % 15:
        raise ValueError("bar_minutes must be a multiple of 15")
    wanted = set(ALL_FIELDS if fields is None else fields) | {"close"}
    unknown = wanted - set(ALL_FIELDS)
    if unknown:
        raise ValueError("unknown fields: " + ", ".join(sorted(unknown)))
    bar_ms = bar_minutes * MINUTE_MS
    if symbols is None:
        symbols = [s for s in panel_symbols() if (intraday_dir / (s + ".npz")).exists()]
    symbols = list(symbols)

    per_symbol = {}
    lo, hi = None, None
    for symbol in symbols:
        bars = load_bars(symbol, intraday_dir)
        res = _resample(bars, bar_ms)
        per_symbol[symbol] = res
        lo = res["ts_end"][0] if lo is None else min(lo, res["ts_end"][0])
        hi = res["ts_end"][-1] if hi is None else max(hi, res["ts_end"][-1])
    if start_ms is not None:
        lo = max(lo, ((start_ms - 1) // bar_ms + 1) * bar_ms)
    if end_ms is not None:
        hi = min(hi, (end_ms // bar_ms) * bar_ms)
    ts_end = np.arange(lo, hi + bar_ms, bar_ms, dtype=np.int64)
    n_t, n_s = len(ts_end), len(symbols)

    def blank(dtype=float):
        if dtype is float:
            return np.full((n_t, n_s), np.nan)
        return np.zeros((n_t, n_s), dtype=dtype)

    fields = {k: (blank() if k in wanted else np.zeros((0, 0))) for k in ALL_FIELDS}
    minutes = blank(np.int16)
    periods = blank(np.int16)

    for s, symbol in enumerate(symbols):
        res = per_symbol[symbol]
        idx = (res["ts_end"] - lo) // bar_ms
        ok = (idx >= 0) & (idx < n_t)
        idx = idx[ok]
        for k in ("open", "high", "low", "close", "volume", "taker", "trades", "rv"):
            if k in wanted:
                fields[k][idx, s] = res[k][ok]
        minutes[idx, s] = res["minutes"][ok]

        fund = load_funding(symbol, cache) if "funding" in wanted else {"ts": np.zeros(0)}
        if len(fund["ts"]):
            fkey = ((fund["ts"] - 1) // bar_ms + 1) * bar_ms
            fidx = (fkey - lo) // bar_ms
            fok = (fidx >= 0) & (fidx < n_t)
            col_n = np.zeros(n_t, dtype=np.int64)
            np.add.at(col_n, fidx[fok], 1)
            col = np.zeros(n_t)
            np.add.at(col, fidx[fok], fund["bps"][fok])
            periods[:, s] = col_n
            fields["funding"][:, s] = np.where(col_n > 0, col, np.nan)

    return Grid(bar_ms, ts_end, symbols, fields["open"], fields["high"], fields["low"],
                fields["close"], fields["volume"], fields["taker"], fields["trades"],
                minutes, fields["rv"], fields["funding"], periods)


def log_returns(close: np.ndarray, lag: int = 1) -> np.ndarray:
    """`(periods, symbols)` log return over `lag` bars ending at each row, in bps."""
    out = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close[lag:] / close[:-lag]
        out[lag:] = np.log(np.where(ratio > 0, ratio, np.nan)) * 10_000.0
    return out


def forward_returns(close: np.ndarray, lag: int = 1) -> np.ndarray:
    """Return over the NEXT `lag` bars, in bps, stamped at the entry row."""
    out = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close[lag:] / close[:-lag]
        out[:-lag] = np.log(np.where(ratio > 0, ratio, np.nan)) * 10_000.0
    return out


def excess(returns: np.ndarray, eligible: Optional[np.ndarray] = None) -> np.ndarray:
    """Subtract each row's cross-sectional mean (over eligible names)."""
    values = returns if eligible is None else np.where(eligible, returns, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=1, keepdims=True)
    return returns - mean


def block_bootstrap_mean(values: np.ndarray, *, block: int, draws: int = 2000,
                         seed: int = 0):
    """95% interval on the mean of a series, resampling whole blocks."""
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    offsets = np.arange(block)
    means = np.empty(draws)
    for i in range(draws):
        idx = (starts[i][:, None] + offsets[None, :]).ravel()[:n]
        means[i] = values[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
