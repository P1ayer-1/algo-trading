"""Fade the edges of a trading range, with stops. Would it have made money?

    python backend\\analysis\\range_backtest.py
    python backend\\analysis\\range_backtest.py --symbols BTCUSDT,ETHUSDT --days 180
    python backend\\analysis\\range_backtest.py --lookback-hours 24 --entry-fracs 0.1 --stop-fracs 0.25 --trend-filters 0.5

The strategy (`trading/strategies/range_trade/levels.py`): take the high and
low of the last N hours, rest a buy near the bottom and a sell near the top,
put a stop beyond each edge for when the range breaks, take profit at the
middle. Meant for a market going sideways.

Why the obvious backtest of this lies
-------------------------------------
**Win rate is set by geometry, not by the market.** On a driftless random walk
a bracket with its stop S away and its target T away wins S/(S+T) of the time.
Rest the entry a quarter of the way in, stop half a width out, target the
middle, and it wins 75% of trades on pure noise - and still loses money,
because for a martingale the expected price P&L of ANY stop/target bracket is
exactly zero, the fees come off the top, and the rare losses are three times
the size of the wins. A high win rate from a range fade is not evidence of
anything. This report does not lead with it.

So the question is not "does it win" but "does it beat a random walk with the
same volatility", and there is a control built for exactly that - below.

Execution on one-minute bars, bracketed
---------------------------------------
Bars say what the high and low were, not in which order. Two things decide a
result and neither is knowable from OHLC, so each is BRACKETED - the way
`passive_sim.py` brackets queue position - rather than guessed:

                                    pessimistic              optimistic
    resting limit fills             trades --through-bps     touched
                                    THROUGH the price
    stop and target in one bar      stop first               target first
    target in the entry's own bar   ignored: it may have     counted
                                    printed before the fill

What is not ambiguous is not bracketed. A bar that fills a resting buy and
whose low reaches the stop was stopped out under both: price had to pass the
entry on its way down. A bar that OPENS beyond a stop fills it at the open,
not at the stop - a gap is where a stop costs more than its distance.

Orders only ever REST. If the rolling range moves its entry level past the
price, a limit there would cross the book and pay taker; the planner refuses
that, so the backtest does not take it either.

Fees are BloFin's for the configured VIP tier: entries and targets pay maker,
stops and time exits pay taker plus `--slippage-bps`.

What is historical and what is not
----------------------------------
**Price path: Binance USDT-M 1m klines** from the free archive. BloFin serves
no deep 1m history, and for majors the two venues differ by a basis of a few
bps, not in shape. **Spreads** enter only as `--slippage-bps` on taker exits;
BloFin's measured spreads (README step 9c) run to 9 bps on ADA, so thin names
are flattered. **Funding is not modelled**: holds are capped at one lookback,
and majors pay about +/-1 bps per 8h. **Liquidation is gated**: a range whose
stop sits beyond the liquidation price at `--leverage` is refused, as a
planner would refuse it, and a stop that gaps through liquidation loses the
whole isolated margin.

The control
-----------
Each UTC day's bars are shuffled among themselves - return, high, low and open
offsets moved together - and the identical strategy runs on the result. A
day's net move survives exactly (a sum does not care about order), and so do
its volatility and the multi-day trend. What is destroyed is the ORDER of
moves inside the day: any tendency for a move to be followed by its reversal.
That tendency is the only thing a range fade can be harvesting, so
`real - control` is its edge. The control's own net should sit near minus its
costs; if it makes money, the simulator is broken, not the market kind.

Selection is out of sample
--------------------------
A small grid runs and ALL of it is printed. One configuration is chosen on the
first 70% of the history - the median across symbols, trades crossing the
split purged - and scored on the last 30%. The last 21 days, the sideways
stretch that prompted this, are reported separately; they sit inside the
out-of-sample part, so nothing about them chose the configuration.

Neither trades nor symbols are independent: a day's trades share its regime,
and the majors move together. Confidence intervals resample whole calendar
days across every symbol at once, which keeps that correlation inside each
block instead of counting it as agreement.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bars_import import KLINE_COLUMNS, Fetcher, daterange, kline_urls  # noqa: E402
from analysis.importer_core import SchemaError, check_epoch_ms  # noqa: E402
from analysis.stats import spearman  # noqa: E402
from trading.risk import Side, liquidation_price  # noqa: E402
from trading.strategies.range_trade.levels import RangeParams  # noqa: E402

MINUTE_MS = 60_000
DAY_MS = 86_400_000
BPS = 10_000.0

DEFAULT_SYMBOLS = ("BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,"
                   "DOGEUSDT,ADAUSDT,LINKUSDT,LTCUSDT,AVAXUSDT")

REASONS = ("target", "stop", "time", "liquidated", "open")
TARGET, STOP, TIME, LIQUIDATED, OPEN = range(len(REASONS))

# A symbol needs this many trades in a segment for its mean to be counted.
MIN_TRADES = 10

# Measured against a real position by analysis/validate_liquidation.py - the
# same constants plan_carry.py uses.
MEASURED_MMR = Decimal("0.005")
MEASURED_FEE_BUFFER_BPS = Decimal("6")

_NEEDED = ("close_time", "open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Ohlc:
    """1-minute bars, sorted, stamped with their CLOSE time in ms."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def __len__(self) -> int:
        return len(self.ts)

    @classmethod
    def empty(cls) -> "Ohlc":
        nothing = np.empty(0)
        return cls(np.empty(0, dtype=np.int64), nothing, nothing, nothing, nothing)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def read_klines(path: Path) -> np.ndarray:
    """(rows, 5) array of close_time, open, high, low, close from one zip.

    `np.loadtxt` rather than the csv reader `bars_import` uses: a year of one
    symbol is 527,000 rows, and a run loads ten symbols. The header, when the
    file has one, is still checked against the column positions this reads by,
    because a silently re-ordered high and low produce a perfectly plausible
    backtest of nothing.
    """
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise SchemaError(f"Expected exactly one CSV in {path.name}, found {names}")
        text = archive.read(names[0]).decode("utf-8").lstrip("﻿")

    head, _, rest = text.partition("\n")
    fields = [part.strip().lower() for part in head.split(",")]
    body = text
    if fields and fields[0] and not _is_number(fields[0]):
        found = {name: (fields.index(name) if name in fields else -1)
                 for name in _NEEDED}
        expected = {name: KLINE_COLUMNS.index(name) for name in _NEEDED}
        if found != expected:
            raise SchemaError(
                f"{path.name}: header {fields} does not put {list(_NEEDED)} at "
                f"{expected}. Binance may have changed the format; update "
                "KLINE_COLUMNS in bars_import.py.")
        body = rest
    if not body.strip():
        return np.empty((0, 5))
    columns = tuple(KLINE_COLUMNS.index(name) for name in _NEEDED)
    return np.loadtxt(io.StringIO(body), delimiter=",", usecols=columns, ndmin=2)


def _to_ms(stamps: np.ndarray) -> np.ndarray:
    """Some Binance archives moved to microsecond stamps. Normalise to ms."""
    out = stamps.astype(np.int64)
    micro = out >= 10**14
    out[micro] //= 1000
    return out


def load_ohlc(paths: Sequence[Path]) -> Ohlc:
    blocks = [block for block in (read_klines(path) for path in paths) if len(block)]
    if not blocks:
        return Ohlc.empty()
    data = np.vstack(blocks)
    # np.unique sorts, and drops the rows monthly and daily archives share.
    ts, first = np.unique(_to_ms(data[:, 0]), return_index=True)
    data = data[first]
    check_epoch_ms(int(ts[0]), "klines close_time")
    return Ohlc(ts=ts, open=data[:, 1].copy(), high=data[:, 2].copy(),
                low=data[:, 3].copy(), close=data[:, 4].copy())


# ---------------------------------------------------------------------------
# The rolling range - must agree with levels.find_range
# ---------------------------------------------------------------------------


def trailing_extreme(values: np.ndarray, window: int, reducer) -> np.ndarray:
    """out[i] = reducer over values[i-window:i] - the PRIOR window, never bar i.

    NaN for the first `window` bars. O(n log window): extremes over power-of-
    two spans are built by doubling, and any window is two overlapping spans.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan)
    if window < 1 or n <= window:
        return out
    table = values
    span = 1
    while span * 2 <= window:
        table = reducer(table[:-span], table[span:])
        span *= 2
    count = n - window + 1
    blocks = reducer(table[:count], table[window - span:window - span + count])
    out[window:] = blocks[:n - window]
    return out


def trailing_max(values: np.ndarray, window: int) -> np.ndarray:
    return trailing_extreme(values, window, np.maximum)


def trailing_min(values: np.ndarray, window: int) -> np.ndarray:
    return trailing_extreme(values, window, np.minimum)


def contiguous(ts: np.ndarray, window: int) -> np.ndarray:
    """True at i when bars i-window .. i are consecutive minutes.

    A window spanning a hole in the archive is measuring a different period
    than its name, and a range across it is not the range anyone saw.
    """
    n = len(ts)
    out = np.zeros(n, dtype=bool)
    if n <= window:
        return out
    breaks = np.concatenate(([0], np.cumsum(np.diff(ts) != MINUTE_MS)))
    out[window:] = breaks[window:] == breaks[:n - window]
    return out


@dataclass
class RollingRange:
    """`find_range` evaluated at every bar, over the bars before it."""

    lookback: int
    high: np.ndarray
    low: np.ndarray
    width_bps: np.ndarray
    trend_ratio: np.ndarray
    valid: np.ndarray

    @classmethod
    def build(cls, ohlc: Ohlc, lookback: int) -> "RollingRange":
        n = len(ohlc)
        high = trailing_max(ohlc.high, lookback)
        low = trailing_min(ohlc.low, lookback)
        valid = contiguous(ohlc.ts, lookback) & np.isfinite(high) & np.isfinite(low)

        first = np.full(n, np.nan)
        last = np.full(n, np.nan)
        if n > lookback:
            first[lookback:] = ohlc.close[:n - lookback]
        last[1:] = ohlc.close[:-1]

        width = high - low
        mid = (high + low) / 2
        with np.errstate(invalid="ignore"):
            width_bps = np.divide(width, mid, out=np.zeros(n), where=mid > 0) * BPS
            trend = np.divide(np.abs(last - first), width,
                              out=np.full(n, np.inf), where=width > 0)
        return cls(lookback, high, low, width_bps, trend, valid)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Costs:
    """Per side, bps of notional. `through_bps` binds only the pessimistic
    bracket: a resting limit counts as filled once price trades that far past
    it, which stands in for a tick of queue."""

    maker_bps: float
    taker_bps: float
    slippage_bps: float = 3.0
    through_bps: float = 1.0


def liquidation_ratios(leverage: float, *, mmr: Decimal = MEASURED_MMR,
                       fee_buffer_bps: Decimal = MEASURED_FEE_BUFFER_BPS
                       ) -> Tuple[float, float]:
    """(long, short) liquidation price as a multiple of entry.

    `risk.liquidation_price` is linear in entry when there is no extra margin,
    so one call per side at an entry of 1 prices every trade - and the
    validated formula stays the only one.
    """
    lev = Decimal(str(leverage))
    long = liquidation_price(entry_price=Decimal(1), leverage=lev, side=Side.LONG,
                             maintenance_margin_rate=mmr,
                             fee_buffer_bps=fee_buffer_bps)
    short = liquidation_price(entry_price=Decimal(1), leverage=lev, side=Side.SHORT,
                              maintenance_margin_rate=mmr,
                              fee_buffer_bps=fee_buffer_bps)
    return (float(long) if long is not None else 0.0,
            float(short) if short is not None else math.inf)


TRADE_FIELDS = ("entry_index", "exit_index", "entry_ts", "exit_ts", "side",
                "entry_price", "exit_price", "reason", "net_bps", "fees_bps")
_DTYPES = (np.int64, np.int64, np.int64, np.int64, np.int8,
           np.float64, np.float64, np.int8, np.float64, np.float64)


@dataclass
class Trades:
    entry_index: np.ndarray
    exit_index: np.ndarray
    entry_ts: np.ndarray
    exit_ts: np.ndarray
    side: np.ndarray
    entry_price: np.ndarray
    exit_price: np.ndarray
    reason: np.ndarray
    net_bps: np.ndarray
    fees_bps: np.ndarray
    # Whole-run counters. Subsets do not carry them.
    ambiguous: int = 0
    refused_liquidation: int = 0

    @classmethod
    def from_rows(cls, rows: Sequence[tuple], **counters) -> "Trades":
        columns = list(zip(*rows)) if rows else [[] for _ in TRADE_FIELDS]
        return cls(*(np.asarray(column, dtype=dtype)
                     for column, dtype in zip(columns, _DTYPES)), **counters)

    @classmethod
    def concat(cls, parts: Sequence["Trades"]) -> "Trades":
        parts = list(parts)
        if not parts:
            return cls.from_rows([])
        return cls(*(np.concatenate([getattr(part, name) for part in parts])
                     for name in TRADE_FIELDS))

    def __len__(self) -> int:
        return len(self.net_bps)

    def between(self, start_ts: float, end_ts: float) -> "Trades":
        """Entered at or after `start_ts` AND closed before `end_ts`.

        Purged, not merely filtered on entry: a trade opened before a split and
        closed after it takes its outcome from the far side, which is the
        leak a purged split exists to stop.
        """
        mask = (self.entry_ts >= start_ts) & (self.exit_ts < end_ts)
        return Trades(*(getattr(self, name)[mask] for name in TRADE_FIELDS))


def simulate(ohlc: Ohlc, rolling: RollingRange, params: RangeParams, costs: Costs,
             *, optimistic: bool, leverage: float,
             start: int = 0, end: Optional[int] = None) -> Trades:
    """Run one configuration over one series. One position at a time.

    Entries are considered on bars [start, end); exits may run past `end`.
    Levels freeze at entry - a stop that follows a moving range is a
    different strategy.
    """
    n = len(ohlc)
    end = n if end is None else min(end, n)
    hold = params.max_hold_minutes
    through = 0.0 if optimistic else costs.through_bps / BPS
    slip = costs.slippage_bps / BPS
    liq_long, liq_short = liquidation_ratios(leverage)
    margin_bps = BPS / leverage

    low, high = rolling.low, rolling.high
    width = high - low
    with np.errstate(invalid="ignore"):
        ok = rolling.valid & (width > 0) & (rolling.width_bps >= params.min_width_bps)
        if params.max_trend_ratio is not None:
            ok &= rolling.trend_ratio <= params.max_trend_ratio
        long_entry = low + params.entry_frac * width
        short_entry = high - params.entry_frac * width
        # Resting only: the bar must OPEN on the far side of the level, or an
        # order there would have crossed the book.
        long_hit = ok & (ohlc.open > long_entry) & (ohlc.low <= long_entry * (1 - through))
        short_hit = ok & (ohlc.open < short_entry) & (ohlc.high >= short_entry * (1 + through))
    candidates = np.flatnonzero(long_hit | short_hit)

    O, H, L, C, ts = ohlc.open, ohlc.high, ohlc.low, ohlc.close, ohlc.ts
    rows: List[tuple] = []
    ambiguous = refused = 0
    i = max(0, start)
    while True:
        at = int(np.searchsorted(candidates, i))
        if at >= len(candidates):
            break
        k = int(candidates[at])
        if k >= end:
            break
        if long_hit[k] and short_hit[k]:
            # One bar crossed both entries. Which filled first decides the
            # trade, and nothing in the bar says.
            ambiguous += 1
            i = k + 1
            continue

        side = 1 if long_hit[k] else -1
        if side > 0:
            fill = float(long_entry[k])
            stop = float(low[k] - params.stop_frac * width[k])
            target = float(low[k] + params.target_frac * width[k])
            liq = fill * liq_long
            beyond_liquidation = stop <= liq
        else:
            fill = float(short_entry[k])
            stop = float(high[k] + params.stop_frac * width[k])
            target = float(high[k] - params.target_frac * width[k])
            liq = fill * liq_short
            beyond_liquidation = stop >= liq
        if beyond_liquidation:
            refused += 1
            i = k + 1
            continue

        last = min(k + hold, n - 1)
        window = slice(k, last + 1)
        if side > 0:
            stop_hit = L[window] <= stop
            gap_stop = O[window] <= stop
            gap_target = O[window] >= target
            target_hit = (H[window] >= target * (1 + through)) | gap_target
            touched_in_fill_bar = bool(H[k] >= target)
        else:
            stop_hit = H[window] >= stop
            gap_stop = O[window] >= stop
            gap_target = O[window] <= target
            target_hit = (L[window] <= target * (1 - through)) | gap_target
            touched_in_fill_bar = bool(L[k] <= target)
        # The fill bar. Its open came BEFORE the fill, so it says nothing about
        # exits. Its stop is certain - price passed the entry on its way to the
        # stop. Its target is not: that print may have preceded the fill.
        gap_stop[0] = False
        gap_target[0] = False
        target_hit[0] = optimistic and touched_in_fill_bar

        hits = stop_hit | target_hit
        if hits.any():
            j = int(np.argmax(hits))
            if gap_stop[j]:
                reason, level = STOP, float(O[k + j])
            elif gap_target[j]:
                reason, level = TARGET, float(O[k + j])
            elif target_hit[j] and (optimistic or not stop_hit[j]):
                reason, level = TARGET, target
            else:
                reason, level = STOP, stop
            exit_index = k + j
        else:
            exit_index = last
            reason = TIME if k + hold <= n - 1 else OPEN
            level = float(C[last])

        entry_fee = costs.maker_bps
        if reason == TARGET:
            exit_price, exit_fee = level, costs.maker_bps
        else:
            exit_price = level * (1 - slip) if side > 0 else level * (1 + slip)
            exit_fee = costs.taker_bps
        fees = entry_fee + exit_fee
        net = side * (exit_price - fill) / fill * BPS - fees
        if reason == STOP and (exit_price <= liq if side > 0 else exit_price >= liq):
            # The stop filled past liquidation, so the margin engine got there
            # first. Isolated: the posted margin is the loss.
            reason, exit_price, fees = LIQUIDATED, liq, entry_fee
            net = -margin_bps - entry_fee

        rows.append((k, exit_index, int(ts[k]), int(ts[exit_index]), side,
                     fill, exit_price, reason, net, fees))
        i = exit_index + 1

    return Trades.from_rows(rows, ambiguous=ambiguous, refused_liquidation=refused)


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------


def shuffle_within_days(ohlc: Ohlc, rng: np.random.Generator) -> Ohlc:
    """The same bars, re-ordered inside each UTC day.

    Every bar keeps its return, and its high, low and open relative to the
    close before it, so the day's moves, volatility and net change all
    survive. Only the sequence inside the day is destroyed.
    """
    n = len(ohlc)
    if n == 0:
        return Ohlc.empty()
    prev = np.empty(n)
    prev[0] = ohlc.open[0]
    prev[1:] = ohlc.close[:-1]
    ret = np.log(ohlc.close / prev)
    up = np.log(ohlc.high / prev)
    down = np.log(ohlc.low / prev)
    gap = np.log(ohlc.open / prev)

    day = ohlc.ts // DAY_MS
    edges = np.concatenate(([0], np.flatnonzero(np.diff(day)) + 1, [n]))
    order = np.arange(n)
    for a, b in zip(edges[:-1], edges[1:]):
        order[a:b] = a + rng.permutation(b - a)

    close = prev[0] * np.exp(np.cumsum(ret[order]))
    new_prev = np.empty(n)
    new_prev[0] = prev[0]
    new_prev[1:] = close[:-1]
    return Ohlc(ts=ohlc.ts.copy(), open=new_prev * np.exp(gap[order]),
                high=new_prev * np.exp(up[order]),
                low=new_prev * np.exp(down[order]), close=close)


# ---------------------------------------------------------------------------
# Is it actually sideways?
# ---------------------------------------------------------------------------


def daily_bars(ohlc: Ohlc) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(high, low, close) per UTC day."""
    day = ohlc.ts // DAY_MS
    starts = np.concatenate(([0], np.flatnonzero(np.diff(day)) + 1))
    ends = np.concatenate((starts[1:], [len(ohlc)])) - 1
    return (np.maximum.reduceat(ohlc.high, starts),
            np.minimum.reduceat(ohlc.low, starts), ohlc.close[ends])


def sideways_now(ohlc: Ohlc, days: int) -> Optional[Dict[str, float]]:
    """The last `days` days, against every `days`-day window in the series.

    Percentiles are the share of windows at or below the latest, the latest
    included. A LOW trend percentile is more sideways than usual.
    """
    if len(ohlc) == 0:
        return None
    high, low, close = daily_bars(ohlc)
    if len(close) < days + 1:
        return None
    trends: List[float] = []
    widths: List[float] = []
    for end in range(days, len(close)):
        hi = float(high[end - days + 1:end + 1].max())
        lo = float(low[end - days + 1:end + 1].min())
        width = hi - lo
        trends.append(abs(close[end] - close[end - days]) / width if width > 0 else math.inf)
        widths.append(width / ((hi + lo) / 2))
    trend = np.array(trends)
    width = np.array(widths)
    return {
        "move": float(close[-1] / close[-1 - days] - 1),
        "width": float(width[-1]),
        "trend": float(trend[-1]),
        "trend_pctile": float(np.mean(trend <= trend[-1])),
        "width_pctile": float(np.mean(width <= width[-1])),
        "windows": float(len(trend)),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def day_blocks(trades: Trades) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(day, net sum, trade count) per calendar day of entry."""
    if not len(trades):
        empty = np.empty(0)
        return empty.astype(np.int64), empty, empty
    days, inverse = np.unique(trades.entry_ts // DAY_MS, return_inverse=True)
    sums = np.bincount(inverse, weights=trades.net_bps, minlength=len(days))
    counts = np.bincount(inverse, minlength=len(days)).astype(np.float64)
    return days, sums, counts


def bootstrap_means(sums: np.ndarray, counts: np.ndarray, rng: np.random.Generator,
                    draws: int) -> Optional[np.ndarray]:
    """Mean per TRADE, resampling whole days. None with fewer than two days."""
    if len(sums) < 2:
        return None
    picks = rng.integers(0, len(sums), size=(draws, len(sums)))
    return sums[picks].sum(axis=1) / np.maximum(counts[picks].sum(axis=1), 1.0)


@dataclass
class Summary:
    trades: int = 0
    days: int = 0
    mean_bps: float = math.nan
    low_bps: float = math.nan
    high_bps: float = math.nan
    win_rate: float = math.nan
    avg_win_bps: float = math.nan
    avg_loss_bps: float = math.nan
    shares: Tuple[float, ...] = field(default_factory=tuple)
    taker_exit_share: float = math.nan
    fees_bps: float = math.nan
    liquidated: int = 0


def summarise(trades: Trades, rng: np.random.Generator, draws: int = 2000) -> Summary:
    summary = Summary(trades=len(trades))
    if not len(trades):
        return summary
    _, sums, counts = day_blocks(trades)
    net = trades.net_bps
    summary.days = len(sums)
    summary.mean_bps = float(net.mean())
    means = bootstrap_means(sums, counts, rng, draws)
    if means is not None:
        summary.low_bps, summary.high_bps = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    wins, losses = net[net > 0], net[net <= 0]
    summary.win_rate = float(len(wins) / len(net))
    summary.avg_win_bps = float(wins.mean()) if len(wins) else math.nan
    summary.avg_loss_bps = float(losses.mean()) if len(losses) else math.nan
    summary.shares = tuple(float(np.mean(trades.reason == code)) for code in range(len(REASONS)))
    # Slippage is charged on these exits only, so this is also the slope of
    # the mean in bps per extra bp of slippage.
    summary.taker_exit_share = float(np.mean(np.isin(trades.reason, (STOP, TIME, OPEN))))
    summary.fees_bps = float(trades.fees_bps.mean())
    summary.liquidated = int(np.sum(trades.reason == LIQUIDATED))
    return summary


def paired_difference(a: Trades, b: Trades, rng: np.random.Generator,
                      draws: int = 2000) -> Tuple[float, float, float]:
    """mean(a) - mean(b), with an interval from resampling the SAME days in both.

    The control is built from the same days as the real series, so pairing by
    day removes the day's own volatility from the comparison.
    """
    if not len(a) or not len(b):
        return math.nan, math.nan, math.nan
    point = float(a.net_bps.mean() - b.net_bps.mean())
    days_a, sums_a, counts_a = day_blocks(a)
    days_b, sums_b, counts_b = day_blocks(b)
    days = np.union1d(days_a, days_b)
    if len(days) < 2:
        return point, math.nan, math.nan
    sa, ca, sb, cb = (np.zeros(len(days)) for _ in range(4))
    sa[np.searchsorted(days, days_a)], ca[np.searchsorted(days, days_a)] = sums_a, counts_a
    sb[np.searchsorted(days, days_b)], cb[np.searchsorted(days, days_b)] = sums_b, counts_b
    picks = rng.integers(0, len(days), size=(draws, len(days)))
    na, nb = ca[picks].sum(axis=1), cb[picks].sum(axis=1)
    usable = (na > 0) & (nb > 0)
    diff = sa[picks].sum(axis=1)[usable] / na[usable] - sb[picks].sum(axis=1)[usable] / nb[usable]
    if not len(diff):
        return point, math.nan, math.nan
    low, high = np.percentile(diff, [2.5, 97.5])
    return point, float(low), float(high)


def equity_path(net_bps: np.ndarray, exposure: float) -> Tuple[float, float]:
    """(final equity multiple, max drawdown) trading `exposure` x equity each time.

    A trade losing more than the whole stake ends the account at zero - which
    is what a liquidation at full allocation is.
    """
    if not len(net_bps):
        return 1.0, 0.0
    returns = np.maximum(exposure * np.asarray(net_bps) / BPS, -1.0)
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    return float(equity[-1]), float(np.max(1.0 - equity / peaks))


# ---------------------------------------------------------------------------
# One symbol, in a worker process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    symbol: str
    paths: Tuple[Path, ...]
    grid: Tuple[RangeParams, ...]
    costs: Costs
    leverage: float
    seed: int
    recent_days: int


@dataclass
class SymbolResult:
    symbol: str
    bars: int
    first_ts: int
    last_ts: int
    sideways: Optional[Dict[str, float]]
    # (grid index, "real" | "control", optimistic) -> trades
    runs: Dict[Tuple[int, str, bool], Trades]


def run_symbol(job: Job) -> SymbolResult:
    ohlc = load_ohlc(job.paths)
    if len(ohlc) == 0:
        return SymbolResult(job.symbol, 0, 0, 0, None, {})
    control = shuffle_within_days(ohlc, np.random.default_rng(job.seed))
    runs: Dict[Tuple[int, str, bool], Trades] = {}
    lookbacks = sorted({params.lookback_minutes for params in job.grid})
    for name, series in (("real", ohlc), ("control", control)):
        for lookback in lookbacks:
            rolling = RollingRange.build(series, lookback)
            for index, params in enumerate(job.grid):
                if params.lookback_minutes != lookback:
                    continue
                for optimistic in (False, True):
                    runs[(index, name, optimistic)] = simulate(
                        series, rolling, params, job.costs,
                        optimistic=optimistic, leverage=job.leverage)
    return SymbolResult(job.symbol, len(ohlc), int(ohlc.ts[0]), int(ohlc.ts[-1]),
                        sideways_now(ohlc, job.recent_days), runs)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def num(value: float, width: int = 8, digits: int = 1, sign: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return f"{'-':>{width}}"
    return f"{value:>{'+' if sign else ''}{width}.{digits}f}"


def pct(value: float, width: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return f"{'-':>{width}}"
    return f"{value:>{width}.0%}"


def stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def segment(trades: Trades, which: str, split_ts: float, recent_ts: float) -> Trades:
    if which == "is":
        return trades.between(-math.inf, split_ts)
    if which == "oos":
        return trades.between(split_ts, math.inf)
    if which == "recent":
        return trades.between(recent_ts, math.inf)
    raise ValueError(which)


def mean_or_nan(trades: Trades, minimum: int = MIN_TRADES) -> float:
    return float(trades.net_bps.mean()) if len(trades) >= minimum else math.nan


def ci(summary: Summary) -> str:
    return f"{num(summary.mean_bps)} [{num(summary.low_bps, 6)}, {num(summary.high_bps, 6)}]"


def report(results: Sequence[SymbolResult], grid: Sequence[RangeParams], *,
           costs: Costs, leverage: float, vip: int, split_ts: float,
           recent_ts: float, recent_days: int, draws: int, seed: int,
           start: date, end: date) -> None:
    rng = np.random.default_rng(seed + 1)
    usable = [result for result in results if result.bars]
    width = 108
    rule = "=" * width

    def section(title: str) -> None:
        print("\n" + rule + "\n" + title + "\n" + rule)

    section(f"RANGE FADE BACKTEST  {len(usable)} symbols, {start} .. {end}, "
            f"1m bars, BloFin VIP {vip} fees")
    print(f"  fees    maker {costs.maker_bps:.2f} / taker {costs.taker_bps:.2f} bps "
          f"per side; stop and time exits add {costs.slippage_bps:g} bps slippage")
    print(f"  fills   pessimistic: {costs.through_bps:g} bp trade-through, stop "
          "first when a bar holds both.  optimistic: touch, target first")
    print(f"  split   choose on entries before {stamp(split_ts)}, score after; "
          f"'recent' = entries from {stamp(recent_ts)}")
    print(f"  risk    {leverage:g}x isolated. Stops beyond liquidation are "
          "refused. Funding not modelled. All figures bps of notional per trade.")

    # ---- 1. sideways ------------------------------------------------------
    section(f"1. IS IT ACTUALLY SIDEWAYS?  the last {recent_days} days against "
            f"every {recent_days}-day window in the history")
    print(f"  {'symbol':<10}{'move':>8}{'range':>8}{'trend ratio':>13}"
          f"{'trend pctile':>14}{'range pctile':>14}")
    print("  " + "-" * 67)
    for result in usable:
        s = result.sideways
        if not s:
            print(f"  {result.symbol:<10}  not enough days")
            continue
        print(f"  {result.symbol:<10}{s['move']:>+8.1%}{s['width']:>8.1%}"
              f"{s['trend']:>13.2f}{pct(s['trend_pctile'], 14)}{pct(s['width_pctile'], 14)}")
    print("  trend ratio = net move / range width (0 went nowhere, 1 edge to edge).\n"
          "  pctile = share of the history's windows at or below this one: a LOW "
          "trend pctile is\n  more sideways than usual, a LOW range pctile is a "
          "tighter range than usual.")

    # ---- 2. grid ----------------------------------------------------------
    section("2. THE WHOLE GRID  pessimistic fills; medians are across symbols "
            f"with >= {MIN_TRADES} trades")
    stats = []
    for index, params in enumerate(grid):
        def means(which: str, name: str = "real", optimistic: bool = False) -> List[float]:
            return [mean_or_nan(segment(r.runs[(index, name, optimistic)], which,
                                        split_ts, recent_ts)) for r in usable]

        def pooled(which: str, name: str = "real", optimistic: bool = False) -> Trades:
            return Trades.concat([segment(r.runs[(index, name, optimistic)], which,
                                          split_ts, recent_ts) for r in usable])

        is_means = np.array(means("is"))
        oos_means = np.array(means("oos"))
        is_trades = pooled("is")
        recent_p, recent_o = pooled("recent"), pooled("recent", optimistic=True)
        finite_is = np.isfinite(is_means)
        stats.append({
            "index": index,
            "is_med": float(np.nanmedian(is_means)) if finite_is.any() else math.nan,
            "is_pos": int(np.sum(is_means[finite_is] > 0)),
            "is_n": int(finite_is.sum()),
            "is_trades": len(is_trades),
            "is_win": float(np.mean(is_trades.net_bps > 0)) if len(is_trades) else math.nan,
            "ctrl_is": mean_or_nan(pooled("is", "control"), 1),
            "oos_med": float(np.nanmedian(oos_means)) if np.isfinite(oos_means).any() else math.nan,
            "oos_pos": int(np.sum(oos_means[np.isfinite(oos_means)] > 0)),
            "oos_n": int(np.isfinite(oos_means).sum()),
            "recent_p": mean_or_nan(recent_p, 1),
            "recent_o": mean_or_nan(recent_o, 1),
            "recent_n": len(recent_p),
        })

    eligible = [row for row in stats
                if row["is_n"] >= math.ceil(len(usable) / 2) and np.isfinite(row["is_med"])]
    if not eligible:
        print("  No configuration traded often enough in-sample to be chosen.")
        return
    chosen = max(eligible, key=lambda row: row["is_med"])
    selected = chosen["index"]

    print(f"  {'config':<30}{'IS med':>8}{'IS +':>7}{'IS n':>8}{'win%':>6}"
          f"{'ctrl IS':>9}{'OOS med':>9}{'OOS +':>7}"
          f"{'recent P':>10}{'recent O':>10}{'n':>5}")
    print("  " + "-" * (width - 2))
    for row in sorted(stats, key=lambda r: -r["is_med"] if np.isfinite(r["is_med"]) else math.inf):
        mark = "*" if row["index"] == selected else " "
        print(f" {mark}{grid[row['index']].label():<30}{num(row['is_med'])}"
              f"{row['is_pos']:>4}/{row['is_n']:<2}{row['is_trades']:>8,}{pct(row['is_win'])}"
              f"{num(row['ctrl_is'], 9)}{num(row['oos_med'], 9)}"
              f"{row['oos_pos']:>4}/{row['oos_n']:<2}"
              f"{num(row['recent_p'], 10)}{num(row['recent_o'], 10)}{row['recent_n']:>5}")
    both = [(row["is_med"], row["oos_med"]) for row in stats
            if np.isfinite(row["is_med"]) and np.isfinite(row["oos_med"])]
    rank_corr = (spearman(np.array([a for a, _ in both]), np.array([b for _, b in both]))
                 if len(both) >= 3 else math.nan)
    print(f"\n  * chosen on IS median. IS+ / OOS+ = symbols with a positive mean. "
          f"ctrl = the shuffled-day control.\n  recent P / O = pooled mean over the "
          f"last {recent_days} days, pessimistic / optimistic.")
    print(f"  Does the in-sample ranking predict out-of-sample? Spearman across "
          f"configs: {num(rank_corr, 5, 2)}")

    # ---- 3 & 4. the chosen configuration ------------------------------------
    params = grid[selected]

    def detail(which: str, title: str) -> Dict[str, object]:
        section(title)
        print(f"  config {params.label()}   hold {params.max_hold_minutes / 60:g}h   "
              f"min width {params.min_width_bps:g} bps")
        print(f"\n  {'symbol':<10}{'trades':>7}{'win%':>6}{'tgt%':>6}{'stop%':>6}"
              f"{'time%':>6}{'pess':>8}{'opt':>8}{'ctrl P':>8}{'ctrl O':>8}{'real-ctrl':>11}")
        print("  " + "-" * 82)
        parts: Dict[Tuple[str, bool], List[Trades]] = {
            key: [] for key in (("real", False), ("real", True), ("control", False), ("control", True))}
        for r in usable:
            seg = {key: segment(r.runs[(selected, *key)], which, split_ts, recent_ts)
                   for key in parts}
            for key, trades in seg.items():
                parts[key].append(trades)
            real = seg[("real", False)]
            if len(real):
                reason = [np.mean(real.reason == code) for code in (TARGET, STOP, TIME)]
                diff = real.net_bps.mean() - seg[("control", False)].net_bps.mean() \
                    if len(seg[("control", False)]) else math.nan
                print(f"  {r.symbol:<10}{len(real):>7}{pct(np.mean(real.net_bps > 0))}"
                      f"{pct(reason[0])}{pct(reason[1])}{pct(reason[2])}"
                      f"{num(real.net_bps.mean())}{num(mean_or_nan(seg[('real', True)], 1))}"
                      f"{num(mean_or_nan(seg[('control', False)], 1))}"
                      f"{num(mean_or_nan(seg[('control', True)], 1))}{num(diff, 11)}")
            else:
                print(f"  {r.symbol:<10}{0:>7}")
        pooled = {key: Trades.concat(value) for key, value in parts.items()}
        sums = {key: summarise(trades, rng, draws) for key, trades in pooled.items()}
        diff = paired_difference(pooled[("real", False)], pooled[("control", False)], rng, draws)
        real = sums[("real", False)]
        print(f"\n  POOLED, 95% interval from resampling whole days across all symbols")
        print(f"    real, pessimistic     {ci(real)}   {real.trades:,} trades on {real.days} days")
        print(f"    real, optimistic      {ci(sums[('real', True)])}")
        print(f"    control, pessimistic  {ci(sums[('control', False)])}   <- a random walk "
              "with the same days")
        print(f"    control, optimistic   {ci(sums[('control', True)])}")
        print(f"    real - control (P)    {num(diff[0])} [{num(diff[1], 6)}, {num(diff[2], 6)}]"
              "   <- the mean-reversion edge")
        if real.trades:
            print(f"\n    wins {real.win_rate:.0%} averaging {num(real.avg_win_bps, 0)}, "
                  f"losses averaging {num(real.avg_loss_bps, 0)}; fees {real.fees_bps:.1f} "
                  f"bps/trade; {real.liquidated} liquidated")
            print(f"    each extra bp of stop slippage costs {real.taker_exit_share:.2f} "
                  "bps per trade")
        return {"sums": sums, "diff": diff, "pooled": pooled}

    oos = detail("oos", f"3. OUT OF SAMPLE  entries from {stamp(split_ts)} - the "
                        "config above, never re-fitted")
    recent = detail("recent", f"4. THE LAST {recent_days} DAYS  the regime you are "
                              "looking at, inside the out-of-sample part")

    # ---- 5. leverage --------------------------------------------------------
    section(f"5. WHAT {leverage:g}x DOES TO IT  out of sample, pessimistic, "
            "every trade at full allocation")
    print(f"  {'symbol':<10}{'trades':>7}{'refused':>9}{'liq':>5}{'equity 1x':>11}"
          f"{f'equity {leverage:g}x':>12}{f'max DD {leverage:g}x':>12}")
    print("  " + "-" * 64)
    for r in usable:
        full = r.runs[(selected, "real", False)]
        trades = segment(full, "oos", split_ts, recent_ts)
        one, _ = equity_path(trades.net_bps, 1.0)
        levered, drawdown = equity_path(trades.net_bps, leverage)
        print(f"  {r.symbol:<10}{len(trades):>7}{full.refused_liquidation:>9}"
              f"{int(np.sum(trades.reason == LIQUIDATED)):>5}{one:>11.3f}"
              f"{levered:>12.3f}{drawdown:>12.0%}")
    print(f"  equity = multiple of starting capital. `refused` counts the whole "
          f"history: ranges whose\n  stop sat beyond the {leverage:g}x liquidation "
          "price. Leverage multiplies the per-trade number;\n  it cannot change "
          "its sign.")

    # ---- verdict ------------------------------------------------------------
    section("VERDICT")
    pess, opt = oos["sums"][("real", False)], oos["sums"][("real", True)]
    edge = oos["diff"]
    if pess.trades == 0:
        print("  The chosen configuration placed no out-of-sample trades.")
        return
    if np.isfinite(opt.high_bps) and opt.high_bps < 0:
        print("  LOSES MONEY OUT OF SAMPLE EVEN UNDER OPTIMISTIC FILLS.")
        print(f"  Best in-sample of {len(grid)} configurations; out of sample it "
              f"averaged {pess.mean_bps:+.1f} bps (pessimistic) to {opt.mean_bps:+.1f} "
              "bps (optimistic)\n  per trade, and the optimistic interval sits "
              "entirely below zero.")
    elif np.isfinite(pess.low_bps) and pess.low_bps > 0:
        print("  PROFITABLE OUT OF SAMPLE UNDER PESSIMISTIC FILLS.")
        print(f"  {pess.mean_bps:+.1f} bps per trade [{pess.low_bps:+.1f}, "
              f"{pess.high_bps:+.1f}]. Confirm on a later period before believing it.")
    elif np.isfinite(opt.low_bps) and opt.low_bps > 0:
        print("  THE SIGN IS DECIDED BY THE FILL ASSUMPTION, NOT THE MARKET.")
        print(f"  Optimistic {opt.mean_bps:+.1f}, pessimistic {pess.mean_bps:+.1f} bps "
              "per trade. Bars cannot settle it; a fill simulation on trades can.")
    else:
        print("  NOT DISTINGUISHABLE FROM ZERO OUT OF SAMPLE.")
        print(f"  Pessimistic {pess.mean_bps:+.1f} [{pess.low_bps:+.1f}, "
              f"{pess.high_bps:+.1f}], optimistic {opt.mean_bps:+.1f} bps per trade.")

    if np.isfinite(edge[1]) and edge[1] > 0:
        print(f"\n  Against the control it earns {edge[0]:+.1f} bps [{edge[1]:+.1f}, "
              f"{edge[2]:+.1f}]: the order of moves inside a day DOES favour a fade.")
    elif np.isfinite(edge[2]) and edge[2] < 0:
        print(f"\n  It does WORSE than the shuffled control by {-edge[0]:.1f} bps "
              f"[{edge[1]:+.1f}, {edge[2]:+.1f}]: inside a day, moves at these "
              "scales tend to\n  continue rather than reverse, which is the "
              "opposite of what a fade needs.")
    else:
        print(f"\n  Against the control: {edge[0]:+.1f} bps [{edge[1]:+.1f}, "
              f"{edge[2]:+.1f}]. No evidence the real ordering of moves helps a fade "
              "more than\n  a random walk with the same days would.")

    print(f"\n  In-sample median {chosen['is_med']:+.1f} -> out-of-sample median "
          f"{chosen['oos_med']:+.1f} bps for the chosen config.")
    rp, ro = recent["sums"][("real", False)], recent["sums"][("real", True)]
    if rp.trades:
        print(f"\n  The last {recent_days} days: {rp.mean_bps:+.1f} (pessimistic) / "
              f"{ro.mean_bps:+.1f} (optimistic) bps per trade over {rp.trades} trades "
              f"on {rp.days} days.\n  That is {rp.days} days of one regime; the "
              "interval above is what it can and cannot say.")
    print(f"\n  Win rate {pess.win_rate:.0%} out of sample. Read it against the "
          "control, not on its own:\n  the geometry of a stop and a target sets it, "
          "on a random walk as much as here.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_floats(text: str) -> List[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_trends(text: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for part in text.split(","):
        part = part.strip().lower()
        if part:
            out.append(None if part in ("none", "off", "any") else float(part))
    return out


def build_grid(args) -> List[RangeParams]:
    grid = [
        RangeParams(lookback_minutes=int(round(hours * 60)), entry_frac=entry,
                    stop_frac=stop, target_frac=args.target_frac,
                    min_width_bps=args.min_width_bps, max_trend_ratio=trend)
        for hours in parse_floats(args.lookback_hours)
        for entry in parse_floats(args.entry_fracs)
        for stop in parse_floats(args.stop_fracs)
        for trend in parse_trends(args.trend_filters)
    ]
    problems = sorted({f"{params.label()}: {reason}"
                       for params in grid for reason in params.problems()})
    if problems:
        raise SystemExit("Refusing to run a grid with incoherent settings:\n  - "
                         + "\n  - ".join(problems))
    if not grid:
        raise SystemExit("The grid is empty - every list argument needs a value.")
    return grid


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS,
                        help="Binance USDT-M symbols, comma separated.")
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="Last UTC day, inclusive. Default: yesterday.")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cache", type=Path, default=repo_root / "data" / "cache")
    parser.add_argument("--lookback-hours", default="4,24,72")
    parser.add_argument("--entry-fracs", default="0.1,0.25")
    parser.add_argument("--stop-fracs", default="0.25,0.5")
    parser.add_argument("--trend-filters", default="none,0.5",
                        help="Max trend ratio per config; 'none' disables.")
    parser.add_argument("--target-frac", type=float, default=0.5)
    parser.add_argument("--min-width-bps", type=float, default=50.0)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--through-bps", type=float, default=1.0)
    parser.add_argument("--in-sample", type=float, default=0.7)
    parser.add_argument("--recent-days", type=int, default=21)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    reasons = []
    if args.leverage <= 0:
        reasons.append("--leverage must be positive")
    if not 0 < args.in_sample < 1:
        reasons.append("--in-sample must be between 0 and 1")
    if args.recent_days < 1 or args.recent_days >= args.days:
        reasons.append("--recent-days must be at least 1 and shorter than --days")
    if args.slippage_bps < 0 or args.through_bps < 0:
        reasons.append("--slippage-bps and --through-bps cannot be negative")
    if reasons:
        raise SystemExit("Refusing to run:\n  - " + "\n  - ".join(reasons))
    grid = build_grid(args)

    from config import MAKER_FEE_BPS, TAKER_FEE_BPS, VIP_TIER

    costs = Costs(maker_bps=float(MAKER_FEE_BPS), taker_bps=float(TAKER_FEE_BPS),
                  slippage_bps=args.slippage_bps, through_bps=args.through_bps)
    end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    start = end - timedelta(days=args.days - 1)
    days = daterange(start, end)
    symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]

    jobs: List[Job] = []
    for symbol in symbols:
        print(f"{symbol}: 1m klines {start} .. {end}")
        fetcher = Fetcher(args.cache / symbol)
        paths = fetcher.fetch_all(kline_urls(symbol, days, "klines", "1m"))
        if fetcher.missing:
            print(f"  not in the archive: {', '.join(sorted(fetcher.missing))}")
        if not paths:
            print("  nothing to load - skipped")
            continue
        jobs.append(Job(symbol, tuple(paths), tuple(grid), costs, args.leverage,
                        args.seed, args.recent_days))
    if not jobs:
        raise SystemExit("No symbol had any data.")

    first_ts = datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000
    last_ts = first_ts + args.days * DAY_MS
    split_ts = first_ts + args.in_sample * (last_ts - first_ts)
    recent_ts = last_ts - args.recent_days * DAY_MS

    workers = args.workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1, 6))
    print(f"\nSimulating {len(grid)} configs x 4 variants x {len(jobs)} symbols "
          f"on {workers} process(es)...", flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_symbol, jobs))
    print(f"  done in {time.time() - started:.0f}s")

    report(results, grid, costs=costs, leverage=args.leverage, vip=VIP_TIER,
           split_ts=split_ts, recent_ts=recent_ts, recent_days=args.recent_days,
           draws=args.draws, seed=args.seed, start=start, end=end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
