"""A daily cross-sectional panel: 100 perpetuals, five years, funding included.

    python backend\\analysis\\panel_daily.py                    # build / refresh
    python backend\\analysis\\panel_daily.py --symbols BTCUSDT,ETHUSDT
    python backend\\analysis\\panel_daily.py --no-funding        # prices only

Why this exists
---------------
Every dead branch in this repo forecast PRICE at a horizon of seconds to one
day, where the round trip is a large fraction of the move being forecast: the
range fade needed a centre IC of 0.177 (step 9l) and measured 0.011. The one
thing that has ever cleared its cost here is the funding carry, and it cleared
it because it is a CASH FLOW held for 30 days, so a 32 bps four-leg round trip
is amortised against hundreds of bps of collected funding.

That points at a quadrant nothing here has tested: a cross-section of many
instruments held for days to weeks. At a 7-day hold the taker round trip is
~10 bps against cross-sectional return dispersion of several hundred, so the
break-even IC is ~0.02 rather than 0.177 - an order of magnitude easier bar
than any previous step faced, on data already on disk.

What it reads, and what it does not re-download
-----------------------------------------------
Prices come from the 1-minute archive `fetch_klines.py` already pulled (5.7 GB,
100 symbols, 2021-09-12..2026-09-10). Aggregating those to days is expensive
once and free afterwards, so each symbol's daily rows are cached to
`data/panel/daily/<SYMBOL>.csv` with the source file list beside them; the
cache is rebuilt only when that list changes.

Funding is the one thing not in the bulk archive. It comes from
`/fapi/v1/fundingRate`, 1000 records a page, ~6 pages per symbol for five
years, cached to `data/cache/<SYMBOL>/funding.json` and extended incrementally.

Two alignment rules, both of which are the whole result if got wrong
-------------------------------------------------------------------
**A day's row closes at 00:00 UTC of the next day** and contains only
information timestamped at or before that instant. `close` is the last 1m close
of the day.

**Funding for day D is the funding that ACCRUED during day D**, i.e. the
settlements stamped 08:00 D, 16:00 D and 00:00 D+1 - because Binance's
settlement at 08:00 pays for the interval 00:00..08:00. A row therefore never
contains a rate that had not printed when the row closed. Bucketing on
`fundingTime`'s own date instead would shift the series one settlement into the
future, which reads as skill.

`minutes` is reported per row and is the only defence against a listing gap or
an exchange outage turning into a fabricated return: a day with 200 of its 1440
minutes present has a `close` that is not a close, and the harness filters on
it rather than trusting the row count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bars_import import KLINE_COLUMNS  # noqa: E402
from analysis.binance_import import read_zip_rows  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "data" / "cache"
PANEL = REPO_ROOT / "data" / "panel"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

DAY_MS = 86_400_000
MINUTES_PER_DAY = 1440

DAILY_COLUMNS = [
    "date", "symbol", "open", "high", "low", "close",
    "quote_volume", "trades", "taker_buy_frac", "minutes", "rv_bps",
]
PANEL_COLUMNS = DAILY_COLUMNS + ["funding_bps", "funding_periods"]


# ---------------------------------------------------------------------------
# Per-symbol daily aggregation from the 1m archive
# ---------------------------------------------------------------------------


def kline_zips(symbol: str, cache: Path = CACHE) -> List[Path]:
    """Every cached 1m archive for a symbol.

    Monthly and daily archives overlap at the month edges by design (the
    fetcher takes monthlies for complete months and dailies for the ragged
    ends), so the aggregator de-duplicates on minute timestamp rather than
    assuming the file set is disjoint.
    """
    directory = cache / symbol
    if not directory.is_dir():
        return []
    return sorted(directory.glob(symbol + "-1m-*.zip"))


def _day_key(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d")


@dataclass
class DayAccumulator:
    """One UTC day of 1-minute bars, folded as they arrive."""

    first_ts: int
    open: float
    high: float
    low: float
    close: float
    last_ts: int
    quote_volume: float = 0.0
    trades: float = 0.0
    taker_buy_quote: float = 0.0
    minutes: int = 0
    sum_sq_logret: float = 0.0
    previous_close: float = 0.0


def aggregate_symbol(symbol: str, cache: Path = CACHE) -> List[Dict[str, object]]:
    """Fold a symbol's whole 1m history into daily rows.

    `close_time` is the anchor, matching `bars_import.load_bars`: a bar stamped
    00:00:59.999 on day D belongs to day D. Minutes are de-duplicated by
    timestamp because monthly and daily archives overlap.

    Realised volatility is the root sum of squared 1-minute log returns within
    the day, in bps - the intraday number a daily bar cannot carry, and the
    reason aggregating the 1m archive is worth its cost over downloading 1d
    bars. Steps are taken WITHIN a day only; the overnight step belongs to the
    daily return, not to intraday vol.
    """
    seen: Dict[int, Tuple[float, float, float, float, float, float, float]] = {}
    for path in kline_zips(symbol, cache):
        for row in read_zip_rows(path, KLINE_COLUMNS):
            try:
                ts = int(float(row["close_time"]))
                if ts > 10_000_000_000_000:          # microseconds in some archives
                    ts //= 1000
                seen[ts] = (
                    float(row["open"]), float(row["high"]), float(row["low"]),
                    float(row["close"]), float(row["quote_volume"]),
                    float(row["count"]), float(row["taker_buy_quote_volume"]),
                )
            except (TypeError, ValueError, KeyError):
                continue

    days: Dict[str, DayAccumulator] = {}
    for ts in sorted(seen):
        opn, high, low, close, quote, count, taker_quote = seen[ts]
        if not (close > 0 and high > 0 and low > 0):
            continue
        key = _day_key(ts)
        day = days.get(key)
        if day is None:
            day = DayAccumulator(first_ts=ts, open=opn, high=high, low=low,
                                 close=close, last_ts=ts)
            days[key] = day
        else:
            day.high = max(day.high, high)
            day.low = min(day.low, low)
            if ts >= day.last_ts:
                day.close = close
                day.last_ts = ts
            if day.previous_close > 0:
                step = math.log(close / day.previous_close)
                day.sum_sq_logret += step * step
        day.previous_close = close
        day.quote_volume += quote
        day.trades += count
        day.taker_buy_quote += taker_quote
        day.minutes += 1

    rows: List[Dict[str, object]] = []
    for key in sorted(days):
        day = days[key]
        rows.append({
            "date": key,
            "symbol": symbol,
            "open": day.open,
            "high": day.high,
            "low": day.low,
            "close": day.close,
            "quote_volume": day.quote_volume,
            "trades": day.trades,
            "taker_buy_frac": (day.taker_buy_quote / day.quote_volume
                               if day.quote_volume > 0 else float("nan")),
            "minutes": day.minutes,
            "rv_bps": math.sqrt(day.sum_sq_logret) * 10_000.0,
        })
    return rows


def sources_signature(symbol: str, cache: Path) -> str:
    """What the daily cache was built from. Rebuild when this changes.

    Names and sizes, not mtimes: re-downloading an identical archive must not
    invalidate a cache that is still correct, while a new month or a file that
    grew must.
    """
    return "|".join(path.name + ":" + str(path.stat().st_size)
                    for path in kline_zips(symbol, cache))


def daily_cache_paths(symbol: str, panel: Path = PANEL) -> Tuple[Path, Path]:
    directory = panel / "daily"
    return directory / (symbol + ".csv"), directory / (symbol + ".sources")


def load_or_build_daily(symbol: str, *, cache: Path = CACHE, panel: Path = PANEL,
                        force: bool = False) -> List[Dict[str, object]]:
    """Daily rows for one symbol, from cache when the source set is unchanged."""
    csv_path, sources_path = daily_cache_paths(symbol, panel)
    signature = sources_signature(symbol, cache)
    if not signature:
        return []

    if not force and csv_path.exists() and sources_path.exists():
        if sources_path.read_text(encoding="utf-8").strip() == signature:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                return [
                    {key: (value if key in ("date", "symbol") else float(value))
                     for key, value in row.items()}
                    for row in csv.DictReader(handle)
                ]

    rows = aggregate_symbol(symbol, cache)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAILY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    sources_path.write_text(signature, encoding="utf-8")
    return rows


def _build_one(args: Tuple[str, str, str, bool]) -> Tuple[str, int]:
    symbol, cache, panel, force = args
    rows = load_or_build_daily(symbol, cache=Path(cache), panel=Path(panel),
                               force=force)
    return symbol, len(rows)


# ---------------------------------------------------------------------------
# Funding history
# ---------------------------------------------------------------------------


def get_json(url: str, opener: Optional[Callable[[str], bytes]] = None) -> object:
    """Injectable fetch with a back-off. Tests pass an `opener` and no socket."""
    if opener is not None:
        return json.loads(opener(url))
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (418, 429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < 4:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise RuntimeError("gave up on " + url)


def fetch_funding(symbol: str, *, start_ms: int, cache: Path = CACHE,
                  opener: Optional[Callable[[str], bytes]] = None,
                  ) -> Dict[int, float]:
    """`{fundingTime_ms: rate_bps}` back to `start_ms`, cached and extended.

    The endpoint pages forward from `startTime` 1000 records at a time and has
    no bulk archive, so this is the one part of the panel that costs requests:
    about six per symbol for five years, and one per symbol per refresh
    afterwards. The cache is a dict keyed by settlement time, so re-fetching an
    overlapping page is idempotent rather than duplicating rows.
    """
    path = cache / symbol / "funding.json"
    rates: Dict[int, float] = {}
    if path.exists():
        try:
            rates = {int(key): float(value)
                     for key, value in json.loads(path.read_text()).items()}
        except (ValueError, TypeError):
            rates = {}

    cursor = max(rates) + 1 if rates else start_ms
    if rates and min(rates) > start_ms + DAY_MS:
        cursor = start_ms                      # cache starts too late; refill

    while True:
        url = (FUNDING_URL + "?symbol=" + symbol + "&startTime=" + str(cursor)
               + "&limit=1000")
        payload = get_json(url, opener)
        if not isinstance(payload, list) or not payload:
            break
        newest = cursor
        for row in payload:
            try:
                ts = int(row["fundingTime"])
                rates[ts] = float(row["fundingRate"]) * 10_000.0
                newest = max(newest, ts)
            except (KeyError, TypeError, ValueError):
                continue
        if len(payload) < 1000 or newest <= cursor:
            break
        cursor = newest + 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(key): value
                                for key, value in sorted(rates.items())}))
    return rates


def funding_by_day(rates: Dict[int, float]) -> Dict[str, Tuple[float, int]]:
    """`{date: (bps accrued during that UTC day, periods)}`.

    A settlement stamped T pays for the interval ENDING at T, so the rate
    stamped 00:00 on day D+1 accrued during day D and belongs to day D's row.
    Shifting this by one settlement is the difference between a feature and a
    look-ahead, so the bucket key is the instant just BEFORE the settlement.

    The stamp is rounded to its nominal minute first. Settlements print a few
    milliseconds late - measured on ADAUSDT's 5,479 settlements, which are all
    8h apart to within 1.2s - so subtracting a millisecond from a stamp of
    00:00:00.002 leaves it on the wrong side of midnight, and the day it
    belongs to gets 2 periods while the next gets 4. Rounding first removes the
    jitter exactly instead of guessing a back-off big enough to absorb it.
    """
    out: Dict[str, List[float]] = {}
    for ts, bps in rates.items():
        nominal = int(round(ts / 60_000.0)) * 60_000
        bucket = out.setdefault(_day_key(nominal - 1), [0.0, 0.0])
        bucket[0] += bps
        bucket[1] += 1
    return {day: (value[0], int(value[1])) for day, value in out.items()}


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------


def cached_symbols(cache: Path = CACHE) -> List[str]:
    """Symbols with a 1m archive. ASCII only, matching `fetch_klines`."""
    out: List[str] = []
    if not cache.is_dir():
        return out
    for directory in sorted(cache.iterdir()):
        if not directory.is_dir() or not directory.name.isascii():
            continue
        if kline_zips(directory.name, cache):
            out.append(directory.name)
    return out


def build_panel(symbols: Sequence[str], *, cache: Path = CACHE, panel: Path = PANEL,
                with_funding: bool = True, force: bool = False,
                workers: int = 6) -> List[Dict[str, object]]:
    print("Aggregating 1m archives to days for " + str(len(symbols))
          + " symbols (" + str(workers) + " processes)...")
    jobs = [(symbol, str(cache), str(panel), force) for symbol in symbols]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (symbol, count) in enumerate(pool.map(_build_one, jobs), start=1):
            print("\r  {}/{}  {:<16} {:,} days      ".format(
                index, len(symbols), symbol, count), end="", flush=True)
    print()

    daily = {symbol: load_or_build_daily(symbol, cache=cache, panel=panel)
             for symbol in symbols}

    funding: Dict[str, Dict[str, Tuple[float, int]]] = {}
    if with_funding:
        print("Fetching funding history (cached; ~6 requests per new symbol)...")
        earliest = min((rows[0]["date"] for rows in daily.values() if rows),
                       default="2021-09-12")
        start_ms = int(datetime.strptime(str(earliest), "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        for index, symbol in enumerate(symbols, start=1):
            try:
                funding[symbol] = funding_by_day(
                    fetch_funding(symbol, start_ms=start_ms, cache=cache))
            except Exception as exc:                      # noqa: BLE001
                print("\n  " + symbol + ": funding unavailable (" + str(exc) + ")")
                funding[symbol] = {}
            print("\r  {}/{}  {:<16} {:,} days      ".format(
                index, len(symbols), symbol, len(funding[symbol])),
                end="", flush=True)
        print()

    rows: List[Dict[str, object]] = []
    for symbol in symbols:
        table = funding.get(symbol, {})
        for row in daily[symbol]:
            bps, periods = table.get(str(row["date"]), (float("nan"), 0))
            out = dict(row)
            out["funding_bps"] = bps
            out["funding_periods"] = periods
            rows.append(out)
    rows.sort(key=lambda row: (row["date"], row["symbol"]))
    return rows


def write_panel(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PANEL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PANEL_COLUMNS})


def report(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        print("No rows. Is data/cache populated? Run fetch_klines.py first.")
        return
    symbols = sorted({str(row["symbol"]) for row in rows})
    dates = sorted({str(row["date"]) for row in rows})
    complete = [row for row in rows if float(row["minutes"]) >= MINUTES_PER_DAY - 10]
    with_funding = [row for row in rows if float(row["funding_periods"]) > 0]
    print()
    print("rows                {:,}".format(len(rows)))
    print("symbols             {}".format(len(symbols)))
    print("dates               {}  ({} .. {})".format(len(dates), dates[0], dates[-1]))
    print("complete days       {:,}  ({:.1f}%)".format(
        len(complete), 100.0 * len(complete) / len(rows)))
    print("days with funding   {:,}  ({:.1f}%)".format(
        len(with_funding), 100.0 * len(with_funding) / len(rows)))

    breadth: Dict[str, int] = {}
    for row in complete:
        breadth[str(row["date"])] = breadth.get(str(row["date"]), 0) + 1
    if breadth:
        counts = sorted(breadth.values())
        print("symbols per day     min {}  median {}  max {}".format(
            counts[0], counts[len(counts) // 2], counts[-1]))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", help="comma-separated; default every cached symbol")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-funding", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="re-aggregate the 1m archives even if cached")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else cached_symbols(args.cache))
    if not symbols:
        raise SystemExit(
            "No 1m archives under " + str(args.cache) + ".\n"
            "  Run: python backend\\analysis\\fetch_klines.py --top 100 --years 5")

    rows = build_panel(symbols, cache=args.cache, panel=args.panel,
                       with_funding=not args.no_funding, force=args.force,
                       workers=args.workers)
    out = args.out or (args.panel / "daily.csv")
    write_panel(rows, out)
    report(rows)
    print("\nwrote " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
