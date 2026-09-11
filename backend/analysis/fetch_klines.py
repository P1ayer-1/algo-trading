"""Bulk 1m klines from Binance's free archive, with the gaps reported.

    python backend\\analysis\\fetch_klines.py --top 100 --years 5
    python backend\\analysis\\fetch_klines.py --symbols BTCUSDT,ETHUSDT --days 365
    python backend\\analysis\\fetch_klines.py --top 100 --years 5 --dry-run

Why this exists
---------------
`range_harness.py` measures a model against a threshold, and what it kept
finding first was the sample: a year of one symbol is ~363 independent 24h
windows, and ten correlated majors do not multiply that. Any model with real
capacity - an attention network over bars, an RL policy - needs orders of
magnitude more than that before it can be tested rather than merely fitted.

The archive is free, needs no key, and is current to yesterday: monthly files
for complete past months, daily for the edges, which turns five years of a
hundred symbols into about 9,000 requests rather than 180,000.

What it does NOT do
-------------------
It does not make the samples independent. A hundred symbols that all follow
BTC are not a hundred times the information, and five years of one venue is
still one venue. It buys history and breadth; it does not buy the effective N
that a t-statistic wants, and `range_harness` keeps correcting for that
regardless of how much is downloaded here.

Tokenised equities and commodities are excluded by default. Binance lists them
as TRADIFI_PERPETUAL, and on 2026-09-11 seven of the hundred most-traded
contracts were gold, crude, silver and single stocks - instruments with
trading hours, whose session gaps a crypto model would read as structure.

Coverage is reported per symbol because a listing date is invisible otherwise:
a symbol that started trading in 2024 silently returns three years of 404s,
and a model trained on "five years" of it is trained on whatever arrived.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bars_import import Fetcher, daterange, kline_urls  # noqa: E402

TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
# Measured from the cache on 2026-09-11 over 380 daily and 110 monthly 1m
# archives: daily median 0.057 MB, monthly median 1.67 MB (0.055 MB/day), so a
# symbol-year is about 20 MB. Used only for the --dry-run estimate.
MB_PER_SYMBOL_DAY = 0.055


@dataclass
class Coverage:
    symbol: str
    files: int = 0
    days_wanted: int = 0
    days_covered: int = 0
    first_day: Optional[str] = None
    last_day: Optional[str] = None
    bytes_on_disk: int = 0

    @property
    def share(self) -> float:
        return self.days_covered / self.days_wanted if self.days_wanted else 0.0


def fetch_ticker_payload(opener: Optional[Callable[[str], bytes]] = None) -> List[dict]:
    """24h stats for every futures symbol. `opener` is injectable for tests."""
    if opener is None:
        def opener(url: str) -> bytes:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
    payload = json.loads(opener(TICKER_URL))
    if not isinstance(payload, list):
        raise SystemExit(f"{TICKER_URL} did not return a list of tickers.")
    return payload


def fetch_exchange_info(opener: Optional[Callable[[str], bytes]] = None) -> List[dict]:
    """Every listed contract with its asset class. `opener` is injectable."""
    if opener is None:
        def opener(url: str) -> bytes:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
    payload = json.loads(opener(EXCHANGE_INFO_URL))
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not symbols:
        raise SystemExit(f"{EXCHANGE_INFO_URL} returned no symbols.")
    return symbols


def crypto_perpetuals(symbols: Sequence[dict]) -> set:
    """Crypto perps only - not gold, not crude, not Samsung.

    Binance lists tokenised equities and commodities as TRADIFI_PERPETUAL
    (XAUUSDT, CLUSDT, SOXLUSDT, SKHYNIXUSDT), and on 2026-09-11 seven of the
    hundred most-traded contracts were those. They do not belong in a pooled
    crypto dataset: an equity trades in sessions, so its bars carry overnight
    and weekend gaps that a crypto model reads as market structure, and the
    dynamics of a commodity are not what a range model for BTC is being asked
    about. The exchange labels the difference, so this filters on the label
    rather than on a list of names that goes stale at the next listing.
    """
    return {str(row.get("symbol")) for row in symbols
            if row.get("contractType") == "PERPETUAL"
            and row.get("underlyingType") == "COIN"
            and row.get("status") == "TRADING"}


def rank_symbols(payload: Sequence[dict], *, top: int, quote: str = "USDT",
                 min_volume_usd: float = 0.0,
                 allowed: Optional[set] = None) -> List[str]:
    """The `top` most-traded perpetuals in `quote`, busiest first.

    Dated futures (BTCUSDT_240329) are excluded: they expire, so their history
    stops for reasons that have nothing to do with the market. `allowed`, when
    given, restricts the universe further - see `crypto_perpetuals`.
    """
    rows: List[Tuple[float, str]] = []
    for row in payload:
        symbol = str(row.get("symbol", ""))
        if not symbol.endswith(quote) or "_" in symbol:
            continue
        if not (symbol.isascii() and symbol.isalnum()):
            # Binance lists meme perps under CJK tickers - four of the top
            # hundred on 2026-09-11. The archive path is the symbol verbatim,
            # and a non-ASCII path cannot be encoded into an HTTP request line,
            # so these are unfetchable rather than merely unusual.
            continue
        if allowed is not None and symbol not in allowed:
            continue
        try:
            volume = float(row.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            continue
        if volume < min_volume_usd:
            continue
        rows.append((volume, symbol))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [symbol for _, symbol in rows[:max(0, top)]]


def covered_days(names: Sequence[str], symbol: str, days: Sequence[date],
                 interval: str) -> List[date]:
    """Days a present file accounts for - a monthly file covers its whole month.

    Names may carry the .zip suffix or not; what distinguishes a monthly file
    from a daily one is the stamp it ends with, YYYY-MM against YYYY-MM-DD.
    """
    prefix = f"{symbol}-{interval}-"
    months, exact = set(), set()
    for name in names:
        if not name.startswith(prefix):
            continue
        stamp = name[len(prefix):]
        if stamp.endswith(".zip"):
            stamp = stamp[:-len(".zip")]
        parts = stamp.split("-")
        if len(parts) == 2:
            months.add(stamp)
        elif len(parts) == 3:
            exact.add(stamp)
    return [day for day in days
            if day.isoformat() in exact
            or f"{day.year:04d}-{day.month:02d}" in months]


def coverage_of(cache: Path, symbol: str, days: Sequence[date],
                interval: str) -> Coverage:
    directory = cache / symbol
    report = Coverage(symbol=symbol, days_wanted=len(days))
    if not directory.exists():
        return report
    present = [path for path in directory.iterdir()
               if path.name.startswith(f"{symbol}-{interval}-") and path.suffix == ".zip"]
    report.files = len(present)
    report.bytes_on_disk = sum(path.stat().st_size for path in present)
    covered = covered_days([path.stem for path in present], symbol, days, interval)
    report.days_covered = len(covered)
    if covered:
        report.first_day = covered[0].isoformat()
        report.last_day = covered[-1].isoformat()
    return report


def write_manifest(path: Path, reports: Sequence[Coverage], *, start: date, end: date,
                   interval: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval": interval,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": [{
            "symbol": report.symbol, "files": report.files,
            "days_wanted": report.days_wanted, "days_covered": report.days_covered,
            "first_day": report.first_day, "last_day": report.last_day,
            "megabytes": round(report.bytes_on_disk / 1e6, 1),
        } for report in reports],
    }, indent=2), encoding="utf-8")


def problems(args) -> List[str]:
    reasons = []
    if args.top < 1 and not args.symbols:
        reasons.append("--top must be at least 1, or pass --symbols")
    if args.days < 1:
        reasons.append("--days must be at least 1")
    if args.workers < 1:
        reasons.append("--workers must be at least 1")
    if args.min_volume_usd < 0:
        reasons.append("--min-volume-usd cannot be negative")
    return reasons


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--symbols", default=None,
                        help="Explicit list. Overrides --top.")
    parser.add_argument("--top", type=int, default=100,
                        help="How many symbols, by 24h quote volume.")
    parser.add_argument("--years", type=float, default=None,
                        help="Shorthand for --days.")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--cache", type=Path, default=repo_root / "data" / "cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-volume-usd", type=float, default=1e6)
    parser.add_argument("--include-tradfi", action="store_true",
                        help="Keep tokenised equities and commodities "
                             "(XAUUSDT, SOXLUSDT...). They have trading hours.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the disk it needs. Fetch nothing.")
    args = parser.parse_args(argv)
    if args.years is not None:
        args.days = int(round(args.years * 365))
    reasons = problems(args)
    if reasons:
        raise SystemExit("Refusing to run:\n  - " + "\n  - ".join(reasons))

    end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    start = end - timedelta(days=args.days - 1)
    days = daterange(start, end)

    if args.symbols:
        symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    else:
        print("Ranking symbols by 24h volume...")
        allowed = None
        if not args.include_tradfi:
            allowed = crypto_perpetuals(fetch_exchange_info())
            print(f"  {len(allowed)} crypto perpetuals listed; tokenised equities "
                  "and commodities excluded")
        symbols = rank_symbols(fetch_ticker_payload(), top=args.top,
                               min_volume_usd=args.min_volume_usd, allowed=allowed)
        if not symbols:
            raise SystemExit("No symbol cleared the volume floor.")
    print(f"{len(symbols)} symbols, {start} .. {end} ({len(days)} days), "
          f"{args.interval} klines")

    if args.dry_run:
        requests = sum(len(kline_urls(symbol, days, "klines", args.interval))
                       for symbol in symbols)
        gigabytes = len(symbols) * len(days) * MB_PER_SYMBOL_DAY / 1000
        print(f"  {requests:,} requests, about {gigabytes:.1f} GB if every symbol "
              f"traded for the whole period.")
        print(f"  Symbols: {', '.join(symbols[:12])}"
              f"{' ...' if len(symbols) > 12 else ''}")
        print("  Nothing downloaded (--dry-run).")
        return 0

    reports: List[Coverage] = []
    failures: List[str] = []
    downloaded = cached = 0
    for index, symbol in enumerate(symbols, start=1):
        fetcher = Fetcher(args.cache / symbol, workers=args.workers)
        try:
            fetcher.fetch_all(kline_urls(symbol, days, "klines", args.interval))
        except Exception as exc:  # noqa: BLE001
            # One symbol must not end a run of a hundred: the archive is the
            # slow half, and whatever arrived before the failure is still on
            # disk and still counted by coverage_of below.
            failures.append(f"{symbol}: {type(exc).__name__}: {exc}")
            print(f"  [{index:>3}/{len(symbols)}] {symbol:<14}FAILED "
                  f"({type(exc).__name__}) - continuing", flush=True)
        downloaded += fetcher.downloaded
        cached += fetcher.cached
        report = coverage_of(args.cache, symbol, days, args.interval)
        reports.append(report)
        print(f"  [{index:>3}/{len(symbols)}] {symbol:<14}{report.days_covered:>6,}"
              f"/{report.days_wanted:<6,} days {report.share:>5.0%}   "
              f"{report.first_day or '-'} .. {report.last_day or '-'}   "
              f"{report.bytes_on_disk / 1e6:>8,.0f} MB", flush=True)

    manifest = args.manifest or (args.cache / f"manifest-{args.interval}.json")
    write_manifest(manifest, reports, start=start, end=end, interval=args.interval)

    total_mb = sum(report.bytes_on_disk for report in reports) / 1e6
    full = [report for report in reports if report.share >= 0.99]
    partial = [report for report in reports if report.share < 0.5]
    print(f"\n  {downloaded:,} files downloaded, {cached:,} already cached, "
          f"{total_mb / 1000:.1f} GB on disk")
    print(f"  {len(full)} symbol(s) cover the whole period; {len(partial)} cover "
          f"under half of it")
    if failures:
        print(f"  {len(failures)} symbol(s) failed:")
        for failure in failures[:10]:
            print(f"    {failure}")
    if partial:
        names = ", ".join(f"{report.symbol} ({report.share:.0%})" for report in partial[:10])
        print(f"    thin: {names}")
        print("    Usually a listing date rather than a fault - but a model trained "
              "on these\n    is trained on whatever arrived, so read the manifest "
              "before pooling them.")
    print(f"  manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
