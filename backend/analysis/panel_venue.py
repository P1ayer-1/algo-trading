"""The same daily panel from any of several venues, behind one adapter.

    python backend\\analysis\\panel_venue.py --venue bybit --top 60
    python backend\\analysis\\panel_venue.py --venue mexc --top 60
    python backend\\analysis\\panel_venue.py --list

Why this is generic where the first three panels were not
---------------------------------------------------------
`panel_daily`, `panel_blofin` and `panel_hyperliquid` are one file per venue
because each was written to answer a different question. This one exists
because the question changed shape: step 9r's pair trade needs a venue with LOW
funding to pair against BloFin's high funding, Binance was that venue and is
not available everywhere, and the right answer is therefore not "port the
strategy to venue X" but "make substituting a venue cheap".

An adapter is four functions - the universe with its 24h volume, daily candles,
funding history, and how the venue spells a coin. Everything else, including
both alignment rules, is shared.

Measured live on 2026-09-12, mean funding in bps per day over ~50 settlements
on ten majors, every one of these answering with no API key:

    BloFin 3.78 | Aster 2.18 | Bitget 2.09 | Hyperliquid 1.60 | KuCoin 1.56
    OKX 1.47 | dYdX 1.38 | MEXC 1.33 | Bybit 1.07 | Gate 1.02 | Kraken -0.51

BloFin is the dearest of the ten, which is what makes it the short leg of the
pair whichever venue ends up on the other side. The spread is what decides the
trade, so it is reported per venue rather than assumed.

What differs between venues, and what must not
----------------------------------------------
**Cadence.** 8h on most, 4h on some contracts, 1h on Hyperliquid. A daily TOTAL
is the comparable unit; a per-settlement rate is not. The interval is read off
the settlement timestamps rather than from documentation, because a venue that
changed cadence documents the new one.

**History depth.** Measured 2026-09-12: Bybit pages a full history 200 at a
time; MEXC gives 17 pages of 100, about 18 months; Bitget pages 100 at a time;
Gate returns only ~90 settlements however large a limit it is given, which is
30 days and not enough to backtest. Gate is therefore listed as a live-spread
venue and not as a panel source, and `--list` says so rather than producing a
short panel that looks like the others.

**Nothing else.** Both alignment rules are enforced here, once: a row closes at
00:00 UTC, and funding for day D is what ACCRUED during day D - the settlement
stamped 00:00 on D+1, rounded to its nominal minute first. A venue whose panel
disagreed with the others about either would produce a funding difference that
is a parsing artifact, and the entire pair trade is a funding difference.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.panel_daily import write_panel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "data" / "cache"
PANEL = REPO_ROOT / "data" / "panel"
DAY_MS = 86_400_000
PARKINSON = 1.0 / (2.0 * math.sqrt(math.log(2.0)))


def _day_key(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d")


def get_json(url: str, *, attempts: int = 4, pause: float = 1.5) -> Any:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0",
                              "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:                             # noqa: BLE001
            if attempt == attempts - 1:
                raise
            time.sleep(pause * (attempt + 1))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


@dataclass
class Listing:
    symbol: str                   # the venue's own spelling
    coin: str                     # canonical base asset
    volume_usd_24h: float


Candle = Tuple[int, float, float, float, float, float, float]
# (day_start_ms, open, high, low, close, quote_volume, trades)


@dataclass
class Adapter:
    name: str
    universe: Callable[[], List[Listing]]
    candles: Callable[[str], List[Candle]]
    funding: Callable[[str], Dict[int, float]]
    panel_source: bool = True
    note: str = ""


# -- Bybit -------------------------------------------------------------------


def _bybit_universe() -> List[Listing]:
    rows = get_json("https://api.bybit.com/v5/market/tickers?category=linear")
    out = []
    for row in rows["result"]["list"]:
        symbol = str(row.get("symbol", ""))
        if not symbol.endswith("USDT"):
            continue
        try:
            volume = float(row.get("turnover24h") or 0.0)
        except (TypeError, ValueError):
            continue
        out.append(Listing(symbol, symbol[:-4], volume))
    return out


def _bybit_candles(symbol: str) -> List[Candle]:
    """Daily klines, paged backwards. Bybit returns newest first, 1000 max."""
    out: Dict[int, Candle] = {}
    end = int(time.time() * 1000)
    for _ in range(6):
        url = ("https://api.bybit.com/v5/market/kline?category=linear&symbol="
               + symbol + "&interval=D&limit=1000&end=" + str(end))
        rows = get_json(url)["result"]["list"]
        if not rows:
            break
        for row in rows:
            ts = int(row[0])
            close, volume = float(row[4]), float(row[6])
            out[ts] = (ts, float(row[1]), float(row[2]), float(row[3]),
                       close, volume, 0.0)
        oldest = min(int(row[0]) for row in rows)
        if oldest >= end - DAY_MS or len(rows) < 1000:
            break
        end = oldest - 1
    return [out[ts] for ts in sorted(out)]


def _bybit_funding(symbol: str) -> Dict[int, float]:
    """Paged FORWARD with startTime, 200 at a time."""
    out: Dict[int, float] = {}
    cursor = 1_672_531_200_000                        # 2023-01-01
    now = int(time.time() * 1000)
    for _ in range(80):
        if cursor >= now:
            break
        url = ("https://api.bybit.com/v5/market/funding/history?category=linear"
               "&symbol=" + symbol + "&limit=200&startTime=" + str(cursor)
               + "&endTime=" + str(min(now, cursor + 200 * 8 * 3_600_000)))
        rows = get_json(url)["result"]["list"]
        if not rows:
            cursor += 200 * 8 * 3_600_000
            continue
        newest = cursor
        for row in rows:
            ts = int(row["fundingRateTimestamp"])
            out[ts] = float(row["fundingRate"]) * 10_000.0
            newest = max(newest, ts)
        cursor = newest + 1
    return out


# -- MEXC --------------------------------------------------------------------


def _mexc_universe() -> List[Listing]:
    rows = get_json("https://contract.mexc.com/api/v1/contract/ticker")["data"]
    out = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if not symbol.endswith("_USDT"):
            continue
        try:
            volume = float(row.get("amount24") or 0.0)
        except (TypeError, ValueError):
            continue
        out.append(Listing(symbol, symbol[:-5], volume))
    return out


def _mexc_candles(symbol: str) -> List[Candle]:
    end = int(time.time())
    start = end - 5 * 365 * 86400
    url = ("https://contract.mexc.com/api/v1/contract/kline/" + symbol
           + "?interval=Day1&start=" + str(start) + "&end=" + str(end))
    data = get_json(url).get("data") or {}
    times = data.get("time") or []
    out: List[Candle] = []
    for index, ts in enumerate(times):
        try:
            out.append((int(ts) * 1000, float(data["open"][index]),
                        float(data["high"][index]), float(data["low"][index]),
                        float(data["close"][index]), float(data["amount"][index]),
                        0.0))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    out.sort()
    return out


def _mexc_funding(symbol: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    page = 1
    while page <= 40:
        url = ("https://contract.mexc.com/api/v1/contract/funding_rate/history"
               "?symbol=" + symbol + "&page_num=" + str(page) + "&page_size=100")
        data = get_json(url).get("data") or {}
        rows = data.get("resultList") or []
        if not rows:
            break
        for row in rows:
            try:
                out[int(row["settleTime"])] = float(row["fundingRate"]) * 10_000.0
            except (KeyError, TypeError, ValueError):
                continue
        total = int(data.get("totalPage") or page)
        if page >= total:
            break
        page += 1
    return out


# -- Bitget ------------------------------------------------------------------


def _bitget_universe() -> List[Listing]:
    rows = get_json("https://api.bitget.com/api/v2/mix/market/tickers"
                    "?productType=usdt-futures")["data"]
    out = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if not symbol.endswith("USDT"):
            continue
        try:
            volume = float(row.get("usdtVolume") or 0.0)
        except (TypeError, ValueError):
            continue
        out.append(Listing(symbol, symbol[:-4], volume))
    return out


def _bitget_candles(symbol: str) -> List[Candle]:
    out: Dict[int, Candle] = {}
    end = int(time.time() * 1000)
    for _ in range(12):
        url = ("https://api.bitget.com/api/v2/mix/market/history-candles?symbol="
               + symbol + "&productType=usdt-futures&granularity=1D&limit=200"
               "&endTime=" + str(end))
        rows = get_json(url).get("data") or []
        if not rows:
            break
        for row in rows:
            ts = int(row[0])
            out[ts] = (ts, float(row[1]), float(row[2]), float(row[3]),
                       float(row[4]), float(row[6]), 0.0)
        oldest = min(int(row[0]) for row in rows)
        if oldest >= end - DAY_MS or len(rows) < 200:
            break
        end = oldest - 1
    return [out[ts] for ts in sorted(out)]


def _bitget_funding(symbol: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    page = 1
    while page <= 40:
        url = ("https://api.bitget.com/api/v2/mix/market/history-fund-rate?symbol="
               + symbol + "&productType=usdt-futures&pageSize=100&pageNo=" + str(page))
        rows = get_json(url).get("data") or []
        if not rows:
            break
        before = len(out)
        for row in rows:
            try:
                out[int(row["fundingTime"])] = float(row["fundingRate"]) * 10_000.0
            except (KeyError, TypeError, ValueError):
                continue
        if len(out) == before:
            break
        page += 1
    return out


# -- Gate, live spread only --------------------------------------------------


def _gate_universe() -> List[Listing]:
    rows = get_json("https://api.gateio.ws/api/v4/futures/usdt/tickers")
    out = []
    for row in rows:
        contract = str(row.get("contract", ""))
        if not contract.endswith("_USDT"):
            continue
        try:
            volume = float(row.get("volume_24h_quote") or 0.0)
        except (TypeError, ValueError):
            continue
        out.append(Listing(contract, contract[:-5], volume))
    return out


def _gate_candles(symbol: str) -> List[Candle]:
    rows = get_json("https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
                    "?contract=" + symbol + "&interval=1d&limit=2000")
    out = []
    for row in rows:
        try:
            out.append((int(row["t"]) * 1000, float(row["o"]), float(row["h"]),
                        float(row["l"]), float(row["c"]),
                        float(row.get("sum") or 0.0), 0.0))
        except (KeyError, TypeError, ValueError):
            continue
    out.sort()
    return out


def _gate_funding(symbol: str) -> Dict[int, float]:
    rows = get_json("https://api.gateio.ws/api/v4/futures/usdt/funding_rate"
                    "?contract=" + symbol + "&limit=1000")
    return {int(row["t"]) * 1000: float(row["r"]) * 10_000.0 for row in rows}



# -- Kraken Futures ----------------------------------------------------------


def _kraken_universe() -> List[Listing]:
    """Kraken's perps are USD-margined (`PF_XBTUSD`), not USDT.

    That is a real difference from every other venue here and not just a naming
    one: a pair whose legs settle in different currencies carries the USDT/USD
    basis on top of the funding difference. It is small and it is not zero, and
    it belongs in the write-up rather than in a footnote.
    """
    rows = get_json("https://futures.kraken.com/derivatives/api/v3/tickers")
    out = []
    for row in rows.get("tickers", []):
        symbol = str(row.get("symbol", "")).upper()
        if not symbol.startswith("PF_") or not symbol.endswith("USD"):
            continue
        try:
            volume = float(row.get("volumeQuote") or 0.0)
        except (TypeError, ValueError):
            continue
        coin = symbol[3:-3]
        if coin == "XBT":                 # Kraken's name for BTC
            coin = "BTC"
        out.append(Listing(symbol, coin, volume))
    return out


def _kraken_candles(symbol: str) -> List[Candle]:
    now = int(time.time())
    url = ("https://futures.kraken.com/api/charts/v1/trade/" + symbol
           + "/1d?from=" + str(now - 5 * 365 * 86400) + "&to=" + str(now))
    rows = get_json(url).get("candles") or []
    out: List[Candle] = []
    for row in rows:
        try:
            close = float(row["close"])
            out.append((int(row["time"]), float(row["open"]), float(row["high"]),
                        float(row["low"]), close,
                        float(row["volume"]) * close, 0.0))
        except (KeyError, TypeError, ValueError):
            continue
    out.sort()
    return out


def _kraken_funding(symbol: str) -> Dict[int, float]:
    """`relativeFundingRate`, never `fundingRate`.

    Kraken publishes both: `fundingRate` is an ABSOLUTE amount in quote
    currency per contract, and `relativeFundingRate` is the fraction every
    other venue here reports. Reading the wrong one gives a number three or
    four orders of magnitude too large that still looks like a rate.
    """
    url = ("https://futures.kraken.com/derivatives/api/v4/historicalfundingrates"
           "?symbol=" + symbol)
    rows = get_json(url).get("rates") or []
    out: Dict[int, float] = {}
    for row in rows:
        try:
            stamp = str(row["timestamp"]).replace("Z", "+00:00")
            ts = int(datetime.fromisoformat(stamp).timestamp() * 1000)
            out[ts] = float(row["relativeFundingRate"]) * 10_000.0
        except (KeyError, TypeError, ValueError):
            continue
    return out


ADAPTERS: Dict[str, Adapter] = {
    "bybit": Adapter("Bybit", _bybit_universe, _bybit_candles, _bybit_funding,
                     note="deepest history; funding pages 200 at a time"),
    "mexc": Adapter("MEXC", _mexc_universe, _mexc_candles, _mexc_funding,
                    note="~18 months of funding, 17 pages of 100"),
    "bitget": Adapter("Bitget", _bitget_universe, _bitget_candles,
                      _bitget_funding, note="funding pages 100 at a time"),
    "kraken": Adapter("Kraken", _kraken_universe, _kraken_candles,
                      _kraken_funding,
                      note="CHEAPEST funding of eleven venues measured; hourly; "
                           "~367 days of history; USD-margined, so a pair "
                           "against a USDT venue carries the USDT/USD basis"),
    "gate": Adapter("Gate", _gate_universe, _gate_candles, _gate_funding,
                    panel_source=False,
                    note="returns only ~90 settlements however large the limit, "
                         "which is 30 days - fine for a live spread, too short "
                         "to backtest"),
}


# ---------------------------------------------------------------------------
# Assembly, shared by every venue
# ---------------------------------------------------------------------------


def funding_by_day(rates: Dict[int, float]) -> Dict[str, Tuple[float, int]]:
    """`{date: (bps accrued during that UTC day, settlements)}`.

    The one rule every panel in this repo shares: a settlement stamped T paid
    for the interval ENDING at T, so it belongs to the day before T, with the
    stamp rounded to its nominal minute first because settlements print
    milliseconds late. A venue that disagreed here would produce a funding
    difference that is a parsing artifact, and the pair trade IS a funding
    difference.
    """
    out: Dict[str, List[float]] = {}
    for ts, bps in rates.items():
        nominal = int(round(ts / 60_000.0)) * 60_000
        bucket = out.setdefault(_day_key(nominal - 1), [0.0, 0.0])
        bucket[0] += bps
        bucket[1] += 1
    return {day: (value[0], int(value[1])) for day, value in out.items()}


def settlement_interval_hours(rates: Dict[int, float]) -> Optional[float]:
    """Read the cadence off the timestamps, not off the documentation."""
    stamps = sorted(rates)
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if len(gaps) < 3:
        return None
    return statistics.median(gaps) / 3_600_000.0


def coin_rows(coin: str, candles: Sequence[Candle],
              funding: Dict[str, Tuple[float, int]],
              expected_per_day: int) -> List[Dict[str, object]]:
    """Panel rows in the shared schema.

    A day carrying fewer than most of its expected settlements is marked as
    having no funding rather than reported with a partial total, for the same
    reason the Hyperliquid panel does it: a partial day looks like unusually
    cheap funding and is really missing data.

    A candle with no volume AND no trades is a day the venue did not trade the
    coin - several of these list contracts with backfilled prices - so
    `minutes` is zeroed and the shared loader treats it as incomplete.
    """
    minimum = max(1, expected_per_day - 1)
    rows: List[Dict[str, object]] = []
    for ts, opn, high, low, close, quote, trades in candles:
        if not (close > 0 and high >= low > 0):
            continue
        date = _day_key(ts)
        bps, periods = funding.get(date, (float("nan"), 0))
        if periods < minimum:
            bps, periods = float("nan"), 0
        rows.append({
            "date": date, "symbol": coin,
            "open": opn, "high": high, "low": low, "close": close,
            "quote_volume": quote, "trades": trades,
            "taker_buy_frac": float("nan"),
            "minutes": 1440 if (quote > 0 or trades > 0) else 0,
            "rv_bps": math.log(high / low) * PARKINSON * 10_000.0,
            "funding_bps": bps, "funding_periods": periods,
        })
    return rows


def build(adapter: Adapter, *, top: int, min_volume: float, cache: Path,
          refresh: bool, pause: float) -> List[Dict[str, object]]:
    listings = [item for item in adapter.universe()
                if item.volume_usd_24h >= min_volume]
    listings.sort(key=lambda item: item.volume_usd_24h, reverse=True)
    listings = listings[:top]
    print("{}: {} contracts over ${:,.0f}/24h".format(
        adapter.name, len(listings), min_volume))

    directory = cache / adapter.name.lower()
    rows: List[Dict[str, object]] = []
    intervals: Dict[float, int] = {}
    for index, listing in enumerate(listings, start=1):
        path = directory / listing.symbol
        candles_path, funding_path = path / "candles.json", path / "funding.json"
        try:
            if not refresh and candles_path.exists():
                candles = [tuple(row) for row in json.loads(candles_path.read_text())]
            else:
                candles = adapter.candles(listing.symbol)
                path.mkdir(parents=True, exist_ok=True)
                candles_path.write_text(json.dumps(candles))
            if not refresh and funding_path.exists():
                rates = {int(k): float(v)
                         for k, v in json.loads(funding_path.read_text()).items()}
            else:
                rates = adapter.funding(listing.symbol)
                path.mkdir(parents=True, exist_ok=True)
                funding_path.write_text(
                    json.dumps({str(k): v for k, v in rates.items()}))
        except Exception as exc:                      # noqa: BLE001
            print("\n  " + listing.symbol + ": skipped (" + str(exc)[:70] + ")")
            continue

        hours = settlement_interval_hours(rates) or 8.0
        intervals[round(hours, 1)] = intervals.get(round(hours, 1), 0) + 1
        rows.extend(coin_rows(listing.coin, candles, funding_by_day(rates),
                              max(1, int(round(24.0 / hours)))))
        print("\r  {}/{}  {:<14} {:,} days, {:,} settlements at {:.0f}h    ".format(
            index, len(listings), listing.symbol, len(candles), len(rates), hours),
            end="", flush=True)
        if pause:
            time.sleep(pause)
    print()
    print("  settlement intervals: " + ", ".join(
        "{:g}h x{}".format(h, n) for h, n in sorted(intervals.items())))
    rows.sort(key=lambda row: (row["date"], row["symbol"]))
    return rows


def report(rows: Sequence[Dict[str, object]], adapter: Adapter) -> None:
    if not rows:
        print("No rows.")
        return
    dates = sorted({str(row["date"]) for row in rows})
    coins = sorted({str(row["symbol"]) for row in rows})
    funded = [row for row in rows if float(row["funding_periods"]) > 0]
    breadth: Dict[str, int] = {}
    for row in funded:
        breadth[str(row["date"])] = breadth.get(str(row["date"]), 0) + 1
    counts = sorted(breadth.values()) or [0]
    print()
    print("rows                {:,}".format(len(rows)))
    print("coins               {}".format(len(coins)))
    print("dates               {}  ({} .. {})".format(len(dates), dates[0], dates[-1]))
    print("days with funding   {:,}  ({:.1f}%)".format(
        len(funded), 100.0 * len(funded) / len(rows)))
    print("coins per day       min {}  median {}  max {}".format(
        counts[0], counts[len(counts) // 2], counts[-1]))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--venue", choices=sorted(ADAPTERS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--min-volume", type=float, default=2e6)
    parser.add_argument("--pause", type=float, default=0.15)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.list or not args.venue:
        print("{:<10}{:<14}{}".format("venue", "panel source", "note"))
        for key, adapter in sorted(ADAPTERS.items()):
            print("{:<10}{:<14}{}".format(
                key, "yes" if adapter.panel_source else "NO", adapter.note))
        return 0

    adapter = ADAPTERS[args.venue]
    if not adapter.panel_source:
        raise SystemExit(
            adapter.name + " is not a panel source: " + adapter.note
            + "\n  Use it for a live spread reading instead.")

    rows = build(adapter, top=args.top, min_volume=args.min_volume,
                 cache=args.cache, refresh=args.refresh, pause=args.pause)
    out = args.out or (PANEL / (args.venue + "-daily.csv"))
    write_panel(rows, out)
    report(rows, adapter)
    print("\nwrote " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
