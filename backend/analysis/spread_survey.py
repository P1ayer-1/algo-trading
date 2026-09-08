"""Is there anything out there whose spread covers the fee? Rank symbols and venues.

    python backend\\analysis\\spread_survey.py --date 2026-09-01
    python backend\\analysis\\spread_survey.py --date 2026-09-01 --hours 2 ^
        --exchanges binance-futures,bybit,bitget-futures
    python backend\\analysis\\spread_survey.py --date 2026-09-01 ^
        --symbols SOLUSDT,DOGEUSDT,LINKUSDT --keep-downloads

`passive_sim.py` measured what a passive quote earns on Binance BTCUSDT perp
and the answer failed on arithmetic before adverse selection was involved: a
0.013 bps spread against a 1.2 bps maker round trip. A passive round trip
captures the whole spread and pays two maker fees, so the gate is simply

    median spread >= COST_MAKER_MAKER_BPS

and BTC perp misses it by roughly two orders of magnitude. That is a property
of the instrument, not a law, so this asks the obvious follow-up: **is there
anything that clears it?**

The gate itself lives in `passive_sim.clears_fee_gate`, and is imported rather
than restated, so the survey and the simulator can never disagree about what
passing means.

What it does NOT tell you
-------------------------
**A wide spread is not free money, and this tool cannot see the difference.**
Market makers widen precisely because the flow is more informed; spread and
adverse selection move together, so a symbol that clears the gate here has
only earned the right to be simulated, not the right to be traded. The output
is a shortlist for `passive_sim.py`, and the last line of the report says so.

There is a second trap the report guards against by printing `above gate`
alongside the median. A symbol whose spread is 0.9 bps on median but above 1.2
bps a third of the time is a *selective* quoting opportunity, and a symbol
pinned at 1.3 bps all day is a different, better one. The median alone cannot
separate them.

Cost
----
One `book_snapshot_5` file per symbol per venue, whole-day, from the Tardis
free sample (the 1st of any month). Roughly 15-40 MB each for a liquid perp,
so a ten-symbol survey is a few hundred MB. `--hours` limits the *parse*, not
the download — the archive has no partial-day files. Trades are not fetched at
all: the spread is a book property.

Downloads are cached under `data/tardis/raw` and reused by `passive_sim.py`,
so a symbol surveyed here is already on disk when you simulate it. Pass
`--keep-downloads` to say so explicitly, or `--discard-downloads` to delete
each file once measured, which keeps a wide survey to one file at a time.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.importer_core import download  # noqa: E402
from analysis.passive_sim import (  # noqa: E402
    MAKER_BPS,
    ROUND_TRIP_MAKER_BPS,
    clears_fee_gate,
)
from analysis.tardis_import import BASE_URL, FREE_SAMPLE_DAY, book_events  # noqa: E402

# The repo's existing cross-sectional universe, plus nothing. Majors are where
# the spread is tightest and therefore where the gate is hardest to clear —
# starting here means a pass is meaningful and a failure is expected.
DEFAULT_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
)

DEFAULT_EXCHANGES = ("binance-futures",)

# Five levels rather than 25: the gate is a top-of-book question, and the
# smaller dataset is a third of the download. `passive_sim.py --depth 5`
# reads the same file, so a surveyed symbol needs no second fetch.
SURVEY_DEPTH = 5


@dataclass
class SpreadStats:
    """One (exchange, symbol) measurement, or a reason there isn't one."""

    exchange: str
    symbol: str
    unavailable: Optional[str] = None

    samples: int = 0
    median_bps: float = float("nan")
    p25_bps: float = float("nan")
    p75_bps: float = float("nan")
    above_gate: float = float("nan")   # fraction of book updates clearing it
    median_touch: float = float("nan")  # contracts resting at the best bid
    mid: float = float("nan")

    @property
    def passes(self) -> bool:
        return bool(np.isfinite(self.median_bps) and clears_fee_gate(self.median_bps))


def spread_stats(exchange: str, symbol: str, path: Path,
                 limit_ms: Optional[int]) -> SpreadStats:
    """Measure the spread distribution in a Tardis book snapshot file.

    Reuses the importer's parser, so a schema change lands in one place and
    the survey can never disagree with the simulator about what the file said.
    """
    events = book_events(path, limit_ms, depth=SURVEY_DEPTH)
    if not events:
        return SpreadStats(exchange, symbol, unavailable="no usable book rows")

    spreads: List[float] = []
    touch: List[float] = []
    mids: List[float] = []
    for _, _, message in events:
        data = message["data"]
        bid_price, bid_size = data["bids"][0]
        ask_price, _ = data["asks"][0]
        mid = (bid_price + ask_price) / 2.0
        if mid <= 0 or ask_price <= bid_price:
            continue
        spreads.append((ask_price - bid_price) / mid * 1e4)
        touch.append(bid_size)
        mids.append(mid)

    if len(spreads) < 100:
        return SpreadStats(exchange, symbol,
                           unavailable=f"only {len(spreads)} valid rows")

    array = np.asarray(spreads)
    return SpreadStats(
        exchange=exchange,
        symbol=symbol,
        samples=len(array),
        median_bps=float(np.median(array)),
        p25_bps=float(np.percentile(array, 25)),
        p75_bps=float(np.percentile(array, 75)),
        above_gate=float(np.mean([clears_fee_gate(value) for value in array])),
        median_touch=float(statistics.median(touch)),
        mid=float(statistics.median(mids)),
    )


def dataset_url(exchange: str, symbol: str, date: str) -> str:
    year, month, day = date.split("-")
    return (f"{BASE_URL}/{exchange}/book_snapshot_{SURVEY_DEPTH}/"
            f"{year}/{month}/{day}/{symbol}.csv.gz")


def measure(exchange: str, symbol: str, date: str, cache: Path,
            limit_ms: Optional[int], headers: Optional[dict],
            discard: bool) -> SpreadStats:
    """Fetch and measure one pair, surviving anything that goes wrong with it.

    A survey that aborts on the first symbol a venue does not list is useless,
    and `download` raises SystemExit on a 404 or an unreachable host because
    that is right for a single-symbol importer. Here it is not, so it is
    caught and recorded — one missing symbol must cost a row, not the run.
    """
    destination = (cache /
                   f"{exchange}-{symbol}-book_snapshot_{SURVEY_DEPTH}-{date}.csv.gz")
    try:
        path = download(dataset_url(exchange, symbol, date), destination,
                        headers=headers)
    except SystemExit as exc:
        first = str(exc).strip().splitlines()
        return SpreadStats(exchange, symbol,
                           unavailable=first[0] if first else "download failed")

    try:
        result = spread_stats(exchange, symbol, path, limit_ms)
    except Exception as exc:  # a malformed file is a bad row, not a dead run
        result = SpreadStats(exchange, symbol,
                             unavailable=f"{type(exc).__name__}: {exc}")
    finally:
        if discard:
            destination.unlink(missing_ok=True)
    return result


def report(results: Sequence[SpreadStats], date: str, hours: Optional[float]) -> None:
    measured = [row for row in results if row.unavailable is None]
    missing = [row for row in results if row.unavailable is not None]

    print("\n" + "=" * 78)
    print(f"SPREAD SURVEY  {date}" + (f"  (first {hours:g}h)" if hours else ""))
    print("=" * 78)
    print(f"  gate: a passive round trip captures the whole spread and pays "
          f"{ROUND_TRIP_MAKER_BPS:.2f} bps")
    print(f"        ({MAKER_BPS:.2f} per leg), so an instrument needs a spread "
          f"at or above that.\n")

    if not measured:
        print("  Nothing was measured. Every symbol failed to download:")
        for row in missing:
            print(f"    {row.exchange:<18} {row.symbol:<10} {row.unavailable}")
        return

    print(f"  {'exchange':<18}{'symbol':<10}{'spread':>9}{'p25':>8}{'p75':>8}"
          f"{'above gate':>12}{'touch':>10}   verdict")
    print("  " + "-" * 74)
    for row in sorted(measured, key=lambda item: -item.median_bps):
        verdict = "CLEARS THE GATE" if row.passes else ""
        print(f"  {row.exchange:<18}{row.symbol:<10}{row.median_bps:>9.3f}"
              f"{row.p25_bps:>8.3f}{row.p75_bps:>8.3f}{row.above_gate:>11.1%}"
              f"{row.median_touch:>10.2f}   {verdict}")

    if missing:
        print("\n  not measured:")
        for row in missing:
            print(f"    {row.exchange:<18} {row.symbol:<10} {row.unavailable}")

    passing = [row for row in measured if row.passes]
    selective = [row for row in measured
                 if not row.passes and np.isfinite(row.above_gate)
                 and row.above_gate >= 0.10]

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    widest = max(measured, key=lambda item: item.median_bps)
    if passing:
        names = ", ".join(f"{row.exchange}/{row.symbol}" for row in passing)
        print(f"  {len(passing)} instrument(s) clear the fee gate on median "
              f"spread:\n    {names}")
        print("\n  That is a shortlist, not a result. A wide spread is not free "
              "money — makers\n  widen because the flow is more informed, so "
              "spread and adverse selection\n  move together. Simulate before "
              "believing:")
        for row in passing[:3]:
            print(f"    python backend\\analysis\\passive_sim.py --date {date} "
                  f"^\n        --exchange {row.exchange} --symbol {row.symbol} "
                  "--depth 5 --hours 6")
    elif selective:
        names = ", ".join(f"{row.exchange}/{row.symbol} ({row.above_gate:.0%})"
                          for row in selective)
        print("  Nothing clears the gate on median spread, but some clear it "
              "part of the\n  time:\n    " + names)
        print("\n  That is a *selective* quoting question, not a continuous "
              "one: quote only\n  while the spread is wide. It is a real "
              "strategy and a harder one, because\n  the spread is widest "
              "exactly when the flow is worst. Simulate the best of\n  them "
              "before designing anything around it.")
    else:
        print("  NOTHING CLEARS THE FEE GATE, AND NOTHING COMES CLOSE")
        print(f"  The widest instrument surveyed is {widest.exchange}/"
              f"{widest.symbol} at {widest.median_bps:.3f} bps,\n  which is "
              f"{ROUND_TRIP_MAKER_BPS / max(widest.median_bps, 1e-9):.0f}x "
              "short of the fee.")
        print("\n  Passive market making is not available at this fee schedule "
              "on this\n  universe. The two things that could still change it "
              "are a maker rebate\n  tier (which changes the gate itself) and a "
              "venue or symbol not surveyed\n  here — smaller venues and "
              "less-arbitraged symbols are where spread lives.\n  Otherwise "
              "the honest next branch is funding carry, which needs no spread\n"
              "  and no directional forecast.")


def main(argv: Optional[List[str]] = None) -> int:
    import datetime as dt
    import os

    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", required=True,
                        help="YYYY-MM-DD (UTC). Must be the 1st without a key.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--exchanges", default=",".join(DEFAULT_EXCHANGES),
                        help="Tardis venue ids (binance-futures, bybit, "
                             "bitget-futures, okex-swap, ...).")
    parser.add_argument("--hours", type=float, default=None,
                        help="Only parse the first N hours. Does not shrink "
                             "the download — the archive has no partial days.")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Where downloads live (default data/tardis/raw), "
                             "shared with passive_sim.py.")
    parser.add_argument("--discard-downloads", action="store_true",
                        help="Delete each file once measured. Keeps a wide "
                             "survey to one file on disk at a time.")
    parser.add_argument("--keep-downloads", action="store_true",
                        help="Explicitly keep them (the default).")
    parser.add_argument("--api-key", default=os.environ.get("TARDIS_API_KEY"),
                        help="Paid Tardis key; lifts the 1st-of-month limit.")
    args = parser.parse_args(argv)

    try:
        parsed_date = dt.date.fromisoformat(args.date)
    except ValueError:
        raise SystemExit(f"\n--date must be YYYY-MM-DD, got {args.date!r}")
    if parsed_date.day != FREE_SAMPLE_DAY and not args.api_key:
        first = parsed_date.replace(day=1).isoformat()
        raise SystemExit(
            f"\n{args.date} is not free. Tardis publishes only the 1st of each "
            "month\nwithout an account; every other day returns 401.\n\n"
            f"  python backend\\analysis\\spread_survey.py --date {first}")
    if args.discard_downloads and args.keep_downloads:
        raise SystemExit("\nPass one of --discard-downloads / --keep-downloads.")

    cache = args.cache or repo_root / "data" / "tardis" / "raw"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    limit_ms = int(args.hours * 3_600_000) if args.hours else None
    symbols = [part.strip() for part in args.symbols.split(",") if part.strip()]
    exchanges = [part.strip() for part in args.exchanges.split(",") if part.strip()]

    pairs: List[Tuple[str, str]] = [(exchange, symbol)
                                    for exchange in exchanges
                                    for symbol in symbols]
    print(f"Spread survey - {len(pairs)} instrument(s) on {args.date}")
    print("=" * 78)
    print("Downloading book_snapshot_5 per instrument; no trade files needed.")

    results: List[SpreadStats] = []
    for index, (exchange, symbol) in enumerate(pairs, start=1):
        print(f"\n[{index}/{len(pairs)}] {exchange} {symbol}")
        row = measure(exchange, symbol, args.date, cache, limit_ms, headers,
                      discard=args.discard_downloads)
        if row.unavailable:
            print(f"  skipped: {row.unavailable}")
        else:
            print(f"  median spread {row.median_bps:.3f} bps over "
                  f"{row.samples:,} book updates"
                  + ("   CLEARS THE GATE" if row.passes else ""))
        results.append(row)

    report(results, args.date, args.hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
