"""Test the whole pipeline today, using free Binance historical data.

    python backend\\analysis\\binance_import.py --date 2024-03-01 --hours 2
    python backend\\analysis\\binance_import.py --date 2024-03-01
    python backend\\analysis\\binance_import.py --symbol ETHUSDT --date 2024-03-01

**The date must be between 2023-05-16 and 2024-03-30.** Binance discontinued
the bookTicker dataset after that; no free top-of-book feed exists for later
dates. See BOOK_TICKER_LAST_DATE below.

**Start with `--hours 2`.** A full BTCUSDT day is 20-40M book updates, and the
feature engine processes ~6k events/sec, so a whole day takes roughly an hour.
Two hours of data is ~30k rows — plenty to prove the pipeline works and to get
a first read on the features — and takes about five minutes.

The importer deliberately computes features on *every* event, exactly as the
live bot does, rather than only at the recorder's sample interval. Computing
less often here would be faster but would give imported data subtly different
label anchors than live data, and quietly divergent training inputs are a far
more expensive problem than a slow import.

Then run the normal check against the result:

    python backend\\analysis\\check_features.py --data-dir data\\binance --horizon 5

Why this exists
---------------
Waiting three days to find out whether the recorder, the feature engine and the
evaluation all work end-to-end is a poor feedback loop. Binance publishes
historical USDT-M futures data publicly, free, with no API key, at
https://data.binance.vision — so the pipeline can be exercised against real
market data in about ten minutes.

It downloads two datasets per day:

  bookTicker  every change to the best bid/ask, with sizes
  aggTrades   every trade, with the aggressor side

then feeds them through the *same* OrderBook, TradeTape, FeatureEngine and
FeatureRecorder the live bot uses, and writes a feature CSV in the identical
format. Nothing downstream needs to know the data came from somewhere else.

===========================================================================
WHAT THIS DOES AND DOES NOT TELL YOU
===========================================================================
It DOES answer:
  - Does the pipeline work end to end on real data?
  - Do OBI / OFI / trade-flow carry any predictive information at all?
  - Roughly how large is that information, and does it survive costs?

It does NOT answer:
  - Whether an edge exists *on BloFin*. Different venue, different
    participants, different fee schedule and tick size.

Direction of the bias is worth knowing: Binance BTCUSDT perp is one of the most
liquid and most heavily arbitraged instruments in existence. Any edge there is
competed down hard. A smaller venue like BloFin is generally *less* efficient,
so a signal that shows up on Binance is quite likely present on BloFin too —
while a signal that shows nothing on Binance is weak evidence either way.
Treat this as a pipeline test and a sanity check on the feature set, not as a
substitute for recording BloFin data.

===========================================================================
ONE IMPORTANT LIMITATION: bookTicker is TOP OF BOOK ONLY
===========================================================================
Binance's free feed gives best bid/ask, not full depth. So:

  works fully   obi_1, ofi_*, microprice, spread, ret_*, rv_*, tfi_*
  DEGRADED      obi_5 and obi_20 are identical to obi_1
                bid_depth_20 / ask_depth_20 are top-of-book size only

OFI is unaffected, which matters — it is defined purely on the touch, and it
is the feature with the strongest theoretical basis. The importer prints this
warning every run so a degraded column is never mistaken for a real one.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.importer_core import (  # noqa: E402
    Event,
    SchemaError,
    book_message,
    build_features,
    check_epoch_ms,
    download,
    ensure_clean_output,
    print_summary,
    trade_message,
)
from analysis.importer_core import normalise_timestamp as _normalise_timestamp  # noqa: E402

BASE_URL = "https://data.binance.vision/data/futures/um/daily"

# Binance STOPPED publishing bookTicker. The last daily file is 2024-03-30 (the
# last monthly is 2024-04). aggTrades, klines and bookDepth are still current,
# but there is no top-of-book feed after that date anywhere on the archive —
# spot never had bookTicker at all. So this importer can only ever run inside
# the window below. Verified against the bucket listing on 2026-09-07.
BOOK_TICKER_FIRST_DATE = "2023-05-16"
BOOK_TICKER_LAST_DATE = "2024-03-30"

# Expected columns. Binance added header rows to these files at different
# times, so the parser reads the header when present and falls back to these
# positions when absent — validating the result either way.
BOOK_TICKER_COLUMNS = [
    "update_id", "best_bid_price", "best_bid_qty",
    "best_ask_price", "best_ask_qty", "transaction_time", "event_time",
]
AGG_TRADE_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _looks_like_header(row: List[str]) -> bool:
    """A header row has a non-numeric first field."""
    if not row:
        return False
    try:
        float(row[0])
        return False
    except ValueError:
        return True


def _column_index(header: Optional[List[str]], expected: List[str]) -> Dict[str, int]:
    """Map column name -> index, from the header if present, else by position."""
    if header is None:
        return {name: index for index, name in enumerate(expected)}

    normalised = [column.strip().lower() for column in header]
    mapping = {}
    for name in expected:
        if name in normalised:
            mapping[name] = normalised.index(name)
    missing = [name for name in expected if name not in mapping]
    if missing:
        raise SchemaError(
            f"Columns {missing} are missing from the file header {normalised}.\n"
            "Binance may have changed the format; update the *_COLUMNS lists "
            "in binance_import.py to match."
        )
    return mapping


def read_zip_rows(path: Path, expected: List[str]) -> Iterator[Dict[str, str]]:
    """Yield dict rows from the single CSV inside a Binance daily zip."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise SchemaError(f"Expected exactly one CSV in {path.name}, found {names}")

        with archive.open(names[0]) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
            try:
                first = next(reader)
            except StopIteration:
                return

            if _looks_like_header(first):
                index = _column_index(first, expected)
            else:
                index = _column_index(None, expected)
                if len(first) < len(expected):
                    raise SchemaError(
                        f"{path.name}: expected at least {len(expected)} columns "
                        f"({expected}), got {len(first)}: {first}"
                    )
                yield {name: first[position] for name, position in index.items()}

            for row in reader:
                if len(row) < len(expected):
                    continue
                yield {name: row[position] for name, position in index.items()}


def book_events(path: Path, limit_ms: Optional[int]) -> List[Event]:
    """Convert bookTicker rows into book snapshot messages.

    Each row is emitted as a full `snapshot` with one level per side. That is
    honest about what the data contains — there is no depth to invent — and it
    sidesteps sequence handling entirely, since every row is self-contained.
    """
    events: List[Event] = []
    first_ts: Optional[int] = None
    checked = 0

    for row in read_zip_rows(path, BOOK_TICKER_COLUMNS):
        try:
            bid = float(row["best_bid_price"])
            bid_qty = float(row["best_bid_qty"])
            ask = float(row["best_ask_price"])
            ask_qty = float(row["best_ask_qty"])
            ts = _normalise_timestamp(int(row["transaction_time"]))
        except (ValueError, KeyError):
            continue

        # Validate the first handful of rows hard: if the columns are in a
        # different order than assumed, bid/ask will be nonsense and we want
        # to hear about it immediately, not after processing 20M rows.
        if checked < 50:
            checked += 1
            if bid <= 0 or ask <= 0 or bid_qty < 0 or ask_qty < 0:
                raise SchemaError(
                    f"{path.name}: implausible values on row {checked} "
                    f"(bid={bid}, ask={ask}, bid_qty={bid_qty}, ask_qty={ask_qty}). "
                    "The column mapping is probably wrong."
                )
            if bid >= ask:
                raise SchemaError(
                    f"{path.name}: bid {bid} >= ask {ask} on row {checked}. "
                    "best_bid_price and best_ask_price are likely swapped."
                )
            check_epoch_ms(ts, f"{path.name} row {checked}")

        if first_ts is None:
            first_ts = ts
        if limit_ms is not None and ts - first_ts > limit_ms:
            break

        events.append((ts, len(events), book_message(
            [[bid, bid_qty]], [[ask, ask_qty]], ts, len(events) + 1)))

    return events


def trade_events(path: Path, limit_ms: Optional[int]) -> List[Event]:
    """Convert aggTrades rows into trade messages.

    THE SIGN CONVENTION, because getting it wrong inverts every flow signal:

    Binance reports `is_buyer_maker`. When it is true, the *buyer* was the
    passive maker resting on the bid, so the trade was executed by an
    aggressive SELLER hitting that bid. BloFin's `side` field is the opposite
    convention — it names the aggressor directly.

        is_buyer_maker = true   ->  aggressor = sell
        is_buyer_maker = false  ->  aggressor = buy

    So the mapping inverts. `TradeTape` expects the aggressor, matching BloFin.
    """
    events: List[Event] = []
    first_ts: Optional[int] = None

    for row in read_zip_rows(path, AGG_TRADE_COLUMNS):
        try:
            price = float(row["price"])
            quantity = float(row["quantity"])
            ts = _normalise_timestamp(int(row["transact_time"]))
        except (ValueError, KeyError):
            continue

        raw_flag = str(row["is_buyer_maker"]).strip().lower()
        if raw_flag in ("true", "1"):
            buyer_was_maker = True
        elif raw_flag in ("false", "0"):
            buyer_was_maker = False
        else:
            continue  # unparseable flag: drop rather than guess a direction

        if first_ts is None:
            first_ts = ts
        if limit_ms is not None and ts - first_ts > limit_ms:
            break

        # The inversion described above.
        side = "sell" if buyer_was_maker else "buy"
        events.append((ts, len(events), trade_message(price, quantity, side, ts)))

    return events


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert(
    book_path: Path,
    trade_path: Path,
    out_dir: Path,
    *,
    hours: Optional[float],
    sample_ms: int,
    horizons: Tuple[float, ...],
    threshold_bps: float,
) -> Dict[str, object]:
    limit_ms = int(hours * 3_600_000) if hours else None

    print("\nParsing bookTicker...")
    books = book_events(book_path, limit_ms)
    print(f"  {len(books):,} book updates")

    print("Parsing aggTrades...")
    trades = trade_events(trade_path, limit_ms)
    print(f"  {len(trades):,} trades")

    return build_features(
        books, trades, out_dir,
        sample_ms=sample_ms, horizons=horizons, threshold_bps=threshold_bps,
    )


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Binance symbol, e.g. BTCUSDT (no dash).")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD (UTC).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dir (default data/binance/<symbol>-<date>). "
                             "One directory per import, because the recorder "
                             "names files by today's date and appends.")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Where to keep downloaded zips (default data/binance/zips).")
    parser.add_argument("--hours", type=float, default=None,
                        help="Only convert the first N hours. Start with 2 — a "
                             "full BTCUSDT day is tens of millions of updates.")
    parser.add_argument("--sample-ms", type=int, default=1000)
    parser.add_argument("--horizons", default="300,900,1800",
                        help="Forward label horizons in seconds. Minutes, "
                             "not seconds: see backend/config.py for why "
                             "second-scale horizons cannot clear fees.")
    parser.add_argument("--threshold-bps", type=float, default=10.0,
                        help="Move counted as up/down. Defaults to the "
                             "taker round-trip cost.")
    parser.add_argument("--force", action="store_true", help="Re-download.")
    args = parser.parse_args(argv)

    out_dir = args.out or repo_root / "data" / "binance" / f"{args.symbol}-{args.date}"
    cache = args.cache or repo_root / "data" / "binance" / "zips"
    horizons = tuple(float(part) for part in args.horizons.split(",") if part.strip())

    print(f"Binance {args.symbol} {args.date} (USDT-M futures)")
    print("=" * 66)

    # Fail before downloading a few hundred MB of aggTrades that can't be
    # paired with a book. See BOOK_TICKER_LAST_DATE above for why the window
    # is closed at both ends.
    if not BOOK_TICKER_FIRST_DATE <= args.date <= BOOK_TICKER_LAST_DATE:
        raise SystemExit(
            f"\n{args.date} is outside the range this importer can use.\n\n"
            f"Binance only published bookTicker between {BOOK_TICKER_FIRST_DATE} "
            f"and {BOOK_TICKER_LAST_DATE}.\n"
            "It was discontinued after that, and nothing else on "
            "data.binance.vision\ncarries top of book - aggTrades still runs to "
            "the present, but trades\nalone give no book, so OBI, OFI, spread and "
            "microprice cannot be computed.\n\n"
            "Pick a date inside the window:\n"
            "  python backend\\analysis\\binance_import.py --date 2024-03-01 "
            "--hours 2\n\n"
            "For recent data you need your own depth source: the BloFin recorder, "
            "or\na paid archive such as Tardis."
        )

    ensure_clean_output(out_dir)

    book_url = (f"{BASE_URL}/bookTicker/{args.symbol}/"
                f"{args.symbol}-bookTicker-{args.date}.zip")
    trade_url = (f"{BASE_URL}/aggTrades/{args.symbol}/"
                 f"{args.symbol}-aggTrades-{args.date}.zip")

    book_path = download(
        book_url, cache / Path(book_url).name, force=args.force,
        not_found_hint=(
            f"{args.symbol} has no bookTicker file for {args.date}. The dataset "
            f"only exists\nbetween {BOOK_TICKER_FIRST_DATE} and "
            f"{BOOK_TICKER_LAST_DATE}, and only for symbols listed at the time.\n"
            "Check the spelling (BTCUSDT, not BTC-USDT)."
        ),
    )
    trade_path = download(trade_url, cache / Path(trade_url).name, force=args.force)

    stats = convert(
        book_path, trade_path, out_dir,
        hours=args.hours, sample_ms=args.sample_ms,
        horizons=horizons, threshold_bps=args.threshold_bps,
    )

    print_summary(stats, out_dir)

    print("\n" + "!" * 66)
    print("  DEGRADED COLUMNS - bookTicker is top-of-book only:")
    print("    obi_5, obi_20            identical to obi_1 (no depth available)")
    print("    bid_depth_20/ask_depth_20  best-level size only")
    print("    funding_rate             always 0 (not in this dataset)")
    print("  Ignore those in the IC table. obi_1, ofi_*, tfi_*, microprice,")
    print("  spread, ret_* and rv_* are all fully valid.")
    print("!" * 66)

    print("\nNow run the check:")
    print(f"  python backend\\analysis\\check_features.py --data-dir {out_dir} "
          f"--horizon 5 --cost-bps 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
