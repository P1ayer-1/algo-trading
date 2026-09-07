"""Test the whole pipeline today, using free Binance historical data.

    python backend\\analysis\\binance_import.py --date 2026-09-01 --hours 2
    python backend\\analysis\\binance_import.py --date 2026-09-01
    python backend\\analysis\\binance_import.py --symbol ETHUSDT --date 2026-09-01

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
historical USDⓈ-M futures data publicly, free, with no API key, at
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
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.features import FeatureEngine  # noqa: E402
from trading.orderbook import OrderBook  # noqa: E402
from trading.recorder import FeatureRecorder, LabelConfig  # noqa: E402
from trading.tape import TradeTape  # noqa: E402

BASE_URL = "https://data.binance.vision/data/futures/um/daily"

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


class SchemaError(RuntimeError):
    """Raised when a downloaded file doesn't look like what we expect.

    Failing loudly matters more than usual here: a silently mis-parsed column
    produces a feature file that looks perfectly normal and is entirely wrong.
    """


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download(url: str, destination: Path, *, force: bool = False) -> Path:
    """Fetch a file unless it's already on disk. Streams to a .part file so an
    interrupted download can never be mistaken for a complete one."""
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"  cached  {destination.name} "
              f"({destination.stat().st_size / 1e6:.1f} MB)")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"  fetching {url}")

    started = time.time()
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = downloaded / total * 100
                        print(f"\r    {downloaded/1e6:7.1f} / {total/1e6:.1f} MB "
                              f"({percent:5.1f}%)", end="", flush=True)
            print()
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        if exc.code == 404:
            raise SystemExit(
                f"\nNot found: {url}\n"
                "Binance publishes a day's file after that day closes (UTC), and\n"
                "only for symbols that existed then. Try an earlier date, or check\n"
                "the symbol spelling (BTCUSDT, not BTC-USDT)."
            ) from exc
        raise SystemExit(f"\nHTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"\nCould not reach {url}: {exc.reason}\n"
            "This needs plain internet access. If you are behind a proxy or VPN "
            "that blocks it, download the file manually in a browser and place "
            f"it at {destination}."
        ) from exc

    partial.replace(destination)
    print(f"  saved   {destination.name} "
          f"({destination.stat().st_size / 1e6:.1f} MB in {time.time() - started:.0f}s)")
    return destination


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


def _normalise_timestamp(value: int) -> int:
    """Return milliseconds.

    Binance has used both milliseconds and microseconds across datasets and
    eras. A millisecond timestamp for any plausible date is ~1.7e12; anything
    at ~1.7e15 is microseconds. Guessing wrong here would silently scale every
    horizon by 1000, so it is detected rather than assumed.
    """
    if value > 1e14:
        return value // 1000
    return value


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


def book_events(path: Path, limit_ms: Optional[int]) -> List[Tuple[int, int, dict]]:
    """Convert bookTicker rows into book snapshot messages.

    Each row is emitted as a full `snapshot` with one level per side. That is
    honest about what the data contains — there is no depth to invent — and it
    sidesteps sequence handling entirely, since every row is self-contained.
    """
    events: List[Tuple[int, int, dict]] = []
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
            if not (1_000_000_000_000 < ts < 4_000_000_000_000):
                raise SchemaError(
                    f"{path.name}: timestamp {ts} is not a plausible "
                    "millisecond epoch. Check the transaction_time column."
                )

        if first_ts is None:
            first_ts = ts
        if limit_ms is not None and ts - first_ts > limit_ms:
            break

        events.append((ts, len(events), {
            "arg": {"channel": "books", "instId": "BINANCE"},
            "action": "snapshot",
            "data": {
                "bids": [[bid, bid_qty]],
                "asks": [[ask, ask_qty]],
                "ts": str(ts),
                "seqId": str(len(events) + 1),
                "prevSeqId": "0",
            },
        }))

    return events


def trade_events(path: Path, limit_ms: Optional[int]) -> List[Tuple[int, int, dict]]:
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
    events: List[Tuple[int, int, dict]] = []
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

        events.append((ts, len(events), {
            "arg": {"channel": "trades", "instId": "BINANCE"},
            "data": [{
                "price": str(price),
                "size": str(quantity),
                # The inversion described above.
                "side": "sell" if buyer_was_maker else "buy",
                "ts": str(ts),
            }],
        }))

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

    if not books:
        raise SystemExit("No usable book updates were parsed.")

    # Interleave by (timestamp, per-stream order). Both streams are already
    # sorted, so this restores true event order across them.
    print("Merging and computing features...")
    merged = sorted(books + trades, key=lambda item: (item[0], item[1]))

    book, tape, engine = OrderBook(), TradeTape(), FeatureEngine()
    recorder = FeatureRecorder(
        out_dir,
        label_config=LabelConfig(horizons_seconds=horizons, threshold_bps=threshold_bps),
        sample_interval_ms=sample_ms,
    )

    processed = 0
    try:
        for _, _, message in merged:
            channel = message["arg"]["channel"]
            if channel == "books":
                book.apply(message)
                if book.is_ready and not book.is_crossed():
                    engine.on_book_event(book)
                    recorder.observe(engine.compute(book, tape))
            else:
                if tape.add_message(message["data"]):
                    recorder.observe(engine.compute(book, tape))
            processed += 1
            if processed % 500_000 == 0:
                print(f"  {processed:,} / {len(merged):,} events "
                      f"({recorder.rows_written:,} rows written)")
    finally:
        recorder.close()

    stats = dict(recorder.stats())
    stats["events"] = processed
    stats["books"] = len(books)
    stats["trades"] = len(trades)
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Binance symbol, e.g. BTCUSDT (no dash).")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD (UTC).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dir (default data/binance).")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Where to keep downloaded zips (default data/binance/zips).")
    parser.add_argument("--hours", type=float, default=None,
                        help="Only convert the first N hours. Start with 2 — a "
                             "full BTCUSDT day is tens of millions of updates.")
    parser.add_argument("--sample-ms", type=int, default=250)
    parser.add_argument("--horizons", default="1,5,30")
    parser.add_argument("--threshold-bps", type=float, default=3.0)
    parser.add_argument("--force", action="store_true", help="Re-download.")
    args = parser.parse_args(argv)

    out_dir = args.out or repo_root / "data" / "binance"
    cache = args.cache or out_dir / "zips"
    horizons = tuple(float(part) for part in args.horizons.split(",") if part.strip())

    print(f"Binance {args.symbol} {args.date} (USDⓈ-M futures)")
    print("=" * 66)

    book_url = (f"{BASE_URL}/bookTicker/{args.symbol}/"
                f"{args.symbol}-bookTicker-{args.date}.zip")
    trade_url = (f"{BASE_URL}/aggTrades/{args.symbol}/"
                 f"{args.symbol}-aggTrades-{args.date}.zip")

    book_path = download(book_url, cache / Path(book_url).name, force=args.force)
    trade_path = download(trade_url, cache / Path(trade_url).name, force=args.force)

    stats = convert(
        book_path, trade_path, out_dir,
        hours=args.hours, sample_ms=args.sample_ms,
        horizons=horizons, threshold_bps=args.threshold_bps,
    )

    print("\n" + "=" * 66)
    print(f"  events processed  {stats['events']:,}")
    print(f"  rows written      {stats['rowsWritten']:,}")
    print(f"  rows dropped      {stats['rowsDropped']:,} (no observable future)")
    print(f"  output            {out_dir}")

    print("\n" + "!" * 66)
    print("  DEGRADED COLUMNS — bookTicker is top-of-book only:")
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
