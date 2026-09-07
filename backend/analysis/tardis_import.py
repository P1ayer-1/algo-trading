"""Test the pipeline on RECENT data, using Tardis.dev's free sample days.

    python backend\\analysis\\tardis_import.py --date 2026-09-01 --hours 2
    python backend\\analysis\\tardis_import.py --date 2026-08-01
    python backend\\analysis\\tardis_import.py --symbol ETHUSDT --date 2026-09-01

**The date must be the 1st of a month.** Tardis publishes the first day of
every month free, for every exchange and every data type, with no account and
no API key. Any other day returns 401 unless you set `--api-key` (or
`TARDIS_API_KEY`). That is the whole catch, and it is checked before anything
is downloaded.

Why this exists alongside binance_import.py
-------------------------------------------
Binance stopped publishing its `bookTicker` archive after 2024-03-30, so
`binance_import.py` can only ever run on data that is now years stale. Tardis
carries the same venue (binance-futures) to the present day, so this importer
is the one to reach for when the question is "does this work on *current*
market structure".

It is also a strictly better dataset in one important way:

    binance_import.py   bookTicker    top of book only, ~470 updates/sec
    tardis_import.py    book_snapshot_25  25 levels per side, ~27 updates/sec

So the two trade off against each other, and which one you want depends on
the question:

  * **25 levels of real depth.** `obi_5`, `obi_20`, `bid_depth_20` and
    `ask_depth_20` are all fully valid here. Under `binance_import.py` they
    are degraded placeholders — `obi_5` and `obi_20` are literally copies of
    `obi_1`. If you care about book shape beyond the touch, this is the only
    free source that has it.

  * **Coarser at the touch.** Binance's archived bookTicker fired on every
    change to the best bid/ask (~470/sec). Tardis snapshots come from the
    depth stream at ~27/sec (median gap 26ms). OFI is defined on the touch, so
    it sees less churn here — with 250ms recorder sampling that is still ~7
    book updates per row, which is workable, but it is genuinely less
    granular. If OFI specifically is what you are testing, run *both* and
    compare.

`funding_rate` is 0 in both importers; it is not in either dataset.

Then run the normal check against the result:

    python backend\\analysis\\check_features.py --data-dir data\\tardis --horizon 5

Runtime
-------
A full BTCUSDT day is ~2.4M book snapshots and ~3.6M trades. At ~6k events/sec
that is roughly 15-20 minutes. Start with `--hours 2` (~4 minutes).

The sign convention, because getting it wrong inverts every flow signal
----------------------------------------------------------------------
Tardis normalises `side` to the **aggressor** (the liquidity taker), which is
the same convention BloFin uses and the same thing `TradeTape` expects. So
unlike `binance_import.py` — which has to invert Binance's `is_buyer_maker` —
this importer passes `side` through unchanged. There are tests pinning both.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.importer_core import (  # noqa: E402
    Event,
    SchemaError,
    book_message,
    build_features,
    check_epoch_ms,
    download,
    ensure_clean_output,
    normalise_timestamp,
    print_summary,
    trade_message,
)

BASE_URL = "https://datasets.tardis.dev/v1"

# Tardis gives the 1st of every month away free, for every exchange and data
# type, with no account. Every other day needs a paid key. Verified against
# the live endpoint on 2026-09-07: 2026-09-01 returned 200, 2026-08-02 401.
FREE_SAMPLE_DAY = 1

# Tardis normalises every venue to these column names, so the parser is not
# Binance-specific — `--exchange bybit` or `bitget-futures` works unchanged.
TRADE_COLUMNS = ["timestamp", "side", "price", "amount"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _open_csv(path: Path):
    """Tardis ships gzipped CSV with a header row on every file."""
    return gzip.open(path, "rt", encoding="utf-8", newline="")


def _header_index(header: List[str], required: List[str], path: Path) -> Dict[str, int]:
    columns = {name.strip(): position for position, name in enumerate(header)}
    missing = [name for name in required if name not in columns]
    if missing:
        raise SchemaError(
            f"{path.name}: columns {missing} are missing from the header "
            f"{header[:12]}...\nTardis may have changed its format; update the "
            "column lists in tardis_import.py to match."
        )
    return columns


def _available_levels(columns: Dict[str, int]) -> int:
    """How many complete bid+ask levels this file actually carries."""
    levels = 0
    while all(f"{side}[{levels}].{field}" in columns
              for side in ("bids", "asks")
              for field in ("price", "amount")):
        levels += 1
    return levels


def book_events(path: Path, limit_ms: Optional[int], *, depth: int = 25) -> List[Event]:
    """Convert a Tardis book_snapshot_N file into book snapshot messages.

    Levels are ordered best-first and the trailing ones are blank whenever the
    real book is shallower than N, so parsing stops at the first empty price
    rather than scanning all N columns every row.
    """
    events: List[Event] = []
    first_ts: Optional[int] = None
    checked = 0

    with _open_csv(path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return events

        columns = _header_index(header, ["timestamp"], path)
        available = _available_levels(columns)
        if available == 0:
            raise SchemaError(
                f"{path.name}: no `bids[0].price` / `asks[0].price` columns "
                f"in header {header[:12]}...\nThis does not look like a Tardis "
                "book_snapshot file."
            )
        used = min(depth, available)
        ts_column = columns["timestamp"]
        bid_columns = [(columns[f"bids[{i}].price"], columns[f"bids[{i}].amount"])
                       for i in range(used)]
        ask_columns = [(columns[f"asks[{i}].price"], columns[f"asks[{i}].amount"])
                       for i in range(used)]

        for row in reader:
            try:
                ts = normalise_timestamp(int(row[ts_column]))
            except (ValueError, IndexError):
                continue

            bids: List[List[float]] = []
            for price_at, amount_at in bid_columns:
                price = row[price_at]
                if not price:
                    break
                bids.append([float(price), float(row[amount_at] or 0.0)])
            asks: List[List[float]] = []
            for price_at, amount_at in ask_columns:
                price = row[price_at]
                if not price:
                    break
                asks.append([float(price), float(row[amount_at] or 0.0)])

            if not bids or not asks:
                continue

            # Validate the first handful of rows hard: a hundred-column file
            # is an easy place for a mapping to be quietly wrong, and we want
            # to hear about it now, not after processing 6M rows.
            if checked < 50:
                checked += 1
                _validate_book_row(path, checked, bids, asks, ts)

            if first_ts is None:
                first_ts = ts
            if limit_ms is not None and ts - first_ts > limit_ms:
                break

            events.append((ts, len(events),
                           book_message(bids, asks, ts, len(events) + 1)))

    return events


def _validate_book_row(path: Path, row_number: int,
                       bids: List[List[float]], asks: List[List[float]],
                       ts: int) -> None:
    best_bid, best_ask = bids[0][0], asks[0][0]
    if best_bid <= 0 or best_ask <= 0:
        raise SchemaError(
            f"{path.name}: implausible prices on row {row_number} "
            f"(bid={best_bid}, ask={best_ask}). The column mapping is wrong."
        )
    if best_bid >= best_ask:
        raise SchemaError(
            f"{path.name}: bid {best_bid} >= ask {best_ask} on row "
            f"{row_number}. The bid and ask columns are likely swapped."
        )
    # Tardis orders levels best-first. If they arrive any other way the depth
    # features would silently aggregate the wrong levels.
    if any(a[0] <= b[0] for a, b in zip(bids, bids[1:])):
        raise SchemaError(
            f"{path.name}: bids are not in descending price order on row "
            f"{row_number}: {[level[0] for level in bids[:5]]}"
        )
    if any(a[0] >= b[0] for a, b in zip(asks, asks[1:])):
        raise SchemaError(
            f"{path.name}: asks are not in ascending price order on row "
            f"{row_number}: {[level[0] for level in asks[:5]]}"
        )
    check_epoch_ms(ts, f"{path.name} row {row_number}")


def trade_events(path: Path, limit_ms: Optional[int]) -> List[Event]:
    """Convert a Tardis trades file into trade messages.

    NO SIGN FLIP HERE, and that is deliberate. Tardis's `side` is already the
    aggressor — the liquidity taker — which is exactly what `TradeTape` and
    BloFin mean by `side`. `binance_import.py` inverts because Binance reports
    `is_buyer_maker` instead; this file must not.
    """
    events: List[Event] = []
    first_ts: Optional[int] = None
    checked = 0
    unparseable_sides = 0

    with _open_csv(path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return events

        columns = _header_index(header, TRADE_COLUMNS, path)
        ts_column = columns["timestamp"]
        side_column = columns["side"]
        price_column = columns["price"]
        amount_column = columns["amount"]

        for row in reader:
            try:
                price = float(row[price_column])
                size = float(row[amount_column])
                ts = normalise_timestamp(int(row[ts_column]))
            except (ValueError, IndexError):
                continue

            side = row[side_column].strip().lower()
            if side not in ("buy", "sell"):
                # Drop rather than guess a direction; a wrong side is worse
                # than a missing trade. But a file where *every* side is
                # unreadable is a schema problem, not bad luck.
                unparseable_sides += 1
                if checked < 50 and unparseable_sides > 50:
                    raise SchemaError(
                        f"{path.name}: the first 50 rows all have an "
                        f"unreadable `side` (last was {row[side_column]!r}). "
                        "Expected 'buy' or 'sell'."
                    )
                continue

            if checked < 50:
                checked += 1
                if price <= 0 or size < 0:
                    raise SchemaError(
                        f"{path.name}: implausible trade on row {checked} "
                        f"(price={price}, amount={size}). Column mapping wrong."
                    )
                check_epoch_ms(ts, f"{path.name} row {checked}")

            if first_ts is None:
                first_ts = ts
            if limit_ms is not None and ts - first_ts > limit_ms:
                break

            events.append((ts, len(events), trade_message(price, size, side, ts)))

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
    depth: int = 25,
) -> Dict[str, object]:
    limit_ms = int(hours * 3_600_000) if hours else None

    print("\nParsing book snapshots...")
    books = book_events(book_path, limit_ms, depth=depth)
    print(f"  {len(books):,} book updates")

    print("Parsing trades...")
    trades = trade_events(trade_path, limit_ms)
    print(f"  {len(trades):,} trades")

    return build_features(
        books, trades, out_dir,
        sample_ms=sample_ms, horizons=horizons, threshold_bps=threshold_bps,
    )


def dataset_url(exchange: str, data_type: str, date: str, symbol: str) -> str:
    year, month, day = date.split("-")
    return f"{BASE_URL}/{exchange}/{data_type}/{year}/{month}/{day}/{symbol}.csv.gz"


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Exchange-native symbol, e.g. BTCUSDT.")
    parser.add_argument("--exchange", default="binance-futures",
                        help="Tardis exchange id (binance-futures, bybit, "
                             "bitget-futures, okex-swap, ...).")
    parser.add_argument("--date", required=True,
                        help="YYYY-MM-DD (UTC). Must be the 1st without a key.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dir (default "
                             "data/tardis/<exchange>-<symbol>-<date>). One "
                             "directory per import, because the recorder names "
                             "files by today's date and appends.")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Where to keep downloads (default data/tardis/raw).")
    parser.add_argument("--hours", type=float, default=None,
                        help="Only convert the first N hours. Start with 2.")
    parser.add_argument("--depth", type=int, default=25, choices=(5, 25),
                        help="Book levels per side. 25 makes obi_20 real.")
    parser.add_argument("--sample-ms", type=int, default=250)
    parser.add_argument("--horizons", default="1,5,30")
    parser.add_argument("--threshold-bps", type=float, default=3.0)
    parser.add_argument("--api-key", default=os.environ.get("TARDIS_API_KEY"),
                        help="Paid Tardis key; lifts the 1st-of-month limit. "
                             "Defaults to $TARDIS_API_KEY.")
    parser.add_argument("--force", action="store_true", help="Re-download.")
    args = parser.parse_args(argv)

    run_name = f"{args.exchange}-{args.symbol}-{args.date}"
    out_dir = args.out or repo_root / "data" / "tardis" / run_name
    cache = args.cache or repo_root / "data" / "tardis" / "raw"
    horizons = tuple(float(part) for part in args.horizons.split(",") if part.strip())

    try:
        parsed_date = dt.date.fromisoformat(args.date)
    except ValueError:
        raise SystemExit(f"\n--date must be YYYY-MM-DD, got {args.date!r}")

    print(f"Tardis {args.exchange} {args.symbol} {args.date}")
    print("=" * 66)

    # Check before downloading, so a wrong date costs a second and not a
    # 90 MB transfer that ends in a 401.
    if parsed_date.day != FREE_SAMPLE_DAY and not args.api_key:
        first = parsed_date.replace(day=1).isoformat()
        raise SystemExit(
            f"\n{args.date} is not free. Tardis publishes only the 1st of each "
            "month\nwithout an account; every other day returns 401.\n\n"
            f"Use the 1st of that month:\n"
            f"  python backend\\analysis\\tardis_import.py --date {first} "
            "--hours 2\n\n"
            "Or set a paid key with --api-key / $TARDIS_API_KEY for any date."
        )
    if parsed_date > dt.date.today():
        raise SystemExit(f"\n{args.date} is in the future.")

    ensure_clean_output(out_dir)

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    unauthorised = (
        "Tardis only serves the 1st of each month for free. Either pick the "
        "1st,\nor pass a paid key with --api-key / $TARDIS_API_KEY. If you did "
        "pass a key,\nit may be expired or lack access to this exchange."
    )
    not_found = (
        f"{args.symbol} has no data for {args.date} on {args.exchange}.\n"
        "Check the symbol is spelled the way that venue spells it (BTCUSDT on "
        "binance-futures,\nBTCUSDT on bybit, BTC-USDT-SWAP on okex-swap), and "
        "that it was listed then."
    )

    book_type = f"book_snapshot_{args.depth}"
    book_url = dataset_url(args.exchange, book_type, args.date, args.symbol)
    trade_url = dataset_url(args.exchange, "trades", args.date, args.symbol)

    book_path = download(
        book_url, cache / f"{args.exchange}-{args.symbol}-{book_type}-{args.date}.csv.gz",
        force=args.force, headers=headers,
        not_found_hint=not_found, unauthorised_hint=unauthorised,
    )
    trade_path = download(
        trade_url, cache / f"{args.exchange}-{args.symbol}-trades-{args.date}.csv.gz",
        force=args.force, headers=headers,
        not_found_hint=not_found, unauthorised_hint=unauthorised,
    )

    stats = convert(
        book_path, trade_path, out_dir,
        hours=args.hours, sample_ms=args.sample_ms,
        horizons=horizons, threshold_bps=args.threshold_bps, depth=args.depth,
    )

    print_summary(stats, out_dir)

    print("\n" + "!" * 66)
    print(f"  DEPTH: {args.depth} levels per side - obi_1, obi_5, obi_20 and")
    print("  bid_depth_20/ask_depth_20 are all VALID here (unlike the Binance")
    print("  importer, where they are degraded copies of obi_1).")
    print("  funding_rate is still always 0 - not in this dataset.")
    print("  Book updates arrive at ~27/sec, vs ~470/sec for the old Binance")
    print("  bookTicker archive, so OFI sees less touch churn. See the module")
    print("  docstring.")
    print("!" * 66)

    print("\nNow run the check:")
    print(f"  python backend\\analysis\\check_features.py --data-dir {out_dir} "
          f"--horizon 5 --cost-bps 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
