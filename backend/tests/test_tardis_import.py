"""Tardis importer: parsing, schema validation, and the aggressor convention.

The most important test in this file is `test_side_is_the_aggressor_unchanged`.
Tardis's `side` is already the liquidity **taker**, which is what `TradeTape`
and BloFin mean by `side` — so this importer must pass it through untouched.
`binance_import.py` sits next to it and must do the exact opposite, because
Binance reports `is_buyer_maker`. One test file pins each direction; getting
either backwards would produce a model confidently predicting the wrong way.

The fixtures here are written by hand rather than downloaded, so the schema
assumptions are stated explicitly. The parser re-validates them at runtime and
raises `SchemaError` on the first few rows if a real file ever differs.
"""

import csv
import gzip
import io
from pathlib import Path

import pytest

from analysis.importer_core import SchemaError, normalise_timestamp
from analysis.tardis_import import (
    TRADE_COLUMNS,
    _available_levels,
    book_events,
    dataset_url,
    trade_events,
)

# Microseconds, as Tardis writes them.
BASE_US = 1_788_220_800_000_000


def write_csv_gz(path: Path, header, rows) -> Path:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    return path


def book_header(levels=25):
    """Tardis interleaves the sides: asks[0], bids[0], asks[1], bids[1], ..."""
    header = ["exchange", "symbol", "timestamp", "local_timestamp"]
    for index in range(levels):
        header += [f"asks[{index}].price", f"asks[{index}].amount",
                   f"bids[{index}].price", f"bids[{index}].amount"]
    return header


def book_row(index, levels=25, filled=None, mid=78500.0):
    """One snapshot: asks walk up from mid+0.1, bids walk down from mid-0.1."""
    filled = levels if filled is None else filled
    row = ["binance-futures", "BTCUSDT",
           BASE_US + index * 26_000, BASE_US + index * 26_000 + 500]
    for level in range(levels):
        if level < filled:
            row += [round(mid + 0.1 + level, 1), 1.0 + level,
                    round(mid - 0.1 - level, 1), 2.0 + level]
        else:
            row += ["", "", "", ""]
    return row


def make_book(path, count=60, levels=25, filled=None):
    return write_csv_gz(path, book_header(levels),
                        [book_row(i, levels, filled) for i in range(count)])


def trade_row(index, side):
    return ["binance-futures", "BTCUSDT",
            BASE_US + index * 26_000, BASE_US + index * 26_000 + 500,
            9_000_000 + index, side, 78500.0 + index, 0.5]


def make_trades(path, sides):
    return write_csv_gz(path, ["exchange", "symbol", "timestamp",
                               "local_timestamp", "id", "side", "price", "amount"],
                        [trade_row(i, side) for i, side in enumerate(sides)])


# ---------------------------------------------------------------------------
# The sign convention
# ---------------------------------------------------------------------------


def test_side_is_the_aggressor_unchanged(tmp_path):
    """Tardis `side` IS the aggressor. No inversion, unlike binance_import."""
    path = make_trades(tmp_path / "trades.csv.gz", ["buy", "sell", "buy", "sell"])
    events = trade_events(path, None)

    assert [event[2]["data"][0]["side"] for event in events] == \
        ["buy", "sell", "buy", "sell"]


def test_side_casing_and_padding_tolerated(tmp_path):
    path = make_trades(tmp_path / "trades.csv.gz", ["BUY", " sell ", "Buy"])
    events = trade_events(path, None)
    assert [event[2]["data"][0]["side"] for event in events] == ["buy", "sell", "buy"]


def test_unreadable_side_is_dropped_not_guessed(tmp_path):
    path = make_trades(tmp_path / "trades.csv.gz", ["buy", "???", "sell"])
    events = trade_events(path, None)
    assert [event[2]["data"][0]["side"] for event in events] == ["buy", "sell"]


def test_all_sides_unreadable_raises(tmp_path):
    path = make_trades(tmp_path / "trades.csv.gz", ["???"] * 80)
    with pytest.raises(SchemaError, match="unreadable `side`"):
        trade_events(path, None)


# ---------------------------------------------------------------------------
# Book parsing
# ---------------------------------------------------------------------------


def test_book_events_keep_all_levels(tmp_path):
    path = make_book(tmp_path / "book.csv.gz", count=3)
    events = book_events(path, None, depth=25)

    assert len(events) == 3
    data = events[0][2]["data"]
    assert len(data["bids"]) == 25
    assert len(data["asks"]) == 25
    # Best first, and the sides must not be swapped.
    assert data["bids"][0][0] == 78499.9
    assert data["asks"][0][0] == 78500.1
    assert data["bids"][0][0] < data["asks"][0][0]


def test_depth_argument_truncates(tmp_path):
    path = make_book(tmp_path / "book.csv.gz", count=2)
    events = book_events(path, None, depth=5)
    assert len(events[0][2]["data"]["bids"]) == 5


def test_short_book_stops_at_first_blank_level(tmp_path):
    """Real files leave the tail blank when the book is thinner than N."""
    path = make_book(tmp_path / "book.csv.gz", count=2, filled=3)
    events = book_events(path, None, depth=25)
    assert len(events[0][2]["data"]["bids"]) == 3
    assert len(events[0][2]["data"]["asks"]) == 3


def test_available_levels_counts_complete_pairs():
    columns = {name: i for i, name in enumerate(book_header(levels=5))}
    assert _available_levels(columns) == 5
    del columns["bids[3].amount"]
    assert _available_levels(columns) == 3


def test_timestamps_converted_to_milliseconds(tmp_path):
    path = make_book(tmp_path / "book.csv.gz", count=2)
    events = book_events(path, None)
    assert events[0][0] == BASE_US // 1000
    assert events[0][2]["data"]["ts"] == str(BASE_US // 1000)


def test_hours_limit_truncates_the_stream(tmp_path):
    # 26ms apart, so one minute of data is ~2,300 rows.
    path = make_book(tmp_path / "book.csv.gz", count=5000)
    events = book_events(path, limit_ms=60_000)
    assert 0 < len(events) < 5000
    assert events[-1][0] - events[0][0] <= 60_000


# ---------------------------------------------------------------------------
# Schema validation — a wrong mapping must fail loudly, not silently
# ---------------------------------------------------------------------------


def test_swapped_bid_and_ask_raises(tmp_path):
    header = book_header(levels=2)
    rows = [book_row(i, levels=2) for i in range(5)]
    for row in rows:  # swap the best bid and best ask prices
        row[4], row[6] = row[6], row[4]
    path = write_csv_gz(tmp_path / "book.csv.gz", header, rows)
    with pytest.raises(SchemaError, match="likely swapped"):
        book_events(path, None)


def test_unsorted_levels_raise(tmp_path):
    header = book_header(levels=3)
    rows = [book_row(i, levels=3) for i in range(5)]
    for row in rows:  # put bids in ascending order — wrong
        row[6], row[10] = row[10], row[6]
    path = write_csv_gz(tmp_path / "book.csv.gz", header, rows)
    with pytest.raises(SchemaError, match="descending price order"):
        book_events(path, None)


def test_missing_book_columns_raise(tmp_path):
    path = write_csv_gz(tmp_path / "book.csv.gz",
                        ["exchange", "symbol", "timestamp", "local_timestamp"],
                        [["binance-futures", "BTCUSDT", BASE_US, BASE_US]])
    with pytest.raises(SchemaError, match="book_snapshot"):
        book_events(path, None)


def test_missing_trade_columns_raise(tmp_path):
    path = write_csv_gz(tmp_path / "trades.csv.gz",
                        ["exchange", "symbol", "timestamp", "price"],
                        [["binance-futures", "BTCUSDT", BASE_US, 78500.0]])
    with pytest.raises(SchemaError, match="missing from the header"):
        trade_events(path, None)


def test_implausible_epoch_raises(tmp_path):
    header = book_header(levels=2)
    rows = [book_row(i, levels=2) for i in range(5)]
    for row in rows:
        row[2] = 12345  # seconds, not micro/milliseconds
    path = write_csv_gz(tmp_path / "book.csv.gz", header, rows)
    with pytest.raises(SchemaError, match="plausible millisecond epoch"):
        book_events(path, None)


def test_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "book.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("")
    assert book_events(path, None) == []


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_normalise_timestamp_handles_both_units():
    assert normalise_timestamp(1_788_220_800_000) == 1_788_220_800_000
    assert normalise_timestamp(1_788_220_800_000_000) == 1_788_220_800_000


def test_dataset_url_shape():
    assert dataset_url("binance-futures", "trades", "2026-09-01", "BTCUSDT") == (
        "https://datasets.tardis.dev/v1/binance-futures/trades/"
        "2026/09/01/BTCUSDT.csv.gz"
    )


def test_trade_columns_are_the_documented_ones():
    assert TRADE_COLUMNS == ["timestamp", "side", "price", "amount"]
