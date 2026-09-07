"""Binance importer: parsing, schema validation, and the aggressor sign flip.

I could not fetch a real Binance file to verify against (this session's egress
policy blocks the host), so these tests pin the format assumptions explicitly
and the parser validates them at runtime. If Binance's layout differs from
what's assumed here, `SchemaError` fires on the first few rows with a clear
message rather than producing a plausible-looking, wrong dataset.

The most important test in this file is the aggressor one. Binance's
`is_buyer_maker` is the INVERSE of BloFin's `side`, and getting it backwards
would flip the sign of every trade-flow feature — a bug that produces a model
confidently predicting exactly the wrong direction.
"""

import csv
import io
import zipfile
from pathlib import Path

import pytest

from analysis.binance_import import (
    AGG_TRADE_COLUMNS,
    BOOK_TICKER_COLUMNS,
    SchemaError,
    _looks_like_header,
    _normalise_timestamp,
    book_events,
    convert,
    read_zip_rows,
    trade_events,
)

BASE_MS = 1_757_000_000_000


def make_zip(path: Path, rows, header=None):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if header:
        writer.writerow(header)
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(path.stem + ".csv", buffer.getvalue())
    return path


def book_rows(count=50, start=BASE_MS, step=100):
    """(update_id, bid, bid_qty, ask, ask_qty, transaction_time, event_time)"""
    rows = []
    for index in range(count):
        mid = 60000 + index
        rows.append([
            1000 + index, mid - 0.5, 2.0 + index % 3, mid + 0.5, 3.0 + index % 4,
            start + index * step, start + index * step + 2,
        ])
    return rows


def trade_rows(flags, start=BASE_MS, step=100):
    """(agg_id, price, qty, first_id, last_id, transact_time, is_buyer_maker)"""
    return [
        [500 + index, 60000 + index, 1.5, 10, 11, start + index * step, flag]
        for index, flag in enumerate(flags)
    ]


# ---------------------------------------------------------------------------
# THE sign convention
# ---------------------------------------------------------------------------


def test_is_buyer_maker_true_means_an_aggressive_SELL(tmp_path):
    """Binance: buyer was the maker => a seller crossed the spread to hit it."""
    path = make_zip(tmp_path / "t.zip", trade_rows(["true"]), AGG_TRADE_COLUMNS)
    (_, _, event), = trade_events(path, None)
    assert event["data"][0]["side"] == "sell"


def test_is_buyer_maker_false_means_an_aggressive_BUY(tmp_path):
    path = make_zip(tmp_path / "t.zip", trade_rows(["false"]), AGG_TRADE_COLUMNS)
    (_, _, event), = trade_events(path, None)
    assert event["data"][0]["side"] == "buy"


def test_aggressor_flip_propagates_correctly_to_trade_flow_imbalance(tmp_path):
    """End-to-end sign check through TradeTape.

    Three trades where the buyer was NOT the maker are three aggressive buys,
    so trade-flow imbalance must be +1, not -1.
    """
    from trading.tape import TradeTape

    path = make_zip(tmp_path / "t.zip",
                    trade_rows(["false", "false", "false"]), AGG_TRADE_COLUMNS)
    tape = TradeTape()
    for _, _, event in trade_events(path, None):
        tape.add_message(event["data"])
    assert tape.flow_imbalance(60.0) == pytest.approx(1.0)

    path2 = make_zip(tmp_path / "t2.zip",
                     trade_rows(["true", "true", "true"]), AGG_TRADE_COLUMNS)
    tape2 = TradeTape()
    for _, _, event in trade_events(path2, None):
        tape2.add_message(event["data"])
    assert tape2.flow_imbalance(60.0) == pytest.approx(-1.0)


def test_numeric_flags_are_accepted(tmp_path):
    """Some Binance files write 0/1 rather than false/true."""
    path = make_zip(tmp_path / "t.zip", trade_rows(["1", "0"]), AGG_TRADE_COLUMNS)
    events = trade_events(path, None)
    assert [e[2]["data"][0]["side"] for e in events] == ["sell", "buy"]


def test_unparseable_flag_is_dropped_not_guessed(tmp_path):
    """A trade whose direction we cannot determine must be discarded — an
    invented direction is worse than a missing trade."""
    path = make_zip(tmp_path / "t.zip", trade_rows(["maybe", "true"]),
                    AGG_TRADE_COLUMNS)
    assert len(trade_events(path, None)) == 1


# ---------------------------------------------------------------------------
# Header detection and column mapping
# ---------------------------------------------------------------------------


def test_header_and_headerless_files_parse_identically(tmp_path):
    rows = book_rows(10)
    with_header = make_zip(tmp_path / "a.zip", rows, BOOK_TICKER_COLUMNS)
    without = make_zip(tmp_path / "b.zip", rows)
    assert [e[2] for e in book_events(with_header, None)] == \
           [e[2] for e in book_events(without, None)]


def test_headerless_file_keeps_its_first_row(tmp_path):
    """The first data row must not be silently eaten as a header."""
    assert len(book_events(make_zip(tmp_path / "b.zip", book_rows(10)), None)) == 10


def test_columns_are_located_by_name_when_reordered(tmp_path):
    """If Binance reorders columns but keeps the names, follow the names."""
    reordered = ["transaction_time", "best_ask_price", "best_ask_qty",
                 "best_bid_price", "best_bid_qty", "update_id", "event_time"]
    rows = [[BASE_MS, 60000.5, 3.0, 59999.5, 2.0, 1, BASE_MS + 2]]
    path = make_zip(tmp_path / "b.zip", rows, reordered)
    (_, _, event), = book_events(path, None)
    assert event["data"]["bids"] == [[59999.5, 2.0]]
    assert event["data"]["asks"] == [[60000.5, 3.0]]


def test_looks_like_header_detection():
    assert _looks_like_header(["update_id", "best_bid_price"])
    assert not _looks_like_header(["1000", "59999.5"])
    assert not _looks_like_header([])


def test_missing_expected_column_raises_schema_error(tmp_path):
    path = make_zip(tmp_path / "b.zip", [[1, 2, 3]], ["update_id", "foo", "bar"])
    with pytest.raises(SchemaError, match="missing from the file header"):
        list(read_zip_rows(path, BOOK_TICKER_COLUMNS))


def test_zip_with_multiple_csvs_is_rejected(tmp_path):
    path = tmp_path / "b.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.csv", "1\n")
        archive.writestr("b.csv", "2\n")
    with pytest.raises(SchemaError, match="exactly one CSV"):
        list(read_zip_rows(path, BOOK_TICKER_COLUMNS))


# ---------------------------------------------------------------------------
# Validation that stops a wrong mapping from producing a plausible dataset
# ---------------------------------------------------------------------------


def test_swapped_bid_and_ask_are_detected(tmp_path):
    swapped = [[1, 60000.5, 3.0, 59999.5, 2.0, BASE_MS, BASE_MS]]
    path = make_zip(tmp_path / "b.zip", swapped, BOOK_TICKER_COLUMNS)
    with pytest.raises(SchemaError, match="likely swapped"):
        book_events(path, None)


def test_implausible_timestamp_is_detected(tmp_path):
    rows = [[1, 59999.5, 2.0, 60000.5, 3.0, 12345, 12345]]
    path = make_zip(tmp_path / "b.zip", rows, BOOK_TICKER_COLUMNS)
    with pytest.raises(SchemaError, match="plausible"):
        book_events(path, None)


def test_negative_price_is_detected(tmp_path):
    rows = [[1, -5.0, 2.0, 60000.5, 3.0, BASE_MS, BASE_MS]]
    path = make_zip(tmp_path / "b.zip", rows, BOOK_TICKER_COLUMNS)
    with pytest.raises(SchemaError, match="implausible values"):
        book_events(path, None)


def test_microsecond_timestamps_are_converted_to_milliseconds():
    assert _normalise_timestamp(1_757_000_000_000) == 1_757_000_000_000
    assert _normalise_timestamp(1_757_000_000_000_000) == 1_757_000_000_000


def test_microsecond_file_parses_without_tripping_validation(tmp_path):
    rows = [[1, 59999.5, 2.0, 60000.5, 3.0, BASE_MS * 1000, BASE_MS * 1000]]
    path = make_zip(tmp_path / "b.zip", rows, BOOK_TICKER_COLUMNS)
    (ts, _, _), = book_events(path, None)
    assert ts == BASE_MS


# ---------------------------------------------------------------------------
# Time limiting and end-to-end
# ---------------------------------------------------------------------------


def test_hours_limit_truncates_the_stream(tmp_path):
    # 100 rows, 1 second apart; keep the first 10 seconds.
    path = make_zip(tmp_path / "b.zip", book_rows(100, step=1000), BOOK_TICKER_COLUMNS)
    assert len(book_events(path, limit_ms=10_000)) == 11


def test_end_to_end_produces_a_csv_the_check_can_read(tmp_path):
    from analysis.check_features import build_matrix, load_rows

    # 20 minutes of 100ms book updates, with trades every 500ms.
    books = make_zip(tmp_path / "bookTicker.zip",
                     book_rows(12_000, step=100), BOOK_TICKER_COLUMNS)
    trades = make_zip(
        tmp_path / "aggTrades.zip",
        trade_rows(["true", "false"] * 1200, step=500),
        AGG_TRADE_COLUMNS,
    )

    out = tmp_path / "out"
    stats = convert(books, trades, out, hours=None, sample_ms=250,
                    horizons=(1.0, 5.0), threshold_bps=1.0)

    assert stats["rowsWritten"] > 100
    rows = load_rows(out)
    X, y, names, _ = build_matrix(rows, horizon=5.0)
    assert len(y) > 100
    assert "obi_1" in names and "ofi_5s" in names and "tfi_5s" in names


def test_top_of_book_only_means_obi_5_equals_obi_1(tmp_path):
    """Pins the documented limitation, so nobody later mistakes obi_20 for a
    real depth feature when the data came from bookTicker."""
    from trading.features import FeatureEngine
    from trading.orderbook import OrderBook
    from trading.tape import TradeTape

    path = make_zip(tmp_path / "b.zip", book_rows(30), BOOK_TICKER_COLUMNS)
    book, engine = OrderBook(), FeatureEngine()
    snapshot = None
    for _, _, message in book_events(path, None):
        book.apply(message)
        engine.on_book_event(book)
        snapshot = engine.compute(book, TradeTape())
    assert snapshot.obi_1 == snapshot.obi_5 == snapshot.obi_20
