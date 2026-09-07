"""Raw archive: write, rotate, read back, and survive truncation.

The round-trip test is the important one. If replayed events don't reproduce
the live features exactly, then a model trained on replayed data is trained on
something the live bot never sees — a silent, very expensive discrepancy.
"""

import gzip
import json
from pathlib import Path

import pytest

from trading.features import FeatureEngine
from trading.orderbook import OrderBook
from trading.rawlog import RawEventLog, iter_directory, iter_events
from trading.tape import TradeTape


def message(seq, price=100.0):
    return {
        "arg": {"channel": "books", "instId": "BTC-USDT"},
        "action": "update",
        "data": {
            "bids": [[price - 0.5, 5]], "asks": [[price + 0.5, 5]],
            "ts": str(1_700_000_000_000 + seq * 100),
            "seqId": str(seq), "prevSeqId": str(seq - 1),
        },
    }


def test_writes_and_reads_back_identically(tmp_path):
    log = RawEventLog(tmp_path)
    originals = [message(seq) for seq in range(1, 21)]
    for item in originals:
        log.write("books", item)
    log.close()

    recovered = [m for _, _, m in iter_directory(tmp_path / "raw", "books")]
    assert recovered == originals


def test_receive_timestamp_is_stored(tmp_path):
    """Feed latency cannot be reconstructed later if it isn't captured now."""
    log = RawEventLog(tmp_path)
    log.write("books", message(1))
    log.close()
    (received_ms, _, _), = list(iter_directory(tmp_path / "raw", "books"))
    assert received_ms > 1_700_000_000_000


def test_only_configured_channels_are_archived(tmp_path):
    log = RawEventLog(tmp_path, channels={"books"})
    log.write("books", message(1))
    log.write("tickers", {"arg": {"channel": "tickers"}, "data": [{}]})
    log.close()
    assert log.lines_written == 1
    assert not list((tmp_path / "raw").rglob("tickers-*.jsonl.gz"))


def test_disabled_log_writes_nothing(tmp_path):
    log = RawEventLog(tmp_path, enabled=False)
    for seq in range(10):
        log.write("books", message(seq))
    log.close()
    assert log.lines_written == 0
    assert not (tmp_path / "raw").exists() or not list((tmp_path / "raw").rglob("*"))


def test_files_are_split_per_channel(tmp_path):
    log = RawEventLog(tmp_path)
    log.write("books", message(1))
    log.write("trades", {"arg": {"channel": "trades"}, "data": [{"price": "1"}]})
    log.close()
    names = sorted(p.name.split("-")[0] for p in (tmp_path / "raw").rglob("*.jsonl.gz"))
    assert names == ["books", "trades"]


def test_reopening_appends_rather_than_truncating(tmp_path):
    """A restart within the same hour must not destroy that hour's history."""
    first = RawEventLog(tmp_path)
    first.write("books", message(1))
    first.close()

    second = RawEventLog(tmp_path)
    second.write("books", message(2))
    second.close()

    assert len(list(iter_directory(tmp_path / "raw", "books"))) == 2


def test_truncated_final_line_is_tolerated(tmp_path):
    """Killing the process mid-write should cost one message, not the file."""
    log = RawEventLog(tmp_path)
    for seq in range(1, 6):
        log.write("books", message(seq))
    log.close()

    path = next((tmp_path / "raw").rglob("books-*.jsonl.gz"))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        content = handle.read()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content + '{"t":123,"m":{"incompl')  # torn write

    recovered = list(iter_events(path))
    assert len(recovered) == 5  # the five good ones, no exception


def test_compression_actually_helps(tmp_path):
    """Book messages are highly repetitive; if gzip weren't earning its keep
    the storage estimates in the README would be wrong."""
    log = RawEventLog(tmp_path)
    for seq in range(1, 2001):
        log.write("books", message(seq))
    log.close()

    on_disk = log.disk_bytes()
    assert on_disk < log.bytes_estimate / 4, (
        f"expected >4x compression, got {log.bytes_estimate / on_disk:.1f}x"
    )


# ---------------------------------------------------------------------------
# The round-trip that matters
# ---------------------------------------------------------------------------


def test_replayed_events_reproduce_live_features_exactly(tmp_path):
    """Replay must be bit-identical to the live path, or training data and
    production inputs silently diverge."""
    log = RawEventLog(tmp_path)

    live_book, live_tape, live_engine = OrderBook(), TradeTape(), FeatureEngine()
    live_snapshots = []

    events = [
        {
            "arg": {"channel": "books"}, "action": "snapshot",
            "data": {"bids": [[99.5, 5], [99.0, 8]], "asks": [[100.5, 5], [101.0, 8]],
                     "ts": "1700000000000", "seqId": "1", "prevSeqId": "0"},
        }
    ]
    for seq in range(2, 40):
        events.append({
            "arg": {"channel": "books"}, "action": "update",
            "data": {
                "bids": [[99.5, 5 + seq % 7]], "asks": [[100.5, 5 + seq % 5]],
                "ts": str(1_700_000_000_000 + seq * 100),
                "seqId": str(seq), "prevSeqId": str(seq - 1),
            },
        })
        if seq % 4 == 0:
            events.append({
                "arg": {"channel": "trades"},
                "data": [{"price": "100.0", "size": "1", "side": "buy",
                          "ts": str(1_700_000_000_000 + seq * 100)}],
            })

    for event in events:
        channel = event["arg"]["channel"]
        log.write(channel, event)
        if channel == "books":
            live_book.apply(event)
            if live_book.is_ready and not live_book.is_crossed():
                live_engine.on_book_event(live_book)
                live_snapshots.append(live_engine.compute(live_book, live_tape))
        else:
            live_tape.add_message(event["data"])
    log.close()

    # Now replay from disk with fresh state.
    replay_book, replay_tape, replay_engine = OrderBook(), TradeTape(), FeatureEngine()
    replay_snapshots = []
    # Sort on (receive_ms, sequence) — the timestamp alone is too coarse to
    # recover the true interleaving of books and trades.
    merged = sorted(
        list(iter_directory(tmp_path / "raw", "books"))
        + list(iter_directory(tmp_path / "raw", "trades")),
        key=lambda item: (item[0], item[1]),
    )
    for _, _, event in merged:
        channel = event["arg"]["channel"]
        if channel == "books":
            replay_book.apply(event)
            if replay_book.is_ready and not replay_book.is_crossed():
                replay_engine.on_book_event(replay_book)
                replay_snapshots.append(replay_engine.compute(replay_book, replay_tape))
        else:
            replay_tape.add_message(event["data"])

    assert len(replay_snapshots) == len(live_snapshots)
    for live, replayed in zip(live_snapshots, replay_snapshots):
        live_values = live.to_dict()
        replay_values = replayed.to_dict()
        # `received_ts` and anything derived from wall clock legitimately
        # differ between the two runs; everything computed from market data
        # must be identical.
        for key in live_values:
            if key in ("received_ts", "book_age_ms", "tape_staleness_s"):
                continue
            assert live_values[key] == replay_values[key], f"{key} diverged"
