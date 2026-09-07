"""Feed-level routing and desync recovery.

`MicrostructureFeed.handle_message` is pure — it takes a parsed message and
mutates state — so it is testable without a websocket, a network, or the SDK.
The class itself imports the SDK at module load, so these tests are skipped
when it isn't installed rather than failing.
"""

import pytest

pytest.importorskip("blofin", reason="BloFin SDK not installed")

from trading.ingest import MicrostructureFeed  # noqa: E402


def make_feed(tmp_path):
    return MicrostructureFeed("BTC-USDT", data_dir=tmp_path, record=False)


def books(bids, asks, seq, prev=None, ts=1_700_000_000_000, action="update"):
    data = {"bids": bids, "asks": asks, "ts": str(ts), "seqId": str(seq)}
    data["prevSeqId"] = "0" if prev is None else str(prev)
    return {"arg": {"channel": "books"}, "action": action, "data": data}


def seed(feed):
    feed.handle_message(
        books([[100.0, 5], [99.5, 5]], [[100.5, 5], [101.0, 5]], seq=1, action="snapshot")
    )


def test_snapshot_then_update_produces_features(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    feed.handle_message(books([[100.0, 9]], [], seq=2, prev=1, ts=1_700_000_000_100))
    assert feed.latest.is_valid
    assert feed.latest.mid == pytest.approx(100.25)


def test_sequence_gap_requests_resync(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    assert feed.handle_message(books([[100.0, 9]], [], seq=9, prev=8)) is True


def test_persistently_crossed_book_requests_resync(tmp_path):
    """Regression: a crossed book used to invalidate features forever with no
    recovery path, because nothing removes the stale level on its own."""
    feed = make_feed(tmp_path)
    seed(feed)
    # Put a bid above the best ask and never clear it.
    results = []
    for index in range(4):
        results.append(
            feed.handle_message(
                books([[102.0, 5]], [], seq=2 + index, prev=1 + index)
            )
        )
    assert feed.book.is_crossed() or not feed.book.ready
    assert results[-1] is True, "feed must eventually force a resync"
    assert results[0] is False, "a single crossed update should be tolerated"


def test_trades_route_to_the_tape(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    feed.handle_message(
        {
            "arg": {"channel": "trades"},
            "data": [{"price": "100.2", "size": "3", "side": "buy", "ts": "1700000000050"}],
        }
    )
    assert len(feed.tape.trades) == 1
    assert feed.latest.tfi_5s == pytest.approx(1.0)


def test_funding_rate_is_captured(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    feed.handle_message(
        {
            "arg": {"channel": "funding-rate"},
            "data": [{"fundingRate": "0.0002", "fundingTime": "1700000600000"}],
        }
    )
    assert feed.engine.funding_rate == pytest.approx(0.0002)


def test_unknown_channel_is_ignored(tmp_path):
    feed = make_feed(tmp_path)
    seed(feed)
    assert feed.handle_message({"arg": {"channel": "nonsense"}, "data": [{}]}) is False
