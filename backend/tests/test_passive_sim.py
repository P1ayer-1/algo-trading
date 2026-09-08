"""Tests for the passive fill simulator.

Three things carry all the risk here, and every one of them is silent when
wrong.

The **aggressor convention**: a resting bid is filled by *sell* aggressors. Get
that backwards and every fill lands on the opposite side of every move, which
turns adverse selection into a profit and would read as a discovery rather
than a bug.

The **queue bracket**: the pessimistic rule must never fill earlier than the
optimistic one. If the bracket ever inverts, the uncertainty band the whole
report is built on is meaningless.

The **markout sign**: positive has to mean money for both sides.

Nothing here touches the network or reads a file.
"""

import numpy as np
import pytest

from analysis.passive_sim import (
    MODELS,
    ROUND_TRIP_MAKER_BPS,
    Market,
    Quotes,
    assign_buckets,
    bucket_edges,
    collect,
    markout_bps,
    markout_table,
    observable,
    report_economics,
    resolve_fill,
    simulate,
    touch_cancel_share,
)

BASE_TS = 1_700_000_000_000


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def trades(*rows):
    """rows of (ts_offset_ms, price, size, is_buy) -> the four parallel arrays."""
    ts = np.array([BASE_TS + row[0] for row in rows], dtype=np.int64)
    px = np.array([row[1] for row in rows], dtype=float)
    sz = np.array([row[2] for row in rows], dtype=float)
    buy = np.array([row[3] for row in rows], dtype=bool)
    return ts, px, sz, buy


def fill(rows, *, side, price, queue, start=0, deadline=10_000):
    ts, px, sz, buy = trades(*rows)
    return resolve_fill(ts, px, sz, buy, side=side, price=price, queue=queue,
                        start_ts=BASE_TS + start, deadline_ts=BASE_TS + deadline)


def book_message(bid, bid_size, ask, ask_size, ts):
    return {
        "arg": {"channel": "books"},
        "action": "snapshot",
        "data": {"bids": [[bid, bid_size]], "asks": [[ask, ask_size]],
                 "ts": str(ts), "seqId": "1", "prevSeqId": "0"},
    }


def trade_message(price, size, side, ts):
    return {
        "arg": {"channel": "trades"},
        "data": [{"price": str(price), "size": str(size), "side": side,
                  "ts": str(ts)}],
    }


# ---------------------------------------------------------------------------
# The aggressor convention
# ---------------------------------------------------------------------------


def test_a_resting_bid_is_filled_by_sell_aggressors_not_buys():
    # A buy aggressor lifts an ask. It can never reach a resting bid, however
    # much of it prints at that price.
    buys = [(100, 100.0, 999.0, True)]
    assert fill(buys, side="bid", price=100.0, queue=0.0) is None
    sells = [(100, 100.0, 1.0, False)]
    assert fill(sells, side="bid", price=100.0, queue=0.0) == 0


def test_a_resting_ask_is_filled_by_buy_aggressors_not_sells():
    sells = [(100, 100.0, 999.0, False)]
    assert fill(sells, side="ask", price=100.0, queue=0.0) is None
    buys = [(100, 100.0, 1.0, True)]
    assert fill(buys, side="ask", price=100.0, queue=0.0) == 0


def test_a_bid_does_not_fill_on_trades_above_its_price():
    # Sellers hitting a better bid than ours never reach our level.
    rows = [(100, 101.0, 50.0, False), (200, 100.5, 50.0, False)]
    assert fill(rows, side="bid", price=100.0, queue=0.0) is None


def test_an_ask_does_not_fill_on_trades_below_its_price():
    rows = [(100, 99.0, 50.0, True), (200, 99.5, 50.0, True)]
    assert fill(rows, side="ask", price=100.0, queue=0.0) is None


def test_a_trade_through_the_level_fills_it():
    # A sell printing BELOW our bid means everything resting at our price was
    # consumed on the way down, us included.
    rows = [(100, 99.0, 3.0, False)]
    assert fill(rows, side="bid", price=100.0, queue=0.0) == 0
    assert fill(rows, side="bid", price=100.0, queue=2.0) == 0


# ---------------------------------------------------------------------------
# The two bounds
# ---------------------------------------------------------------------------


def test_optimistic_fills_on_the_first_trade_at_the_price():
    rows = [(100, 100.0, 0.5, False), (200, 100.0, 0.5, False)]
    assert fill(rows, side="bid", price=100.0, queue=0.0) == 0


def test_pessimistic_waits_for_cumulative_volume_to_exceed_the_queue():
    # Q = 10. Three prints of 4 -> only the third takes the cumulative past it.
    rows = [(100, 100.0, 4.0, False),
            (200, 100.0, 4.0, False),
            (300, 100.0, 4.0, False)]
    assert fill(rows, side="bid", price=100.0, queue=10.0) == 2


def test_pessimistic_requires_strictly_exceeding_the_queue():
    # Exactly Q traded leaves us as the next order in line, not a fill.
    rows = [(100, 100.0, 10.0, False)]
    assert fill(rows, side="bid", price=100.0, queue=10.0) is None
    rows.append((200, 100.0, 0.001, False))
    assert fill(rows, side="bid", price=100.0, queue=10.0) == 1


def test_only_qualifying_volume_counts_toward_the_queue():
    # Buys at our price and sells above it must not eat our queue.
    rows = [(100, 100.0, 50.0, True),     # wrong aggressor
            (200, 101.0, 50.0, False),    # above our bid
            (300, 100.0, 4.0, False)]     # the only one that counts
    assert fill(rows, side="bid", price=100.0, queue=10.0) is None
    assert fill(rows, side="bid", price=100.0, queue=3.0) == 2


def test_the_bracket_never_inverts():
    # Whatever the tape, the pessimistic fill is at or after the optimistic
    # one. If this ever fails the reported uncertainty band is backwards.
    rng = np.random.default_rng(7)
    for _ in range(200):
        rows = [(int(offset), 100.0 - float(rng.integers(0, 3)),
                 float(rng.random()), bool(rng.integers(0, 2)))
                for offset in np.sort(rng.integers(1, 9_000, size=25))]
        optimistic = fill(rows, side="bid", price=100.0, queue=0.0)
        pessimistic = fill(rows, side="bid", price=100.0, queue=2.0)
        if pessimistic is not None:
            assert optimistic is not None
            assert optimistic <= pessimistic


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def test_trades_before_the_quote_are_not_counted():
    rows = [(50, 100.0, 99.0, False), (200, 100.0, 1.0, False)]
    assert fill(rows, side="bid", price=100.0, queue=10.0,
                start=100) is None


def test_a_trade_stamped_exactly_at_the_quote_time_does_not_fill_it():
    # The quote is placed after everything already carrying that millisecond.
    rows = [(100, 100.0, 5.0, False)]
    assert fill(rows, side="bid", price=100.0, queue=0.0, start=100) is None


def test_a_trade_after_the_deadline_does_not_fill():
    rows = [(9_999, 100.0, 5.0, False)]
    assert fill(rows, side="bid", price=100.0, queue=0.0, deadline=5_000) is None
    assert fill(rows, side="bid", price=100.0, queue=0.0, deadline=10_000) == 0


def test_an_empty_window_returns_no_fill():
    ts, px, sz, buy = trades((100, 100.0, 5.0, False))
    assert resolve_fill(ts, px, sz, buy, side="bid", price=100.0, queue=0.0,
                        start_ts=BASE_TS + 500, deadline_ts=BASE_TS + 600) is None


# ---------------------------------------------------------------------------
# Markout
# ---------------------------------------------------------------------------


def simple_market(mids, *, step_ms=1000, spread=0.0):
    """A market whose mid follows `mids`, one book update every `step_ms`."""
    n = len(mids)
    book_ts = np.array([BASE_TS + i * step_ms for i in range(n)], dtype=np.int64)
    mids = np.asarray(mids, dtype=float)
    return Market(
        book_ts=book_ts,
        bid_px=mids - spread / 2.0, bid_sz=np.ones(n),
        ask_px=mids + spread / 2.0, ask_sz=np.ones(n),
        trade_ts=book_ts, trade_px=mids, trade_sz=np.ones(n),
        trade_is_buy=np.zeros(n, dtype=bool),
    )


def test_a_bid_filled_before_a_rise_marks_out_positive():
    market = simple_market([100.0, 100.0, 100.1])
    out = markout_bps(market, np.array([BASE_TS]), np.array([100.0]), "bid", [0, 2])
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(10.0, rel=1e-3)


def test_an_ask_filled_before_a_rise_marks_out_negative():
    # Same move, opposite side: you sold just before it went up.
    market = simple_market([100.0, 100.0, 100.1])
    out = markout_bps(market, np.array([BASE_TS]), np.array([100.0]), "ask", [0, 2])
    assert out[0, 1] == pytest.approx(-10.0, rel=1e-3)


def test_markout_at_zero_is_the_half_spread():
    # Filled at the bid of a 10bps-wide book, with the mid unchanged.
    market = simple_market([100.0, 100.0, 100.0], spread=0.1)
    out = markout_bps(market, np.array([BASE_TS]), np.array([99.95]), "bid", [0])
    assert out[0, 0] == pytest.approx(5.0, rel=1e-3)


def test_mid_is_held_flat_between_updates_never_interpolated():
    market = simple_market([100.0, 200.0])
    # Halfway between the two updates the mid is still the earlier one.
    assert market.mid_at(np.array([BASE_TS + 500]))[0] == pytest.approx(100.0)
    # Before the first update it clamps rather than extrapolating.
    assert market.mid_at(np.array([BASE_TS - 5_000]))[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# The cancellation share
# ---------------------------------------------------------------------------


def test_a_queue_that_only_cancels_reads_as_all_cancellation():
    market = Market(
        book_ts=np.array([BASE_TS, BASE_TS + 100, BASE_TS + 200], dtype=np.int64),
        bid_px=np.array([100.0, 100.0, 100.0]),
        bid_sz=np.array([10.0, 7.0, 4.0]),
        ask_px=np.array([101.0, 101.0, 101.0]),
        ask_sz=np.array([10.0, 10.0, 10.0]),
        trade_ts=np.array([], dtype=np.int64),
        trade_px=np.array([]), trade_sz=np.array([]),
        trade_is_buy=np.array([], dtype=bool),
    )
    assert touch_cancel_share(market, "bid") == pytest.approx(1.0)


def test_a_queue_that_only_trades_reads_as_no_cancellation():
    market = Market(
        book_ts=np.array([BASE_TS, BASE_TS + 100, BASE_TS + 200], dtype=np.int64),
        bid_px=np.array([100.0, 100.0, 100.0]),
        bid_sz=np.array([10.0, 7.0, 4.0]),
        ask_px=np.array([101.0, 101.0, 101.0]),
        ask_sz=np.array([10.0, 10.0, 10.0]),
        trade_ts=np.array([BASE_TS + 50, BASE_TS + 150], dtype=np.int64),
        trade_px=np.array([100.0, 100.0]),
        trade_sz=np.array([3.0, 3.0]),
        trade_is_buy=np.array([False, False]),
    )
    assert touch_cancel_share(market, "bid") == pytest.approx(0.0)


def test_trades_on_the_wrong_side_are_not_credited_with_the_depletion():
    # Buy aggressors cannot consume the bid queue, so this depletion was
    # cancellation however much printed.
    market = Market(
        book_ts=np.array([BASE_TS, BASE_TS + 100], dtype=np.int64),
        bid_px=np.array([100.0, 100.0]), bid_sz=np.array([10.0, 4.0]),
        ask_px=np.array([101.0, 101.0]), ask_sz=np.array([10.0, 10.0]),
        trade_ts=np.array([BASE_TS + 50], dtype=np.int64),
        trade_px=np.array([100.0]), trade_sz=np.array([6.0]),
        trade_is_buy=np.array([True]),
    )
    assert touch_cancel_share(market, "bid") == pytest.approx(1.0)


def test_depletion_after_a_price_change_is_ignored():
    # The level moved, so the size difference is between two different queues
    # and says nothing about either.
    market = Market(
        book_ts=np.array([BASE_TS, BASE_TS + 100], dtype=np.int64),
        bid_px=np.array([100.0, 99.0]), bid_sz=np.array([10.0, 1.0]),
        ask_px=np.array([101.0, 101.0]), ask_sz=np.array([10.0, 10.0]),
        trade_ts=np.array([], dtype=np.int64), trade_px=np.array([]),
        trade_sz=np.array([]), trade_is_buy=np.array([], dtype=bool),
    )
    assert touch_cancel_share(market, "bid") == 0.0


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def synthetic_session(seconds=200, quote_every_ms=1000):
    """A book that ticks every 100ms with a steady sell tape into the bid."""
    events = []
    for step in range(seconds * 10):
        ts = BASE_TS + step * 100
        events.append(book_message(100.0, 5.0, 100.1, 5.0, ts))
        if step % 5 == 0:
            events.append(trade_message(100.0, 1.0, "sell", ts + 10))
    return events


def test_collect_places_quotes_at_the_touch_on_the_requested_grid():
    market, quotes = collect(synthetic_session(), quote_interval_ms=1000,
                             signals=("obi_1",))
    assert len(quotes) > 100
    # Warmup: nothing is quoted until 60s of mid history exists.
    assert quotes.ts[0] - BASE_TS >= 60_000
    assert np.allclose(quotes.price["bid"], 100.0)
    assert np.allclose(quotes.price["ask"], 100.1)
    # Q is the whole visible size at the level, because we join the back.
    assert np.allclose(quotes.queue["bid"], 5.0)
    gaps = np.diff(quotes.ts)
    assert gaps.min() >= 1000
    assert "obi_1" in quotes.signals


def test_the_bracket_orders_fill_rates_end_to_end():
    market, quotes = collect(synthetic_session(), quote_interval_ms=1000,
                             signals=())
    quotes = observable(quotes, market, timeout_ms=10_000, max_horizon_ms=5_000)
    fills = simulate(market, quotes, timeout_ms=10_000,
                     cancel_share={"bid": 0.5, "ask": 0.5})

    rates = {model: (fills[("bid", model)] >= 0).mean() for model in MODELS}
    # 1 contract every 500ms against a queue of 5 means the optimistic rule
    # fills at once and the pessimistic one needs ~2.5s. Both fit in 10s here,
    # so both fill — but never in the wrong order.
    assert rates["optimistic"] >= rates["cancel-adj"] >= rates["pessimistic"]
    assert rates["optimistic"] == pytest.approx(1.0)

    waits = {}
    for model in MODELS:
        indices = fills[("bid", model)]
        filled = indices >= 0
        waits[model] = np.median(market.trade_ts[indices[filled]] - quotes.ts[filled])
    assert waits["optimistic"] <= waits["cancel-adj"] <= waits["pessimistic"]


def test_the_ask_never_fills_when_the_tape_is_all_sells():
    market, quotes = collect(synthetic_session(), quote_interval_ms=1000,
                             signals=())
    quotes = observable(quotes, market, timeout_ms=10_000, max_horizon_ms=5_000)
    fills = simulate(market, quotes, timeout_ms=10_000,
                     cancel_share={"bid": 0.0, "ask": 0.0})
    assert (fills[("ask", "optimistic")] >= 0).sum() == 0


def test_markout_table_pools_both_sides_and_counts_every_fill():
    market, quotes = collect(synthetic_session(), quote_interval_ms=1000,
                             signals=())
    quotes = observable(quotes, market, timeout_ms=10_000, max_horizon_ms=5_000)
    fills = simulate(market, quotes, timeout_ms=10_000,
                     cancel_share={"bid": 0.0, "ask": 0.0})
    table = markout_table(market, quotes, fills, horizons_s=[0.0, 5.0])
    mean, error, count = table["optimistic"]
    # Only bids fill on this tape, and the book never moves, so the markout is
    # exactly the half spread at every horizon.
    assert count == int((fills[("bid", "optimistic")] >= 0).sum())
    assert mean[0] == pytest.approx(5.0, rel=1e-2)
    assert mean[1] == pytest.approx(5.0, rel=1e-2)
    assert np.isfinite(error).all()


def test_out_of_order_trade_timestamps_are_sorted_rather_than_trusted():
    # The raw archive merges by ARRIVAL time, so an exchange timestamp can
    # arrive late. searchsorted does not fail on an unsorted array — it
    # returns a wrong index — so the whole report would come out plausible
    # and incorrect.
    events = synthetic_session()
    events.append(trade_message(100.0, 1.0, "sell", BASE_TS + 500))
    market, _ = collect(events, quote_interval_ms=1000, signals=())
    assert (np.diff(market.trade_ts) >= 0).all()


def test_malformed_trades_are_skipped_the_way_the_tape_skips_them():
    # A crash partway through a six-hour parse is expensive, and a print the
    # tape rejected must not reach the fill simulation either.
    events = synthetic_session()
    events.append(trade_message(0.0, 1.0, "sell", BASE_TS + 1_000_000))
    events.append(trade_message(100.0, 0.0, "sell", BASE_TS + 1_000_001))
    events.append({"arg": {"channel": "trades"},
                   "data": [{"price": "100.0", "size": "1.0", "side": "sell"}]})
    market, _ = collect(events, quote_interval_ms=1000, signals=())
    assert (market.trade_px > 0).all()
    assert (market.trade_sz > 0).all()


# ---------------------------------------------------------------------------
# The lookahead guard
# ---------------------------------------------------------------------------


def test_quotes_without_an_observable_future_are_dropped():
    market, quotes = collect(synthetic_session(), quote_interval_ms=1000,
                             signals=())
    kept = observable(quotes, market, timeout_ms=10_000, max_horizon_ms=30_000)
    assert len(kept) < len(quotes)
    latest = market.book_ts[-1] - 40_000
    assert kept.ts.max() <= latest
    # Every horizon must be measured on the same quotes, or the curve is not a
    # curve. That is what this guard buys.
    assert len(kept) == int((quotes.ts <= latest).sum())


def test_too_few_observable_quotes_is_an_error_not_a_number():
    market, quotes = collect(synthetic_session(seconds=90),
                             quote_interval_ms=1000, signals=())
    with pytest.raises(SystemExit):
        observable(quotes, market, timeout_ms=10_000, max_horizon_ms=20_000)


# ---------------------------------------------------------------------------
# Signal bucketing
# ---------------------------------------------------------------------------


def test_bucket_edges_are_fitted_on_training_rows_and_applied_unchanged():
    train = np.arange(100, dtype=float)
    edges = bucket_edges(train, 5)
    # A test period that drifts must not silently re-centre the buckets — the
    # drift is exactly what the out-of-sample check is meant to expose.
    drifted = np.arange(500, 600, dtype=float)
    assert (assign_buckets(drifted, edges) == 4).all()


def test_bucket_assignment_covers_every_bucket_on_the_training_rows():
    values = np.linspace(-1.0, 1.0, 500)
    membership = assign_buckets(values, bucket_edges(values, 5))
    assert sorted(set(membership.tolist())) == [0, 1, 2, 3, 4]


def test_the_economics_gate_uses_the_whole_spread_not_the_half(capsys):
    # A passive round trip captures the spread on both legs and pays two maker
    # fees, so an instrument at exactly the round-trip fee is break-even. The
    # earlier version compared the HALF spread against the whole round trip,
    # which is too strict by 2x and would have failed this instrument.
    horizons = [0.0, 5.0]
    table = {model: (np.array([0.0, 0.0]), np.array([0.01, 0.01]), 100)
             for model in MODELS}
    fills = {(side, model): np.zeros(100, dtype=np.int64)
             for side in ("bid", "ask") for model in MODELS}
    report_economics(table, fills, horizons, 5.0, ROUND_TRIP_MAKER_BPS * 1.2)
    out = capsys.readouterr().out
    assert "does not cover the fee" not in out
    assert "NEVER COVERED THE FEE" not in out

    report_economics(table, fills, horizons, 5.0, ROUND_TRIP_MAKER_BPS * 0.8)
    assert "does not cover the fee" in capsys.readouterr().out


def test_a_mostly_tied_feature_collapses_to_fewer_buckets_not_empty_ones():
    # ret_5s on a book that did not move is exactly 0.0, and most quotes land
    # there. Repeated quantile edges would otherwise produce a [0.000, 0.000)
    # bucket holding nothing and reporting statistics on it.
    values = np.array([-1.0] * 10 + [0.0] * 80 + [1.0] * 10, dtype=float)
    edges = bucket_edges(values, 5)
    assert len(edges) == len(np.unique(edges))
    membership = assign_buckets(values, edges)
    for bucket in range(len(edges) + 1):
        assert (membership == bucket).any()
