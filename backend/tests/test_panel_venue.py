"""The generic venue panel: the rules that must not vary between exchanges.

The pair trade IS a funding difference between two venues, so anything the
panels disagree about becomes a signal. Cadence, day boundaries and what counts
as a traded day are enforced here once rather than per adapter, and these tests
are what stop a new adapter quietly inventing its own convention.
"""

import math
from datetime import datetime, timezone

import pytest

from analysis.panel_venue import (
    ADAPTERS,
    Adapter,
    Listing,
    build,
    coin_rows,
    funding_by_day,
    settlement_interval_hours,
)


def at(date_string, hour=0, millisecond=0):
    base = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(base.timestamp() * 1000) + hour * 3_600_000 + millisecond


def candle(date, *, close=100.0, high=101.0, low=99.0, quote=1000.0, trades=5.0):
    return (at(date), 100.0, high, low, close, quote, trades)


# ---------------------------------------------------------------------------
# The shared alignment rule
# ---------------------------------------------------------------------------


def test_funding_buckets_identically_to_every_other_panel():
    """A settlement stamped T paid for the interval ENDING at T.

    If one venue's panel disagreed by a single settlement, the difference
    between it and another venue would carry a constant offset of a third of a
    day's funding - and step 9r's whole result is a difference of a few bps a
    day.
    """
    rates = {at("2026-03-04", 8): 3.0, at("2026-03-04", 16): 2.0,
             at("2026-03-05", 0, 2): 1.0}
    assert funding_by_day(rates) == {"2026-03-04": (6.0, 3)}


def test_cadence_is_read_from_the_timestamps_not_from_documentation():
    """A venue that changed cadence last year documents the new one."""
    eight = {at("2026-03-04", hour): 1.0 for hour in (0, 8, 16)}
    eight[at("2026-03-05", 0)] = 1.0
    assert settlement_interval_hours(eight) == pytest.approx(8.0)

    four = {at("2026-03-04", hour): 1.0 for hour in range(0, 24, 4)}
    assert settlement_interval_hours(four) == pytest.approx(4.0)

    assert settlement_interval_hours({at("2026-03-04"): 1.0}) is None


def test_a_four_hourly_contract_needs_six_settlements_for_a_full_day():
    """The expected count comes from the measured cadence, so an 8h contract
    and a 4h one are both judged against their own full day rather than against
    a hard-coded three."""
    six = {"2026-03-04": (6.0, 6)}
    assert coin_rows("X", [candle("2026-03-04")], six, 6)[0]["funding_periods"] == 6

    partial = {"2026-03-04": (2.0, 2)}
    row = coin_rows("X", [candle("2026-03-04")], partial, 6)[0]
    assert row["funding_periods"] == 0
    assert row["funding_bps"] != row["funding_bps"]


def test_a_day_missing_one_settlement_is_still_counted():
    """Venues occasionally skip a settlement, and discarding the whole day for
    one missing period would throw away real carry. The tolerance is one."""
    almost = {"2026-03-04": (4.0, 2)}
    assert coin_rows("X", [candle("2026-03-04")], almost, 3)[0]["funding_periods"] == 2


# ---------------------------------------------------------------------------
# Traded days
# ---------------------------------------------------------------------------


def test_a_backfilled_pre_listing_candle_is_marked_untraded():
    """Several of these venues list contracts with backfilled prices and no
    volume. A return computed across one is a move that could not have been
    captured, so `minutes` is zeroed and the shared loader excludes it."""
    rows = coin_rows("X", [candle("2026-03-04", quote=0.0, trades=0.0)],
                     {"2026-03-04": (3.0, 3)}, 3)
    assert rows[0]["minutes"] == 0


def test_a_day_with_volume_but_no_trade_count_is_traded():
    """Most of these adapters cannot fill `trades` at all, so a rule that
    required it would mark every day untraded on those venues."""
    rows = coin_rows("X", [candle("2026-03-04", quote=500.0, trades=0.0)],
                     {"2026-03-04": (3.0, 3)}, 3)
    assert rows[0]["minutes"] == 1440


def test_rows_use_the_shared_schema_so_one_harness_reads_every_venue():
    from analysis.panel_daily import PANEL_COLUMNS

    rows = coin_rows("SUI", [candle("2026-03-04")], {"2026-03-04": (6.0, 3)}, 3)
    assert set(rows[0]) == set(PANEL_COLUMNS)
    assert rows[0]["symbol"] == "SUI"
    assert rows[0]["rv_bps"] == pytest.approx(
        math.log(101.0 / 99.0) / (2 * math.sqrt(math.log(2))) * 10_000.0)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def test_a_venue_too_shallow_to_backtest_is_refused_as_a_panel_source():
    """Gate returns ~90 settlements however large a limit it is given.

    Thirty days is fine for reading a live spread and far too short to
    backtest, and producing a panel that looks like the others would invite it
    to be used as one.
    """
    assert ADAPTERS["gate"].panel_source is False
    assert "30 days" in ADAPTERS["gate"].note
    assert all(ADAPTERS[name].panel_source
               for name in ("bybit", "mexc", "bitget"))


def test_build_filters_on_volume_and_takes_the_top_by_it(tmp_path):
    """The universe is ranked by dollar volume, and the same survivorship
    caveat applies as on every other panel: this selects what is big TODAY."""
    listings = [Listing("BIGUSDT", "BIG", 5e7),
                Listing("MIDUSDT", "MID", 5e6),
                Listing("TINYUSDT", "TINY", 1e3)]
    seen = []

    def candles(symbol):
        seen.append(symbol)
        return [candle("2026-03-04")]

    def funding(symbol):
        return {at("2026-03-04", hour): 1.0 for hour in (8, 16)} | {
            at("2026-03-05", 0): 1.0}

    adapter = Adapter("Fake", lambda: listings, candles, funding)
    rows = build(adapter, top=2, min_volume=1e6, cache=tmp_path,
                 refresh=True, pause=0)
    assert seen == ["BIGUSDT", "MIDUSDT"]
    assert {row["symbol"] for row in rows} == {"BIG", "MID"}
    assert rows[0]["funding_bps"] == pytest.approx(3.0)


def test_build_caches_so_a_rerun_costs_no_requests(tmp_path):
    calls = []

    def candles(symbol):
        calls.append(symbol)
        return [candle("2026-03-04")]

    def funding(symbol):
        return {at("2026-03-04", 8): 1.0}

    adapter = Adapter("Fake", lambda: [Listing("XUSDT", "X", 1e7)],
                      candles, funding)
    build(adapter, top=1, min_volume=0, cache=tmp_path, refresh=True, pause=0)
    build(adapter, top=1, min_volume=0, cache=tmp_path, refresh=False, pause=0)
    assert calls == ["XUSDT"]


def test_one_bad_symbol_does_not_abandon_the_rest(tmp_path):
    """A contract that 404s is a fact about the venue, not a reason to lose the
    other sixty-nine."""
    def candles(symbol):
        if symbol == "BADUSDT":
            raise RuntimeError("404")
        return [candle("2026-03-04")]

    adapter = Adapter("Fake", lambda: [Listing("BADUSDT", "BAD", 1e7),
                                       Listing("GOODUSDT", "GOOD", 9e6)],
                      candles, lambda symbol: {at("2026-03-04", 8): 1.0})
    rows = build(adapter, top=2, min_volume=0, cache=tmp_path, refresh=True,
                 pause=0)
    assert {row["symbol"] for row in rows} == {"GOOD"}
