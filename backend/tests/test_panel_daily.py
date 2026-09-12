"""The daily panel: the two alignments that would invent an edge if wrong.

Both failures this guards against look like skill rather than like a bug. A
funding settlement bucketed into the day AFTER the interval it paid for gives
every row tomorrow's carry today. And a day assembled from a handful of
surviving minutes produces a `close` that is not a close, so the return across
it is a return across an unknown span.
"""

from datetime import datetime, timezone

import pytest

from analysis.panel_daily import (
    MINUTES_PER_DAY,
    DayAccumulator,
    aggregate_symbol,
    build_panel,
    cached_symbols,
    fetch_funding,
    funding_by_day,
    kline_zips,
    load_or_build_daily,
    sources_signature,
)

DAY_MS = 86_400_000


def at(date_string, hour=0, millisecond=0):
    """Epoch ms for a UTC wall clock, so the tests read as wall clock."""
    base = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(base.timestamp() * 1000) + hour * 3_600_000 + millisecond


# ---------------------------------------------------------------------------
# Funding alignment
# ---------------------------------------------------------------------------


def test_funding_settles_into_the_day_whose_interval_it_paid_for():
    """A settlement stamped 00:00 on day D+1 covers 16:00..00:00 of day D.

    Bucketing it on its own date would hand day D+1 a rate that had not
    printed when day D+1's row opened, and hand it one day early for the whole
    five-year sample. The three settlements below are exactly one day's carry:
    3 + 2 + 1 = 6 bps, all of it attributed to 2026-03-04.
    """
    rates = {
        at("2026-03-04", 8): 3.0,
        at("2026-03-04", 16): 2.0,
        at("2026-03-05", 0): 1.0,
    }
    table = funding_by_day(rates)
    assert table["2026-03-04"] == (6.0, 3)
    assert "2026-03-05" not in table


def test_millisecond_jitter_does_not_move_a_settlement_across_midnight():
    """Measured on ADAUSDT: settlements print up to ~1.2s late.

    A stamp of 00:00:00.002 minus one millisecond is still 00:00:00.001 on the
    new day, which gave the day it belonged to 2 periods and the next 4. The
    stamp is rounded to its nominal minute first, so the jitter is removed
    exactly rather than absorbed by a guessed back-off.
    """
    rates = {
        at("2026-03-04", 8, 3): 3.0,
        at("2026-03-04", 16, 0): 2.0,
        at("2026-03-05", 0, 2): 1.0,       # two milliseconds late
    }
    assert funding_by_day(rates) == {"2026-03-04": (6.0, 3)}


def test_funding_day_boundary_is_exclusive_at_the_start():
    """00:00 of day D pays for day D-1, so day D's own row must not see it."""
    rates = {at("2026-03-04", 0): 9.0, at("2026-03-04", 8): 1.0}
    table = funding_by_day(rates)
    assert table["2026-03-03"] == (9.0, 1)
    assert table["2026-03-04"] == (1.0, 1)


def test_fetch_funding_pages_and_is_idempotent_on_overlap(tmp_path):
    """Re-fetching a page that overlaps the cache must not double-count.

    The endpoint pages forward from `startTime` and the cursor resumes from the
    newest cached stamp, so an overlapping page is normal. Rates are stored in
    a dict keyed by settlement time; a list would accumulate duplicates and
    every duplicate is a day of carry counted twice.
    """
    pages = {}
    first = [{"fundingTime": at("2026-03-04", h), "fundingRate": "0.0001"}
             for h in (0, 8, 16)]
    second = [{"fundingTime": at("2026-03-05", h), "fundingRate": "0.0002"}
              for h in (0, 8)]

    calls = []

    def opener(url):
        import json
        calls.append(url)
        start = int(url.split("startTime=")[1].split("&")[0])
        rows = [r for r in first + second if r["fundingTime"] >= start]
        return json.dumps(rows).encode()

    start_ms = at("2026-03-04")
    rates = fetch_funding("FAKEUSDT", start_ms=start_ms, cache=tmp_path,
                          opener=opener)
    assert len(rates) == 5
    assert rates[at("2026-03-04", 0)] == pytest.approx(1.0)
    assert rates[at("2026-03-05", 8)] == pytest.approx(2.0)

    again = fetch_funding("FAKEUSDT", start_ms=start_ms, cache=tmp_path,
                          opener=opener)
    assert len(again) == 5, "an overlapping refetch duplicated settlements"


def test_fetch_funding_refills_when_the_cache_starts_too_late(tmp_path):
    """A cache built from a later start must not silently truncate history.

    Resuming from `max(cached)` is right for extending forward and wrong when
    the caller now wants earlier history than the cache holds; the refill test
    is on `min(cached)`, not on the file's existence.
    """
    import json
    directory = tmp_path / "FAKEUSDT"
    directory.mkdir()
    (directory / "funding.json").write_text(
        json.dumps({str(at("2026-03-10", 0)): 5.0}))

    asked = []

    def opener(url):
        asked.append(int(url.split("startTime=")[1].split("&")[0]))
        return b"[]"

    fetch_funding("FAKEUSDT", start_ms=at("2026-03-01"), cache=tmp_path,
                  opener=opener)
    assert asked[0] == at("2026-03-01")


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------


def write_klines(directory, symbol, rows):
    """A Binance-shaped 1m archive: one headerless CSV inside one zip."""
    import csv
    import io
    import zipfile

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (symbol + "-1m-2026-03-04.zip")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(symbol + "-1m-2026-03-04.csv", buffer.getvalue())
    return path


def kline_row(close_time, open_, high, low, close, volume=1.0, quote=10.0,
              count=3.0, taker_base=0.5, taker_quote=6.0):
    """The eleven columns `KLINE_COLUMNS` names, in order."""
    return [close_time - 59_999, open_, high, low, close, volume, close_time,
            quote, count, taker_base, taker_quote]


def test_aggregate_folds_minutes_into_days_on_close_time(tmp_path):
    """Hand-computed: two minutes on 03-04, one on 03-05.

    open is the first minute's open, close the last minute's close, high and
    low the extremes, volume the sum, and the day key comes from `close_time`
    - a bar stamped 00:00:59.999 belongs to the day it closed in.
    """
    write_klines(tmp_path / "FAKEUSDT", "FAKEUSDT", [
        kline_row(at("2026-03-04", 0) + 59_999, 10.0, 11.0, 9.5, 10.5,
                  quote=100.0, count=5.0, taker_quote=60.0),
        kline_row(at("2026-03-04", 23) + 59_999, 10.5, 12.0, 10.0, 11.0,
                  quote=300.0, count=7.0, taker_quote=90.0),
        kline_row(at("2026-03-05", 0) + 59_999, 11.0, 11.2, 10.9, 11.1,
                  quote=50.0, count=2.0, taker_quote=25.0),
    ])
    rows = {row["date"]: row for row in aggregate_symbol("FAKEUSDT", tmp_path)}

    assert set(rows) == {"2026-03-04", "2026-03-05"}
    first = rows["2026-03-04"]
    assert first["open"] == 10.0
    assert first["close"] == 11.0
    assert first["high"] == 12.0
    assert first["low"] == 9.5
    assert first["quote_volume"] == 400.0
    assert first["trades"] == 12.0
    assert first["minutes"] == 2
    assert first["taker_buy_frac"] == pytest.approx(150.0 / 400.0)


def test_incomplete_days_are_reported_not_hidden(tmp_path):
    """A day with two of its 1440 minutes still produces a row.

    Dropping it would make the gap invisible and the next return would span an
    unknown number of days. `minutes` is what the harness filters on, so the
    aggregator's job is to report it honestly rather than to decide.
    """
    write_klines(tmp_path / "FAKEUSDT", "FAKEUSDT", [
        kline_row(at("2026-03-04", 0) + 59_999, 10.0, 10.0, 10.0, 10.0),
        kline_row(at("2026-03-04", 1) + 59_999, 10.0, 10.0, 10.0, 10.0),
    ])
    rows = aggregate_symbol("FAKEUSDT", tmp_path)
    assert len(rows) == 1
    assert rows[0]["minutes"] == 2 < MINUTES_PER_DAY


def test_overlapping_monthly_and_daily_archives_do_not_double_count(tmp_path):
    """The fetcher keeps monthlies and dailies that overlap at month edges.

    The same minute therefore arrives twice, and summing volume over the file
    set rather than over de-duplicated minutes would inflate every edge day.
    """
    import csv
    import io
    import zipfile

    directory = tmp_path / "FAKEUSDT"
    directory.mkdir(parents=True)
    row = kline_row(at("2026-03-04", 0) + 59_999, 10.0, 10.0, 10.0, 10.0,
                    quote=100.0)
    for name in ("FAKEUSDT-1m-2026-03.zip", "FAKEUSDT-1m-2026-03-04.zip"):
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerow(row)
        with zipfile.ZipFile(directory / name, "w") as archive:
            archive.writestr("x.csv", buffer.getvalue())

    rows = aggregate_symbol("FAKEUSDT", tmp_path)
    assert len(rows) == 1
    assert rows[0]["minutes"] == 1
    assert rows[0]["quote_volume"] == 100.0


def test_realised_vol_uses_within_day_steps_only(tmp_path):
    """The overnight step belongs to the daily return, not to intraday vol.

    Two minutes 1% apart inside the day give rv = |log(1.01)| * 1e4 = 99.5 bps.
    A third minute on the next day must not contribute a step to either day.
    """
    write_klines(tmp_path / "FAKEUSDT", "FAKEUSDT", [
        kline_row(at("2026-03-04", 0) + 59_999, 100.0, 100.0, 100.0, 100.0),
        kline_row(at("2026-03-04", 1) + 59_999, 100.0, 101.0, 100.0, 101.0),
        kline_row(at("2026-03-05", 0) + 59_999, 101.0, 200.0, 101.0, 200.0),
    ])
    rows = {row["date"]: row for row in aggregate_symbol("FAKEUSDT", tmp_path)}
    import math
    assert rows["2026-03-04"]["rv_bps"] == pytest.approx(
        abs(math.log(101.0 / 100.0)) * 10_000.0, rel=1e-9)
    assert rows["2026-03-05"]["rv_bps"] == 0.0, "an overnight gap leaked into rv"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_daily_cache_is_rebuilt_when_an_archive_grows(tmp_path):
    """A stale cache is worse than no cache: it silently answers about old data.

    The signature is names and sizes, so re-downloading an identical file keeps
    the cache while a file that gained rows invalidates it.
    """
    cache, panel = tmp_path / "cache", tmp_path / "panel"
    write_klines(cache / "FAKEUSDT", "FAKEUSDT", [
        kline_row(at("2026-03-04", 0) + 59_999, 10.0, 10.0, 10.0, 10.0),
    ])
    first = load_or_build_daily("FAKEUSDT", cache=cache, panel=panel)
    assert len(first) == 1

    signature = sources_signature("FAKEUSDT", cache)
    write_klines(cache / "FAKEUSDT", "FAKEUSDT", [
        kline_row(at("2026-03-04", 0) + 59_999, 10.0, 10.0, 10.0, 10.0),
        kline_row(at("2026-03-05", 0) + 59_999, 10.0, 10.0, 10.0, 10.0),
    ])
    assert sources_signature("FAKEUSDT", cache) != signature
    second = load_or_build_daily("FAKEUSDT", cache=cache, panel=panel)
    assert len(second) == 2


def test_cached_symbols_skips_non_ascii_and_empty_directories(tmp_path):
    """`fetch_klines` refuses CJK tickers because the archive path is the symbol
    verbatim; a directory that exists for one must not enter the panel either."""
    (tmp_path / "EMPTYUSDT").mkdir()
    (tmp_path / "牛来USDT").mkdir()
    write_klines(tmp_path / "FAKEUSDT", "FAKEUSDT", [
        kline_row(at("2026-03-04", 0) + 59_999, 10.0, 10.0, 10.0, 10.0),
    ])
    assert cached_symbols(tmp_path) == ["FAKEUSDT"]


def test_build_panel_joins_funding_onto_days(tmp_path):
    """The join is by date string, and a day with no settlement is NaN, not 0.

    Zero funding and unknown funding are different claims: one is a day the
    perp cost nothing to hold, the other is a day this file cannot speak for.
    """
    cache, panel = tmp_path / "cache", tmp_path / "panel"
    write_klines(cache / "FAKEUSDT", "FAKEUSDT", [
        kline_row(at("2026-03-04", 0) + 59_999, 10.0, 10.0, 10.0, 10.0),
        kline_row(at("2026-03-05", 0) + 59_999, 10.0, 10.0, 10.0, 10.0),
    ])
    rows = build_panel(["FAKEUSDT"], cache=cache, panel=panel,
                       with_funding=False, workers=1)
    assert [row["funding_periods"] for row in rows] == [0, 0]
    assert all(row["funding_bps"] != row["funding_bps"] for row in rows)
