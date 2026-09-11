"""Bulk fetching: which symbols, and which days actually arrived.

Coverage is the load-bearing part. A symbol listed in 2024 answers three years
of a five-year request with 404s, and without this the dataset silently
contains "whatever arrived" under a name that says five years.
"""

import json
import zipfile
from datetime import date

import pytest

from analysis.fetch_klines import (
    Coverage,
    coverage_of,
    covered_days,
    crypto_perpetuals,
    fetch_exchange_info,
    fetch_ticker_payload,
    rank_symbols,
    write_manifest,
)
from analysis.bars_import import daterange

PAYLOAD = [
    {"symbol": "BTCUSDT", "quoteVolume": "50000000000"},
    {"symbol": "ETHUSDT", "quoteVolume": "20000000000"},
    {"symbol": "TINYUSDT", "quoteVolume": "500000"},
    {"symbol": "ETHBTC", "quoteVolume": "9000000000"},
    {"symbol": "BTCUSDT_240329", "quoteVolume": "40000000000"},
    {"symbol": "SOLUSDT", "quoteVolume": "9000000000"},
]


def test_symbols_are_ranked_by_volume_busiest_first():
    assert rank_symbols(PAYLOAD, top=3) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_dated_futures_and_other_quotes_are_excluded():
    """A dated future's history stops when it expires, which is a fact about the
    contract rather than about the market."""
    ranked = rank_symbols(PAYLOAD, top=10)
    assert "BTCUSDT_240329" not in ranked
    assert "ETHBTC" not in ranked


def test_the_volume_floor_drops_thin_symbols():
    assert "TINYUSDT" not in rank_symbols(PAYLOAD, top=10, min_volume_usd=1e6)
    assert "TINYUSDT" in rank_symbols(PAYLOAD, top=10, min_volume_usd=0.0)


def test_a_ticker_payload_is_read_through_an_injected_opener():
    """No network in the tests, by the same rule as every other HTTP client here."""
    payload = fetch_ticker_payload(opener=lambda url: json.dumps(PAYLOAD).encode())
    assert payload[0]["symbol"] == "BTCUSDT"


def test_a_payload_that_is_not_a_list_is_refused():
    with pytest.raises(SystemExit):
        fetch_ticker_payload(opener=lambda url: b'{"code": -1}')


def test_a_monthly_file_covers_its_whole_month():
    days = daterange(date(2026, 3, 1), date(2026, 3, 31))
    names = ["BTCUSDT-1m-2026-03"]
    assert len(covered_days(names, "BTCUSDT", days, "1m")) == 31


def test_daily_files_cover_only_their_own_day():
    days = daterange(date(2026, 3, 1), date(2026, 3, 5))
    names = ["BTCUSDT-1m-2026-03-02", "BTCUSDT-1m-2026-03-04"]
    covered = covered_days(names, "BTCUSDT", days, "1m")
    assert covered == [date(2026, 3, 2), date(2026, 3, 4)]


def test_another_symbols_files_do_not_count(tmp_path):
    days = daterange(date(2026, 3, 1), date(2026, 3, 31))
    assert covered_days(["ETHUSDT-1m-2026-03"], "BTCUSDT", days, "1m") == []


def test_coverage_reads_the_cache_directory(tmp_path):
    directory = tmp_path / "BTCUSDT"
    directory.mkdir()
    for name in ("BTCUSDT-1m-2026-03", "BTCUSDT-1m-2026-04-01"):
        with zipfile.ZipFile(directory / f"{name}.zip", "w") as archive:
            archive.writestr(f"{name}.csv", "1,2,3\n")

    days = daterange(date(2026, 3, 1), date(2026, 4, 2))
    report = coverage_of(tmp_path, "BTCUSDT", days, "1m")
    assert report.files == 2
    assert report.days_wanted == 33
    assert report.days_covered == 32            # all of March plus April 1
    assert report.first_day == "2026-03-01" and report.last_day == "2026-04-01"
    assert report.share == pytest.approx(32 / 33)
    assert report.bytes_on_disk > 0


def test_a_symbol_with_no_directory_reports_nothing_covered(tmp_path):
    report = coverage_of(tmp_path, "NOPEUSDT", daterange(date(2026, 3, 1), date(2026, 3, 5)), "1m")
    assert report.files == 0 and report.days_covered == 0 and report.share == 0.0


def test_the_manifest_records_what_arrived(tmp_path):
    path = tmp_path / "manifest-1m.json"
    write_manifest(path, [Coverage("BTCUSDT", files=2, days_wanted=10, days_covered=9,
                                   first_day="2026-03-01", last_day="2026-03-09",
                                   bytes_on_disk=2_500_000)],
                   start=date(2026, 3, 1), end=date(2026, 3, 10), interval="1m")
    written = json.loads(path.read_text())
    assert written["interval"] == "1m"
    assert written["symbols"][0]["days_covered"] == 9
    assert written["symbols"][0]["megabytes"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Asset class
# ---------------------------------------------------------------------------

LISTINGS = [
    {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "underlyingType": "COIN",
     "status": "TRADING"},
    {"symbol": "XAUUSDT", "contractType": "TRADIFI_PERPETUAL",
     "underlyingType": "COMMODITY", "status": "TRADING"},
    {"symbol": "SOXLUSDT", "contractType": "TRADIFI_PERPETUAL",
     "underlyingType": "EQUITY", "status": "TRADING"},
    {"symbol": "SKHYNIXUSDT", "contractType": "TRADIFI_PERPETUAL",
     "underlyingType": "KR_EQUITY", "status": "TRADING"},
    {"symbol": "DEADUSDT", "contractType": "PERPETUAL", "underlyingType": "COIN",
     "status": "SETTLING"},
]


def test_only_trading_crypto_perpetuals_survive_the_asset_class_filter():
    """Gold, crude and single stocks trade in sessions, so their bars carry
    overnight and weekend gaps a crypto model would read as structure."""
    assert crypto_perpetuals(LISTINGS) == {"BTCUSDT"}


def test_ranking_can_be_restricted_to_an_allowed_universe():
    payload = PAYLOAD + [{"symbol": "XAUUSDT", "quoteVolume": "99000000000"}]
    assert rank_symbols(payload, top=3)[0] == "XAUUSDT"
    assert "XAUUSDT" not in rank_symbols(payload, top=3, allowed={"BTCUSDT", "ETHUSDT"})


def test_exchange_info_is_read_through_an_injected_opener():
    payload = json.dumps({"symbols": LISTINGS}).encode()
    assert fetch_exchange_info(opener=lambda url: payload)[0]["symbol"] == "BTCUSDT"


def test_exchange_info_without_symbols_is_refused():
    empty = json.dumps({"symbols": []}).encode()
    with pytest.raises(SystemExit):
        fetch_exchange_info(opener=lambda url: empty)
