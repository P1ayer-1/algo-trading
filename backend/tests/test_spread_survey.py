"""Tests for the spread survey and the fee gate it shares with the simulator.

The gate is one line of arithmetic that decides whether a whole branch of the
project is worth pursuing, so it is pinned hard and in one place. The thing it
must not do is compare one leg's spread capture against two legs' fees, which
is too strict by a factor of two and would reject exactly the instruments the
survey exists to find.

The survey's other job is to survive a bad instrument. A ten-symbol run that
aborts because one venue does not list one of them is useless, and the
importer's `download` raises SystemExit on a 404 by design.

Nothing here touches the network.
"""

import gzip

import numpy as np
import pytest

from analysis import spread_survey
from analysis.passive_sim import ROUND_TRIP_MAKER_BPS, clears_fee_gate
from analysis.spread_survey import SpreadStats, measure, spread_stats


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_gate_compares_the_whole_spread_against_the_whole_round_trip():
    # A passive round trip captures half the spread on the way in and half on
    # the way out, so an instrument whose spread exactly equals the round-trip
    # fee is exactly break-even, not a factor of two short.
    assert clears_fee_gate(ROUND_TRIP_MAKER_BPS)
    assert clears_fee_gate(ROUND_TRIP_MAKER_BPS + 1e-9)
    assert not clears_fee_gate(ROUND_TRIP_MAKER_BPS - 1e-9)


def test_the_gate_does_not_use_the_half_spread():
    # The bug this replaces: `half_spread >= round_trip` would reject an
    # instrument at 1.5x the fee, which is a tradeable spread.
    spread = ROUND_TRIP_MAKER_BPS * 1.5
    assert clears_fee_gate(spread)
    assert not clears_fee_gate(spread / 2.0)


def test_a_one_tick_btc_spread_fails_by_two_orders_of_magnitude():
    assert not clears_fee_gate(0.013)


# ---------------------------------------------------------------------------
# Measuring a file
# ---------------------------------------------------------------------------


def write_book_file(path, rows, levels=5):
    """A Tardis book_snapshot_N file: `rows` of (ts, bid, bid_size, ask)."""
    header = ["exchange", "symbol", "timestamp", "local_timestamp"]
    for side in ("asks", "bids"):
        for level in range(levels):
            header += [f"{side}[{level}].price", f"{side}[{level}].amount"]
    lines = [",".join(header)]
    for ts, bid, bid_size, ask in rows:
        cells = ["binance-futures", "TESTUSDT", str(ts), str(ts)]
        for level in range(levels):
            cells += [f"{ask + level}", "1.0"]
        for level in range(levels):
            cells += [f"{bid - level}", f"{bid_size if level == 0 else 1.0}"]
        lines.append(",".join(cells))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def test_spread_is_measured_in_bps_against_the_mid(tmp_path):
    # bid 999.5 / ask 1000.5 -> 1.0 wide on a 1000 mid -> 10 bps.
    rows = [(1_700_000_000_000 + i * 100, 999.5, 3.0, 1000.5) for i in range(200)]
    path = write_book_file(tmp_path / "book.csv.gz", rows)
    stats = spread_stats("binance-futures", "TESTUSDT", path, None)
    assert stats.unavailable is None
    assert stats.median_bps == pytest.approx(10.0, rel=1e-3)
    assert stats.median_touch == pytest.approx(3.0)
    assert stats.mid == pytest.approx(1000.0)
    assert stats.passes


def test_above_gate_counts_the_fraction_of_time_not_the_average(tmp_path):
    # Half the session pinned wide, half pinned tight. The mean spread would
    # clear the gate; the point of `above gate` is that only half the session
    # actually does, which is a selective quoting question, not a continuous
    # one.
    wide = [(1_700_000_000_000 + i * 100, 999.0, 1.0, 1001.0) for i in range(100)]
    tight = [(1_700_000_000_000 + (100 + i) * 100, 999.999, 1.0, 1000.001)
             for i in range(100)]
    path = write_book_file(tmp_path / "book.csv.gz", wide + tight)
    stats = spread_stats("binance-futures", "TESTUSDT", path, None)
    assert stats.above_gate == pytest.approx(0.5, abs=0.01)


def test_a_file_with_too_few_rows_is_reported_not_averaged(tmp_path):
    rows = [(1_700_000_000_000 + i * 100, 999.5, 1.0, 1000.5) for i in range(10)]
    path = write_book_file(tmp_path / "book.csv.gz", rows)
    stats = spread_stats("binance-futures", "TESTUSDT", path, None)
    assert stats.unavailable is not None
    assert not stats.passes


def test_hours_limits_the_parse(tmp_path):
    # One row per minute for three hours; --hours 1 must stop after one.
    rows = [(1_700_000_000_000 + i * 60_000, 999.5, 1.0, 1000.5)
            for i in range(180)]
    path = write_book_file(tmp_path / "book.csv.gz", rows)
    full = spread_stats("binance-futures", "TESTUSDT", path, None)
    limited = spread_stats("binance-futures", "TESTUSDT", path, 3_600_000)
    assert limited.samples < full.samples


# ---------------------------------------------------------------------------
# Surviving a bad instrument
# ---------------------------------------------------------------------------


def test_a_symbol_a_venue_does_not_list_costs_a_row_not_the_run(monkeypatch,
                                                               tmp_path):
    def refuse(*args, **kwargs):
        raise SystemExit("\nNot found: https://example/DOGEUSDT.csv.gz\ndetail")

    monkeypatch.setattr(spread_survey, "download", refuse)
    row = measure("bybit", "DOGEUSDT", "2026-09-01", tmp_path, None, None, False)
    assert row.unavailable
    assert not row.passes
    assert row.exchange == "bybit" and row.symbol == "DOGEUSDT"


def test_a_malformed_file_costs_a_row_not_the_run(monkeypatch, tmp_path):
    path = tmp_path / "book.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("not,a,book,file\n1,2,3,4\n")
    monkeypatch.setattr(spread_survey, "download", lambda *a, **k: path)
    row = measure("binance-futures", "X", "2026-09-01", tmp_path, None, None, False)
    assert row.unavailable
    assert not row.passes


def test_discard_downloads_removes_the_file_after_measuring(monkeypatch, tmp_path):
    rows = [(1_700_000_000_000 + i * 100, 999.5, 1.0, 1000.5) for i in range(200)]
    destination = (tmp_path /
                   "binance-futures-TESTUSDT-book_snapshot_5-2026-09-01.csv.gz")
    write_book_file(destination, rows)
    monkeypatch.setattr(spread_survey, "download", lambda *a, **k: destination)
    row = measure("binance-futures", "TESTUSDT", "2026-09-01", tmp_path, None,
                  None, discard=True)
    assert row.unavailable is None
    assert row.median_bps == pytest.approx(10.0, rel=1e-3)
    assert not destination.exists()


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_report_survives_a_survey_where_everything_failed(capsys):
    spread_survey.report(
        [SpreadStats("bybit", "AAA", unavailable="404")], "2026-09-01", None)
    assert "Nothing was measured" in capsys.readouterr().out


def test_report_names_the_shortfall_when_nothing_clears(capsys):
    spread = 0.012
    rows = [SpreadStats("binance-futures", "BTCUSDT", samples=1000,
                        median_bps=spread, above_gate=0.0)]
    spread_survey.report(rows, "2026-09-01", None)
    out = capsys.readouterr().out
    assert "NOTHING CLEARS THE FEE GATE" in out
    # Derived from the configured tier rather than hard-coded: the fee
    # schedule is an input, and a test that pins it would have to be edited
    # every time the tier changes, which is exactly when it should still pass.
    assert f"{ROUND_TRIP_MAKER_BPS / spread:.0f}x" in out


def test_report_calls_a_part_time_spread_a_selective_question(capsys):
    rows = [SpreadStats("bybit", "MIDUSDT", samples=1000,
                        median_bps=0.9, above_gate=0.30)]
    spread_survey.report(rows, "2026-09-01", None)
    out = capsys.readouterr().out
    assert "selective" in out.lower()
    assert "NOTHING CLEARS" not in out


def test_a_passing_symbol_is_a_shortlist_and_the_report_says_so(capsys):
    rows = [SpreadStats("bybit", "WIDEUSDT", samples=1000,
                        median_bps=ROUND_TRIP_MAKER_BPS * 2.5, above_gate=0.9)]
    spread_survey.report(rows, "2026-09-01", None)
    out = capsys.readouterr().out
    assert "CLEARS THE GATE" in out
    # A wide spread is not free money, and the report must not let that pass
    # without saying so — it is the single most likely way to misread this.
    assert "adverse selection" in out.lower()
    assert "passive_sim.py" in out


# ---------------------------------------------------------------------------
# The fee schedule the gate is built on
# ---------------------------------------------------------------------------


def test_every_known_tier_has_a_maker_below_its_taker():
    import config

    for tier, (maker, taker) in config.VIP_TIERS.items():
        assert maker < taker, f"VIP {tier} has maker >= taker"


def test_no_tier_pays_a_maker_rebate():
    # The floor is 0% maker at VIP 5. If a negative rate ever appears here it
    # inverts the passive economics entirely and every conclusion in
    # analysis/README under passive_sim.py has to be revisited -- so it should
    # not slip in unnoticed.
    import config

    assert min(maker for maker, _ in config.VIP_TIERS.values()) >= 0


def test_the_tier_is_selectable_and_an_unknown_one_is_refused(monkeypatch):
    import importlib

    import config

    monkeypatch.setenv("BLOFIN_VIP_TIER", "1")
    reloaded = importlib.reload(config)
    assert reloaded.COST_MAKER_MAKER_BPS == reloaded.MAKER_FEE_BPS * 2
    assert reloaded.VIP_TIER == 1

    # Tiers 3 and 4 exist on BloFin but their rates were never confirmed.
    # Guessing them would be worse than refusing.
    monkeypatch.setenv("BLOFIN_VIP_TIER", "3")
    with pytest.raises(SystemExit):
        importlib.reload(config)

    monkeypatch.delenv("BLOFIN_VIP_TIER")
    importlib.reload(config)
