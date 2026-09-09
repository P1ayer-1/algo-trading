"""The venue comparison, and the guards that stop it overclaiming.

The arithmetic here is thin on purpose — `passive_sim` is doing the real work
and is tested separately. What matters at this level is the reasoning: the
control has to be able to veto the conclusion, a thin sample has to be
labelled, and a difference inside its own interval must not be reported as an
effect.
"""

import datetime as dt
import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from analysis.passive_sim import adverse_selection
from analysis.venue_compare import (
    CONTROL,
    MIN_HOURS,
    Measurement,
    binance_symbol,
    difference_in_differences,
    latest_free_sample_day,
    paired_difference,
    report,
)


def measurement(venue, symbol, adverse, stderr, *, hours=24.0, fills=5000,
                spread=1.0):
    return Measurement(venue=venue, symbol=symbol, median_spread_bps=spread,
                       adverse_bps=adverse, adverse_stderr=stderr,
                       markout_at_fill=0.0, fills=fills, span_hours=hours)


def run_report(rows):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(rows, horizon=60.0, model="pessimistic",
               blofin_date="2026-09-09", binance_date="2026-09-01")
    return buffer.getvalue()


def pair(name, blofin_adverse, binance_adverse, stderr=0.05, hours=24.0):
    return (name,
            measurement("blofin", name, blofin_adverse, stderr, hours=hours),
            measurement("binance", binance_symbol(name), binance_adverse,
                        stderr, hours=hours))


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------


def test_symbol_mapping_drops_the_hyphen():
    assert binance_symbol("ADA-USDT") == "ADAUSDT"
    assert binance_symbol("1000BONK-USDT") == "1000BONKUSDT"


def test_the_free_sample_day_is_a_date_not_a_day_number():
    """FREE_SAMPLE_DAY is 1, the day-of-month. Passing it as a date gave
    `'int' object has no attribute 'split'` deep inside the downloader."""
    assert latest_free_sample_day(dt.date(2026, 9, 9)) == "2026-09-01"
    # On the 1st itself, that day is not necessarily archived yet.
    assert latest_free_sample_day(dt.date(2026, 9, 1)) == "2026-08-01"
    assert latest_free_sample_day(dt.date(2026, 1, 1)) == "2025-12-01"


def test_adverse_selection_is_the_decay_from_the_fill():
    """Positive means the price moved against the fill after it happened."""
    horizons = (0.0, 60.0)
    table = {"pessimistic": (np.array([0.5, -0.3]), np.array([0.1, 0.1]), 100)}
    decay, stderr = adverse_selection(table, horizons, "pessimistic", 60.0)

    assert decay == pytest.approx(0.8)
    assert stderr == pytest.approx(np.hypot(0.1, 0.1))


def test_adverse_selection_reports_nothing_without_fills():
    horizons = (0.0, 60.0)
    table = {"pessimistic": (np.array([np.nan, np.nan]),
                             np.array([np.nan, np.nan]), 0)}
    decay, _ = adverse_selection(table, horizons, "pessimistic", 60.0)
    assert np.isnan(decay)


def test_the_difference_interval_widens_with_both_sides():
    blofin = measurement("blofin", "ADA-USDT", 1.0, 0.3)
    binance = measurement("binance", "ADAUSDT", 0.4, 0.4)
    difference, stderr = paired_difference(blofin, binance)

    assert difference == pytest.approx(0.6)
    assert stderr == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The verdict, which is the part that can mislead
# ---------------------------------------------------------------------------


def test_the_control_is_subtracted_not_used_as_a_veto():
    """The point of the control, and the fix to how it was used.

    BTC is tick-bound with an identical spread on both venues, so it cannot
    carry a venue effect: whatever it shows is the date gap. Subtracting it is
    what isolates the venue. The earlier version tested the control's POINT
    estimate against a threshold and vetoed everything above it, which threw
    away both the control's information and its uncertainty.
    """
    control = pair(CONTROL, 3.0, 0.5)     # control shifted by +2.5
    instrument = pair("ADA-USDT", 4.0, 0.5)   # raw +3.5, so +1.0 net

    effect, _ = difference_in_differences(instrument, control)
    assert effect == pytest.approx(1.0)

    report_text = run_report([control, instrument])
    assert "net of control" in report_text
    assert "selects HARDER on: ADA-USDT" in report_text


def test_a_shifted_control_removes_the_shift_from_every_instrument():
    """An instrument that moved exactly as much as the control has no effect."""
    control = pair(CONTROL, 3.0, 0.5)
    instrument = pair("ADA-USDT", 3.0, 0.5)   # identical raw difference

    effect, _ = difference_in_differences(instrument, control)
    assert effect == pytest.approx(0.0)

    report_text = run_report([control, instrument])
    assert "NO INSTRUMENT SHOWS A VENUE EFFECT" in report_text


def test_an_uncertain_control_widens_every_interval():
    """A noisy control must suppress conclusions, without a special case.

    This is what replaces the veto: uncertainty propagates, so a control that
    cannot be pinned down makes every difference-in-differences interval wide
    enough to include zero on its own.
    """
    control = pair(CONTROL, 0.5, 0.5, stderr=10.0)
    instrument = pair("ADA-USDT", 3.0, 0.5, stderr=0.05)

    effect, stderr = difference_in_differences(instrument, control)
    assert effect == pytest.approx(2.5)
    assert 1.96 * stderr > abs(effect), "a noisy control must not permit a claim"

    report_text = run_report([control, instrument])
    assert "NO INSTRUMENT SHOWS A VENUE EFFECT" in report_text


def test_a_significant_control_is_announced_as_systematic_bias():
    report_text = run_report([pair(CONTROL, 3.0, 0.5), pair("ADA-USDT", 4.0, 0.5)])
    assert "EXCLUDES zero" in report_text

    report_text = run_report([pair(CONTROL, 0.5, 0.5), pair("ADA-USDT", 4.0, 0.5)])
    assert "includes zero" in report_text


def test_a_flat_control_lets_a_real_difference_through():
    report_text = run_report([
        pair(CONTROL, 0.50, 0.50),        # control flat
        pair("ADA-USDT", 2.00, 0.50),     # +1.5 bps, far outside +-0.14
    ])

    assert "selects HARDER on: ADA-USDT" in report_text


def test_cheaper_adverse_selection_is_called_out_separately():
    """Wide spread AND cheap adverse selection is the case worth chasing, so
    it must not be reported in the same breath as the opposite."""
    report_text = run_report([
        pair(CONTROL, 0.50, 0.50),
        pair("ADA-USDT", 0.10, 1.50),
    ])

    assert "selects LESS on: ADA-USDT" in report_text
    assert "Confirm it on more days" in report_text


def test_a_difference_inside_its_own_interval_is_not_an_effect():
    report_text = run_report([
        pair(CONTROL, 0.50, 0.50),
        pair("ADA-USDT", 1.00, 0.50, stderr=2.0),   # +0.5 against +-3.9
    ])

    assert "NO INSTRUMENT SHOWS A VENUE EFFECT" in report_text
    assert "null result" in report_text


def test_a_tiny_but_significant_difference_is_still_ignored():
    """Statistical significance is not the same as mattering.

    The original survey found 0.29-0.65 bps of spread in this quantity on ONE
    venue, so a 0.05 bps venue difference is noise wearing a small interval.
    """
    report_text = run_report([
        pair(CONTROL, 0.50, 0.50),
        pair("ADA-USDT", 0.55, 0.50, stderr=0.001),
    ])

    assert "NO INSTRUMENT SHOWS A VENUE EFFECT" in report_text


def test_a_thin_blofin_sample_is_labelled_before_anything_else():
    report_text = run_report([
        pair(CONTROL, 0.50, 0.50, hours=0.7),
        pair("ADA-USDT", 2.00, 0.50, hours=0.7),
    ])

    assert "THE BLOFIN SIDE IS 0.7 HOURS" in report_text
    assert f"{MIN_HOURS:.0f}+ hours" in report_text


def test_a_full_sample_carries_no_thinness_warning():
    report_text = run_report([pair(CONTROL, 0.5, 0.5, hours=48.0)])
    assert "provisional" not in report_text


def test_no_control_means_no_conclusion():
    report_text = run_report([pair("ADA-USDT", 2.0, 0.5)])

    assert "NO CONTROL" in report_text
    assert "selects HARDER" not in report_text


def test_an_unusable_side_is_reported_with_its_reason():
    rows = [(CONTROL,
             Measurement("blofin", CONTROL, unavailable="nothing recorded"),
             measurement("binance", "BTCUSDT", 0.5, 0.05))]
    report_text = run_report(rows)

    assert "not compared" in report_text
    assert "nothing recorded" in report_text
    assert "Nothing could be compared" in report_text
