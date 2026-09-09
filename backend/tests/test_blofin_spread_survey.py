"""The BloFin-native spread survey, against a faked getTickers/getInstruments.

No network. The API is two methods returning dicts, so a stub is enough, and
the things worth testing are all arithmetic on those dicts: the bps
conversion, the tick floor, the filters that keep untradeable instruments out
of the shortlist, and the refusal to average a crossed book.
"""

import pytest

from analysis.blofin_spread_survey import (
    EMPIRICAL_GATE_BPS,
    SpreadSamples,
    collect,
    eligible,
    fetch_instruments,
    poll_once,
)


class FakeAPI:
    """getTickers() walks a script of quote sets, one per poll."""

    def __init__(self, instruments, polls):
        self._instruments = instruments
        self._polls = list(polls)
        self.calls = 0

    def getInstruments(self):
        return {"data": self._instruments}

    def getTickers(self):
        index = min(self.calls, len(self._polls) - 1)
        self.calls += 1
        return {"data": self._polls[index]}


def instrument(inst_id, tick="0.001", contract_type="linear", quote="USDT",
               state="live"):
    return {"instId": inst_id, "tickSize": tick, "contractType": contract_type,
            "quoteCurrency": quote, "state": state}


def ticker(inst_id, bid, ask, volume="1000000"):
    return {"instId": inst_id, "bidPrice": str(bid), "askPrice": str(ask),
            "volCurrency24h": volume}


def test_spread_is_measured_in_bps_off_the_mid():
    entry = SpreadSamples("X-USDT")
    entry.add(bid=0.9995, ask=1.0005)          # 10 bps on a mid of 1.0
    assert entry.median_bps == pytest.approx(10.0, abs=1e-6)


def test_a_crossed_book_is_not_a_measurement():
    """Averaging a crossed quote in would understate the width.

    Understating is the direction that flatters, which is the direction this
    project has been wrong in before.
    """
    entry = SpreadSamples("X-USDT")
    entry.add(bid=1.001, ask=1.000)   # crossed
    entry.add(bid=1.000, ask=1.000)   # locked
    entry.add(bid=0.0, ask=1.000)     # no bid
    assert entry.samples == 0

    entry.add(bid=0.999, ask=1.001)
    assert entry.samples == 1


def test_tick_floor_identifies_a_pinned_instrument():
    """A spread sitting on its tick is a queue, not a payment for risk."""
    instruments = fetch_instruments(FakeAPI([instrument("PIN-USDT", tick="0.01")], []))
    samples = {}
    api = FakeAPI([], [[ticker("PIN-USDT", bid="9.99", ask="10.00")]])
    poll_once(api, samples, instruments)

    entry = samples["PIN-USDT"]
    # One tick of 0.01 on a mid of ~10 is 10 bps, and the quote is one tick.
    assert entry.tick_bps == pytest.approx(10.0, abs=0.05)
    assert entry.median_bps == pytest.approx(10.0, abs=0.05)
    assert entry.tick_bound


def test_a_spread_well_above_its_tick_is_not_pinned():
    instruments = fetch_instruments(FakeAPI([instrument("WIDE-USDT", tick="0.001")], []))
    samples = {}
    api = FakeAPI([], [[ticker("WIDE-USDT", bid="9.95", ask="10.05")]])
    poll_once(api, samples, instruments)

    entry = samples["WIDE-USDT"]
    assert not entry.tick_bound, "makers are setting this width, not the tick"


def test_median_survives_a_transient_wide_quote():
    """The reason the tool polls instead of taking one snapshot.

    A single reading caught ATOM-USDT at 16.3 bps when its resting spread was
    5.4 - a real observation of a wide moment, and a useless median.
    """
    instruments = fetch_instruments(FakeAPI([instrument("A-USDT")], []))
    polls = [[ticker("A-USDT", bid="0.9995", ask="1.0005")]] * 4
    polls.insert(2, [ticker("A-USDT", bid="0.995", ask="1.005")])  # the spike
    api = FakeAPI([], polls)

    samples = {}
    for _ in range(len(polls)):
        poll_once(api, samples, instruments)

    entry = samples["A-USDT"]
    assert entry.samples == 5
    assert entry.median_bps == pytest.approx(10.0, abs=1e-6)
    assert max(entry.spreads_bps) == pytest.approx(100.0, abs=1e-6)


def test_a_spread_that_is_wide_often_shows_in_p75():
    """The median hides a spike; it must not hide a habit.

    An instrument wide a third of the time is a selective quoting question,
    and the p25/p75 columns are what separate that from a pinned one.
    """
    instruments = fetch_instruments(FakeAPI([instrument("A-USDT")], []))
    narrow = [ticker("A-USDT", bid="0.9995", ask="1.0005")]
    wide = [ticker("A-USDT", bid="0.995", ask="1.005")]
    polls = [narrow, narrow, narrow, wide, wide]
    api = FakeAPI([], polls)

    samples = {}
    for _ in range(len(polls)):
        poll_once(api, samples, instruments)

    entry = samples["A-USDT"]
    assert entry.median_bps == pytest.approx(10.0, abs=1e-6)
    assert entry.p75_bps > entry.median_bps


def test_volume_is_priced_in_usd_not_base_units():
    """volCurrency24h is in the BASE currency.

    Comparing it raw across instruments would rank a coin by how many units
    trade rather than by how much money does.
    """
    instruments = fetch_instruments(FakeAPI([instrument("A-USDT")], []))
    samples = {}
    api = FakeAPI([], [[ticker("A-USDT", bid="99", ask="101", volume="1000")]])
    poll_once(api, samples, instruments)

    assert samples["A-USDT"].volume_usd_24h == pytest.approx(100_000.0)


def test_illiquid_instruments_are_filtered_out():
    """A wide spread on something nobody trades is not an opportunity."""
    instruments = fetch_instruments(FakeAPI([instrument("THIN-USDT")], []))
    samples = {}
    api = FakeAPI([], [[ticker("THIN-USDT", bid="0.99", ask="1.01", volume="10")]])
    poll_once(api, samples, instruments)

    entry = samples["THIN-USDT"]
    assert entry.median_bps > EMPIRICAL_GATE_BPS, "it does clear on spread alone"
    assert not eligible(entry, instruments, quote="USDT", linear_only=True,
                        min_volume_usd=1e6)
    assert eligible(entry, instruments, quote="USDT", linear_only=True,
                    min_volume_usd=0)


def test_inverse_contracts_are_excluded_by_default():
    """Their PnL is not linear in price, so the passive arithmetic differs."""
    rows = [instrument("BTC-USD", contract_type="inverse", quote="USD"),
            instrument("BTC-USDT", contract_type="linear")]
    instruments = fetch_instruments(FakeAPI(rows, []))
    samples = {}
    api = FakeAPI([], [[ticker("BTC-USD", bid="99", ask="101"),
                        ticker("BTC-USDT", bid="99", ask="101")]])
    poll_once(api, samples, instruments)

    kept = [inst for inst, entry in samples.items()
            if eligible(entry, instruments, quote=None, linear_only=True,
                        min_volume_usd=0)]
    assert kept == ["BTC-USDT"]


def test_delisted_instruments_are_excluded():
    instruments = fetch_instruments(
        FakeAPI([instrument("DEAD-USDT", state="suspend")], []))
    samples = {}
    api = FakeAPI([], [[ticker("DEAD-USDT", bid="0.99", ask="1.01")]])
    poll_once(api, samples, instruments)

    assert not eligible(samples["DEAD-USDT"], instruments, quote="USDT",
                        linear_only=True, min_volume_usd=0)


def test_a_failed_poll_does_not_end_the_survey():
    """One transient HTTP error must not cost the whole window.

    A failure in this shape - an unhandled exception in the one loop with no
    error handling - is what ended a seven-hour recorder run. Here the window
    is the expensive thing: it cannot be re-polled after the fact.
    """
    instruments = fetch_instruments(FakeAPI([instrument("A-USDT")], []))

    class Flaky(FakeAPI):
        """Fails the first poll, then works."""

        def getTickers(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection reset")
            return {"data": [ticker("A-USDT", bid="0.999", ask="1.001")]}

    api = Flaky([], [])
    samples = collect(api, minutes=0.005, interval=0.02, instruments=instruments)

    assert api.calls > 1, "the survey must keep polling after a failure"
    assert samples["A-USDT"].samples == api.calls - 1
    assert samples["A-USDT"].median_bps == pytest.approx(20.0, abs=1e-6)


def test_a_poll_that_never_succeeds_still_returns():
    """An unreachable venue is an empty result, not a traceback."""
    instruments = fetch_instruments(FakeAPI([instrument("A-USDT")], []))

    class Dead(FakeAPI):
        def getTickers(self):
            self.calls += 1
            raise RuntimeError("host unreachable")

    samples = collect(Dead([], []), minutes=0.0, interval=0.0,
                      instruments=instruments, sleep=lambda _: None)
    assert samples == {}


def test_the_empirical_gate_is_the_fee_plus_measured_adverse_selection():
    """It must stay derived from the fee, not hardcoded.

    The whole point of importing MAKER_BPS is that changing BLOFIN_VIP_TIER
    moves this bar with it.
    """
    from analysis.passive_sim import MAKER_BPS

    assert EMPIRICAL_GATE_BPS == pytest.approx(2 * (0.5 + float(MAKER_BPS)))
