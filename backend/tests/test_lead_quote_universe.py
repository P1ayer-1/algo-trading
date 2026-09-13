"""The lead-quote universe screen: hand-computed gates on canned venue payloads."""

import pytest

from analysis.lead_quote_universe import MIN_SPREAD_TICKS, lines, main, screen, spread_samples

INSTRUMENTS = [
    {"instId": "SUI-USDT", "tickSize": "0.0001", "contractValue": "1", "state": "live"},
    {"instId": "ADA-USDT", "tickSize": "0.0001", "contractValue": "1", "state": "live"},
    {"instId": "BTC-USDT", "tickSize": "0.1", "contractValue": "0.001", "state": "live"},
    {"instId": "TINY-USDT", "tickSize": "0.00001", "contractValue": "1", "state": "live"},
    {"instId": "NOLEAD-USDT", "tickSize": "0.001", "contractValue": "1", "state": "live"},
    {"instId": "BTC-USD", "tickSize": "0.1", "contractValue": "1", "state": "live"},
]
TICKERS = [
    {"instId": "SUI-USDT", "last": "0.72", "bidPrice": "0.7198", "askPrice": "0.7201", "vol24h": "3700000"},
    {"instId": "ADA-USDT", "last": "0.21", "bidPrice": "0.2099", "askPrice": "0.2101", "vol24h": "10000000"},
    {"instId": "BTC-USDT", "last": "110000", "bidPrice": "109999.9", "askPrice": "110000.0", "vol24h": "50000000"},
    {"instId": "TINY-USDT", "last": "0.5", "bidPrice": "0.49998", "askPrice": "0.50001", "vol24h": "100000"},
    {"instId": "NOLEAD-USDT", "last": "5", "bidPrice": "4.999", "askPrice": "5.002", "vol24h": "1000000"},
    {"instId": "BTC-USD", "last": "110000", "bidPrice": "109999", "askPrice": "110001", "vol24h": "1"},
]
BINANCE = [
    {"symbol": "SUIUSDT", "quoteVolume": "152000000", "count": "601033"},
    {"symbol": "ADAUSDT", "quoteVolume": "200000000", "count": "500000"},
    {"symbol": "BTCUSDT", "quoteVolume": "9000000000", "count": "3000000"},
    {"symbol": "TINYUSDT", "quoteVolume": "5000000", "count": "1000"},
]


def _screen(**kw):
    kw.setdefault("min_blofin_usd", 1e6)
    kw.setdefault("min_binance_usd", 2e7)
    return {c.inst_id: c for c in screen(INSTRUMENTS, TICKERS, BINANCE, **kw)}


def test_sui_passes_with_hand_computed_numbers():
    """tick 0.0001 / 0.72 = 1.389 bps; spread 0.0003 = 3 ticks = 4.17 bps of the 0.71995 mid;
    BloFin volume 3.7M contracts x 1 x 0.72 = $2.664M."""
    sui = _screen()["SUI-USDT"]
    assert sui.ok
    assert sui.tick_bps == pytest.approx(1.3889, abs=1e-3)
    assert sui.spread_ticks == pytest.approx(3.0, abs=1e-6)
    assert sui.spread_bps == pytest.approx(0.0003 / 0.71995 * 1e4, abs=1e-3)
    assert sui.blofin_usd == pytest.approx(2_664_000)
    assert sui.binance_usd == pytest.approx(152e6) and sui.binance_trades == 601033


def test_the_gates_name_every_failure():
    """ADA: tick 4.76 bps > 2.5 (9ae's loser). BTC: tick 0.009 bps under the floor AND a
    one-tick spread. TINY: tick 0.00001 / 0.5 = 0.2 bps, under the floor, and it fails both
    volume floors. NOLEAD: no Binance symbol. BTC-USD: coin-margined, not listed at all."""
    got = _screen()
    assert got["ADA-USDT"].reasons == ["tick 4.76 bps > 2.5"]
    assert [r.split(":")[0] for r in got["BTC-USDT"].reasons] == ["tick 0.01 bps < 0.3", "spread 1.0 ticks < 2"]
    tiny = got["TINY-USDT"].reasons
    assert tiny[0].startswith("tick 0.20 bps < 0.3")
    assert "BloFin $0.1M/day < $1.0M" in tiny and "Binance $5M/day < $20M" in tiny
    assert got["NOLEAD-USDT"].reasons == ["no Binance USDT-M future named NOLEADUSDT"]
    assert "BTC-USD" not in got


def test_sorted_by_blofin_volume_and_the_table_hides_failures_unless_asked():
    cands = screen(INSTRUMENTS, TICKERS, BINANCE, min_blofin_usd=1e6, min_binance_usd=2e7)
    assert [c.inst_id for c in cands][:3] == ["BTC-USDT", "NOLEAD-USDT", "SUI-USDT"]   # $5.5B, $5M, $2.66M
    shown = lines(cands, show_all=False, top=10)
    assert len(shown) == 2 and shown[1].startswith("SUI-USDT")
    everything = lines(cands, show_all=True, top=10)
    assert len(everything) == 1 + 5


def test_the_spread_gate_is_the_median_of_samples_and_room_is_their_share():
    """LINK read 1, 2 and 6 ticks in three reads (2026-09-13). Five samples of SUI at
    1, 1, 3, 3, 4 ticks: median 3 passes, room 3/5. Five at 1, 1, 1, 3, 4: median 1 fails."""
    snapshots = [[{"instId": "SUI-USDT", "bidPrice": "0.7200", "askPrice": str(0.72 + k * 0.0001)}]
                 for k in (1, 1, 3, 3, 4)]
    samples = spread_samples(snapshots)
    assert samples["SUI-USDT"] == pytest.approx([0.0001, 0.0001, 0.0003, 0.0003, 0.0004])
    got = _screen(spreads=samples)["SUI-USDT"]
    assert got.ok and got.spread_ticks == pytest.approx(3.0) and got.room_share == pytest.approx(0.6)
    narrow = spread_samples([[{"instId": "SUI-USDT", "bidPrice": "0.7200", "askPrice": str(0.72 + k * 0.0001)}]
                             for k in (1, 1, 1, 3, 4)])
    got = _screen(spreads=narrow)["SUI-USDT"]
    assert got.reasons == ["spread 1.0 ticks < 2: nothing to post inside"] and got.room_share == pytest.approx(0.4)


def test_studied_instruments_carry_their_9ae_verdict():
    got = _screen()
    assert got["ADA-USDT"].note == "9ae: negative every day"
    assert got["SUI-USDT"].note.startswith("9ae: positive")
    assert got["NOLEAD-USDT"].note == ""
    assert "[9ae: negative every day]" in [l for l in lines(list(got.values()), show_all=True, top=9)
                                           if l.startswith("ADA")][0]


def test_the_demo_column_splits_the_launch_lines():
    """FLOCK and USELESS passed the screen and then exited under --confirm without a log:
    the demo host does not list them. A pair not on demo is paper-only."""
    demo = [{"instId": "SUI-USDT"}]
    got = {c.inst_id: c for c in screen(INSTRUMENTS, TICKERS, BINANCE, min_blofin_usd=1e6,
                                        min_binance_usd=2e7, demo_instruments=demo)}
    assert got["SUI-USDT"].on_demo is True and got["ADA-USDT"].on_demo is False
    row = [l for l in lines(list(got.values()), show_all=True, top=9) if l.startswith("ADA")][0]
    assert "  no  " in row


def test_main_prints_the_command_line_for_the_passing_set(capsys):
    payloads = {
        "https://demo-trading-openapi.blofin.com/api/v1/market/instruments?instType=SWAP": {"data": []},
        "https://openapi.blofin.com/api/v1/market/instruments?instType=SWAP": {"data": INSTRUMENTS},
        "https://openapi.blofin.com/api/v1/market/tickers?instType=SWAP": {"data": TICKERS},
        "https://fapi.binance.com/fapi/v1/ticker/24hr": BINANCE,
    }
    slept = []
    assert main(["--top", "5", "--samples", "3", "--interval", "0.5"],
                fetch=lambda url: payloads[url], sleep=slept.append) == 0
    assert slept == [0.5, 0.5]
    out = capsys.readouterr().out
    assert "1 pass the 9ae gate" in out and "over 3 samples" in out
    assert "NOT on the demo host" in out and "for i in SUI-USDT; do" in out
    launch = out.split("paper only:\n")[1].splitlines()[0]
    assert launch.startswith("for i in SUI-USDT; do") and "--confirm" not in launch
    assert MIN_SPREAD_TICKS == 2.0
