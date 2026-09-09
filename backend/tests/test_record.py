"""The headless multi-instrument recorder, and the isolation it depends on.

The whole point of `record.py` is running several instruments at once without
their data touching. These tests are about that separation, not about the
websocket loop, which `test_ingest.py` already covers.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pytest.importorskip("blofin", reason="BloFin SDK not installed")

import record  # noqa: E402
from trading.ingest import MicrostructureFeed  # noqa: E402


def test_each_instrument_writes_to_its_own_directory(tmp_path):
    """The fix for the hazard: two symbols, two trees, no shared file.

    Rows for two instruments are structurally identical - same columns, same
    order, same dtypes - so nothing downstream can tell them apart once they
    are in one file. Directory layout is the only separation there is.
    """
    btc = MicrostructureFeed("BTC-USDT", data_dir=tmp_path, record=False,
                             record_raw=False)
    ada = MicrostructureFeed("ADA-USDT", data_dir=tmp_path, record=False,
                             record_raw=False)

    assert btc.recorder.data_dir == tmp_path / "BTC-USDT"
    assert ada.recorder.data_dir == tmp_path / "ADA-USDT"
    assert btc.raw_log.root == tmp_path / "BTC-USDT" / "raw"
    assert ada.raw_log.root == tmp_path / "ADA-USDT" / "raw"

    assert btc.recorder.data_dir != ada.recorder.data_dir
    assert btc.raw_log.root != ada.raw_log.root


def test_the_same_instrument_twice_is_refused(tmp_path):
    """Two feeds on one symbol interleave two independent books into one file.

    That is corruption wearing the costume of more data, and it would be
    invisible afterwards - the rows are well-formed and the timestamps
    overlap.
    """
    with pytest.raises(SystemExit, match="more than once"):
        record.main(["--instruments", "BTC-USDT,ADA-USDT,BTC-USDT",
                     "--data-dir", str(tmp_path), "--list-cost"])


def test_an_empty_instrument_list_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="No instruments"):
        record.main(["--instruments", " , ", "--data-dir", str(tmp_path),
                     "--list-cost"])


def test_list_cost_reports_the_arithmetic_without_connecting(tmp_path):
    """--list-cost must not open a socket. It is the thing you run first."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = record.main([
            "--instruments", "BTC-USDT,ADA-USDT,PUMP-USDT",
            "--data-dir", str(tmp_path), "--list-cost",
        ])

    assert code == 0
    report = buffer.getvalue()
    assert "3 instrument(s)" in report
    # 3 x 140 MB/day
    assert "0.42 GB" in report
    assert "12.60 GB" in report
    # Nothing should have been created by a cost estimate.
    assert list(tmp_path.iterdir()) == []


def test_the_cost_scales_with_the_number_of_instruments(tmp_path):
    def daily_line(count):
        names = ",".join(f"SYM{index}-USDT" for index in range(count))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            record.main(["--instruments", names, "--data-dir", str(tmp_path),
                         "--list-cost"])
        for line in buffer.getvalue().splitlines():
            if "per day" in line:
                return float(line.split()[-2])
        raise AssertionError("no per-day line")

    assert daily_line(2) == pytest.approx(2 * record.MB_PER_DAY_PER_INSTRUMENT / 1000)
    assert daily_line(8) == pytest.approx(8 * record.MB_PER_DAY_PER_INSTRUMENT / 1000)


def test_build_feed_carries_the_stall_watchdog_through(tmp_path):
    """A headless run is exactly the one nobody is watching.

    If the stall timeout did not reach these feeds, a silent subscription
    would cost the whole overnight window on that instrument with no error
    anywhere.
    """
    feed = record.build_feed("ADA-USDT", tmp_path)
    assert feed.stall_timeout_s > 0
    assert feed.inst_id == "ADA-USDT"
