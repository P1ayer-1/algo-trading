"""The lead-quote board and live follower, against hand-computed logs.

The board is where the S3 decision gets read - a sign of net bps per fill and
its standard error, per pair and pooled - so the pooling has to be over fills
and the run states have to mean what they say. The follower is only useful if
it says each fill exactly once, including a row it caught half-written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lead_quote_board import parse_since
from trading.strategies.lead_quote.monitor import (
    LogTail, MonitorReport, Narrator, board, run_started_ms, run_state, summarise)

T0 = 1_789_261_200_000          # 2026-09-13 01:00:00 UTC


def write_log(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def closed(t, net, passive=True):
    return {"t": t, "event": "paper_fill_closed", "side": 1, "entry": 1.0, "exit": 1.0,
            "hold_ms": 1000, "passive_exit": passive, "net_bps": net}


def test_the_run_start_comes_from_the_file_name_the_runner_stamps():
    assert run_started_ms(Path("2026-09-13T010000-demo.jsonl")) == T0
    assert run_started_ms(Path("notes.jsonl")) is None


def test_the_board_pools_over_fills_not_over_run_means(tmp_path):
    """Two runs: three fills of +4, +4, +1 and one fill of -5. Over fills the
    pooled net is (4+4+1-5)/4 = +1.00; a mean of the two run means would say
    (3 + -5)/2 = -1.00 and flip the sign. SE by hand: deviations 3, 3, 0, -6,
    squares sum 54, variance 54/3 = 18, SE sqrt(18)/2 = 2.12."""
    a = write_log(tmp_path / "SUI-USDT" / "lead_quote" / "2026-09-13T010000-demo.jsonl", [
        {"t": T0, "event": "start", "inst_id": "SUI-USDT", "dry_run": False},
        closed(T0 + 1000, 4.0), closed(T0 + 2000, 4.0), closed(T0 + 3000, 1.0, passive=False),
        {"t": T0 + 3_600_000, "event": "stop"},
    ])
    b = write_log(tmp_path / "DOGE-USDT" / "lead_quote" / "2026-09-13T020000-paper.jsonl", [
        {"t": T0 + 3_600_000, "event": "start", "inst_id": "DOGE-USDT", "dry_run": True},
        closed(T0 + 7_200_000, -5.0, passive=False),
    ])
    reports = [summarise(a), summarise(b)]
    lines = board(reports, now_ms=T0 + 7_260_000)
    pooled = next(line for line in lines if line.startswith("ALL "))
    # runs, hours (1.0 each), posts, fills, per day (4 / 2 h * 24), net, SE,
    # t (1.00 / 2.12), sum of nets, maker exits (2 of 4)
    assert pooled.split() == ["ALL", "2", "2.0", "0", "4", "48", "+1.00", "±2.12", "+0.5", "+4.0", "50%"]
    first_table = lines[lines.index("NEWEST RUN PER PAIR"):lines.index("EVERY RUN, POOLED")]
    newest = {line.split()[0]: line for line in first_table if line.startswith(("SUI-USDT", "DOGE-USDT"))}
    assert " stopped " in newest["SUI-USDT"] and " demo " in newest["SUI-USDT"]
    assert " live " in newest["DOGE-USDT"] and " paper " in newest["DOGE-USDT"]


def test_a_run_is_live_stopped_or_silent():
    """A stop row wins; otherwise three minutes without a row is silent. A
    killed process writes no stop row, so without the clock it would read as
    running forever."""
    report = MonitorReport("SUI-USDT", last_ms=T0)
    assert run_state(report, T0 + 180_000) == "live"
    assert run_state(report, T0 + 180_001) == "silent"
    report.stopped = True
    assert run_state(report, T0 + 10 ** 9) == "stopped"


def test_an_open_fill_is_not_called_a_run_that_ended_holding_while_it_runs(tmp_path):
    path = write_log(tmp_path / "SUI-USDT" / "lead_quote" / "2026-09-13T010000-demo.jsonl", [
        {"t": T0, "event": "paper_fill", "side": 1, "entry": 1.0, "wait_ms": 10},
    ])
    assert summarise(path).problems == ["1 paper fills not closed at the last row"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"t": T0 + 1, "event": "stop"}) + "\n")
    assert summarise(path).problems == ["1 paper fills never closed (run ended holding)"]


def test_the_tail_returns_a_half_written_row_once_its_newline_lands(tmp_path):
    """The runner can be mid-write when the follower reads. Parsing the
    fragment would drop the row; skipping it without keeping it would lose it."""
    path = tmp_path / "SUI-USDT" / "lead_quote" / "2026-09-13T010000-demo.jsonl"
    path.parent.mkdir(parents=True)
    tail = LogTail(lambda: [path] if path.exists() else [])
    assert tail.poll() == []
    row = json.dumps({"t": 1, "event": "paper_fill"}) + "\n"
    path.write_text(row[:10], encoding="utf-8")
    assert tail.poll() == []
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row[10:])
    assert [r["event"] for _, r in tail.poll()] == ["paper_fill"]
    assert tail.poll() == []


def test_the_tail_reads_a_replaced_file_again_from_the_start(tmp_path):
    path = write_log(tmp_path / "run.jsonl", [{"t": 1, "event": "a"}, {"t": 2, "event": "b"}])
    tail = LogTail(lambda: [path])
    assert len(tail.poll()) == 2
    write_log(path, [{"t": 3, "event": "c"}])
    assert [r["event"] for _, r in tail.poll()] == ["c"]


def test_an_exit_line_carries_the_pair_and_pooled_tally():
    """+4 then -2 on SUI: mean +1.00; deviations 3, -3, variance 18/1, SE
    sqrt(18/2) = 3.00. A DOGE fill of +1 makes the pooled mean (4 - 2 + 1)/3 =
    +1.00; deviations 3, -3, 0, variance 18/2 = 9, SE sqrt(9/3) = 1.73."""
    narrator = Narrator()
    sui = Path("data/SUI-USDT/lead_quote/2026-09-13T010000-demo.jsonl")
    doge = Path("data/DOGE-USDT/lead_quote/2026-09-13T010000-demo.jsonl")
    narrator.feed(sui, closed(T0, 4.0))
    narrator.feed(sui, closed(T0 + 1, -2.0))
    [(level, text)] = narrator.feed(doge, closed(T0 + 2, 1.0))
    assert level == "win"
    assert "DOGE-USDT 1 fills +1.00 | all 3 fills +1.00 ±1.73" in text
    assert narrator.tally("SUI-USDT") == "SUI-USDT 2 fills +1.00 ±3.00"
    [(level, _)] = narrator.feed(sui, closed(T0 + 3, -0.5))
    assert level == "loss"


def test_a_refused_cancel_of_a_post_only_the_venue_already_cancelled_is_not_shouted():
    """The demo book cancels a post_only that would cross it, and our later
    cancel is refused. That is the demo book disagreeing with production, not
    a fault (monitor.summarise); a real rejection is said on the 1st and
    every 10th repeat, so a pair rejecting every post cannot flood the feed."""
    narrator = Narrator()
    path = Path("data/FIL-USDT/lead_quote/2026-09-13T010000-demo.jsonl")
    narrator.feed(path, {"t": T0, "event": "demo_order", "cid": "x", "kind": "post", "state": "canceled"})
    assert narrator.feed(path, {"t": T0, "event": "demo_ack", "kind": "cancel", "cid": "x", "ok": False,
                                "code": "102068"}) == []
    said = []
    for i in range(20):
        said += narrator.feed(path, {"t": T0 + i, "event": "demo_ack", "kind": "post", "cid": "p" + str(i),
                                     "ok": False, "code": "102016", "msg": "Precision does not match: 0.001"})
    assert len(said) == 3 and all(level == "warn" for level, _ in said)
    assert "(20 times)" in said[-1][1]


def test_silence_is_said_once_and_so_is_the_recovery():
    narrator = Narrator(silent_after_ms=180_000)
    path = Path("data/SUI-USDT/lead_quote/2026-09-13T010000-demo.jsonl")
    narrator.feed(path, {"t": T0, "event": "lag"})
    assert narrator.check_silence(T0 + 180_000) == []
    [(level, text)] = narrator.check_silence(T0 + 240_000)
    assert level == "warn" and "SILENT" in text and "4.0 min" in text
    assert narrator.check_silence(T0 + 300_000) == []
    assert [t for _, t in narrator.feed(path, {"t": T0 + 310_000, "event": "lag"})][0].endswith("writing again")
    narrator.feed(path, {"t": T0 + 311_000, "event": "stop"})
    assert narrator.check_silence(T0 + 10 ** 9) == []


def test_since_takes_a_utc_time_or_an_age():
    assert parse_since("2026-09-13T01:00", 0) == T0
    assert parse_since("2026-09-13T01", 0) == T0
    assert parse_since("12h", T0) == T0 - 12 * 3_600_000
    with pytest.raises(SystemExit):
        parse_since("yesterday", T0)
