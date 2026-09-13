"""Read a lead-quote run back from its log and score it. READ ONLY.

The runner writes one JSON object per line to `data/<INST>/lead_quote/`:
feed latency samples, every intent, every paper fill with its exit, and the
demo order acknowledgements when orders were mirrored. This turns that into
the numbers step 9ae was measured in - fills, net bps per fill, passive-exit
share - beside the two numbers the backtest could only assume: the leader
and follower feed lag on this host, and the order round trip to the venue.

Beside the one-run summary: `board` pools every run into one table per pair,
and `LogTail` + `Narrator` follow the logs as they are written and say each
paper fill as it lands (`lead_quote_board.py`).

No broker, no network. Nothing here can send anything.
"""

from __future__ import annotations

import calendar
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class MonitorReport:
    inst_id: str
    hours: float = 0.0
    posts: int = 0
    paper_fills: int = 0
    net_bps: float = float("nan")
    net_se: float = float("nan")
    passive_share: float = float("nan")
    fills_per_day: float = float("nan")
    leader_lag_ms: Dict[str, float] = field(default_factory=dict)
    follower_lag_ms: Dict[str, float] = field(default_factory=dict)
    ack_ms: Dict[str, float] = field(default_factory=dict)
    probe_ms: Dict[str, float] = field(default_factory=dict)
    ack_by_kind: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warm_ms: Dict[str, float] = field(default_factory=dict)
    demo_orders: int = 0
    demo_fills: int = 0
    demo_crossed: int = 0          # post_only bids the demo book cancelled on arrival
    gap_episodes: Optional[int] = None
    gap_blocked: Optional[int] = None
    max_gap_bps: Optional[float] = None
    problems: List[str] = field(default_factory=list)
    path: str = ""
    started_ms: Optional[int] = None      # from the file name, which the runner stamps at start
    last_ms: Optional[int] = None
    stopped: bool = False                 # a `stop` row: the runner exited on its own terms
    dry_run: Optional[bool] = None
    fill_nets: List[float] = field(default_factory=list)
    open_fills: int = 0                   # paper fills with no exit yet

    @property
    def alerts(self) -> List[str]:
        return self.problems

    @property
    def critical(self) -> bool:
        return bool(self.problems)

    def lines(self) -> List[str]:
        out = ["{}: {:.2f}h, {} posts, {} paper fills ({:.0f}/day)".format(
            self.inst_id, self.hours, self.posts, self.paper_fills, self.fills_per_day)]
        if self.paper_fills:
            out.append("  net {:+.2f} ±{:.2f} bps per paper fill, passive exits {:.0%}".format(
                self.net_bps, self.net_se, self.passive_share))
        if self.gap_episodes is not None:
            out.append("  leader past the edge {} times (max gap {:.1f} bps), {} with the level occupied".format(
                self.gap_episodes, self.max_gap_bps or 0.0, self.gap_blocked))
        for name, stats in (("leader feed lag", self.leader_lag_ms),
                            ("follower feed lag", self.follower_lag_ms),
                            ("order ack round trip", self.ack_ms),
                            ("probe post+cancel round trip", self.probe_ms),
                            ("keep-warm GET round trip", self.warm_ms)):
            if stats:
                out.append("  {}: p50 {:.0f} ms, p90 {:.0f}, p99 {:.0f} (n {})".format(
                    name, stats["p50"], stats["p90"], stats["p99"], int(stats["n"])))
        if self.ack_by_kind:
            out.append("  ack by kind: " + ", ".join(
                "{} p50 {:.0f} max {:.0f} (n {})".format(kind, st["p50"], st["max"], int(st["n"]))
                for kind, st in self.ack_by_kind.items()))
        if self.demo_orders:
            out.append("  demo: {} orders sent, {} reported filled, {} post_only cancelled on arrival "
                       "(would have crossed the demo book)".format(
                           self.demo_orders, self.demo_fills, self.demo_crossed))
        for problem in self.problems:
            out.append("  PROBLEM: " + problem)
        return out


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)), "max": float(arr.max()), "n": float(len(arr))}


def mean_se(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and its standard error; NaN where undefined (no values; one value for the SE)."""
    if not len(values):
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return float(arr.mean()), se


_RUN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{6})-")


def run_started_ms(path: Path) -> Optional[int]:
    """The UTC start the runner put in the file name (`2026-09-13T133118-demo.jsonl`)."""
    match = _RUN_NAME.match(Path(path).name)
    if not match:
        return None
    return calendar.timegm(time.strptime(match.group(1), "%Y-%m-%dT%H%M%S")) * 1000


def summarise(path: Path) -> MonitorReport:
    events = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    inst = next((e.get("inst_id") for e in events if e.get("inst_id")), Path(path).parent.parent.name)
    report = MonitorReport(inst)
    report.path = str(path)
    report.dry_run = Path(path).stem.endswith("-paper")
    report.started_ms = run_started_ms(path)
    if not events:
        report.problems.append("empty log")
        return report
    times = [e["t"] for e in events if "t" in e]
    if times:
        report.hours = (max(times) - min(times)) / 3.6e6
        report.last_ms = int(max(times))
        if report.started_ms is None:
            report.started_ms = int(min(times))
    report.stopped = any(e.get("event") == "stop" for e in events)
    start = next((e for e in events if e.get("event") == "start"), None)
    if start is not None and "dry_run" in start:
        report.dry_run = bool(start["dry_run"])
    report.posts = sum(1 for e in events if e.get("event") == "intent" and e.get("kind") == "post")
    fills = [e for e in events if e.get("event") == "paper_fill_closed"]
    report.paper_fills = len(fills)
    if fills:
        report.fill_nets = [float(e["net_bps"]) for e in fills]
        report.net_bps, report.net_se = mean_se(report.fill_nets)
        report.passive_share = float(np.mean([1.0 if e.get("passive_exit") else 0.0 for e in fills]))
        report.fills_per_day = len(fills) / max(report.hours, 1e-9) * 24.0
    stats = [e for e in events if e.get("event") == "quoter_stats"]
    if stats:
        report.gap_episodes = int(stats[-1].get("gap_episodes", 0))
        report.gap_blocked = int(stats[-1].get("gap_blocked", 0))
        report.max_gap_bps = float(stats[-1].get("max_gap_bps", 0.0))
    report.leader_lag_ms = _stats([e["lag_ms"] for e in events if e.get("event") == "lag" and e.get("feed") == "leader"])
    report.follower_lag_ms = _stats([e["lag_ms"] for e in events if e.get("event") == "lag" and e.get("feed") == "follower"])
    report.ack_ms = _stats([e["rtt_ms"] for e in events if e.get("event") == "demo_ack" and "rtt_ms" in e])
    report.probe_ms = _stats([e["rtt_ms"] for e in events if e.get("event") == "demo_ack"
                              and e.get("kind") in ("probe_post", "probe_cancel") and e.get("ok")])
    kinds: Dict[str, List[float]] = {}
    for e in events:
        if e.get("event") == "demo_ack" and "rtt_ms" in e and not str(e.get("kind", "")).startswith("probe"):
            kinds.setdefault(str(e.get("kind")), []).append(e["rtt_ms"])
    report.ack_by_kind = {k: _stats(v) for k, v in kinds.items()}
    report.warm_ms = _stats([e["rtt_ms"] for e in events if e.get("event") == "warm" and e.get("ok")])
    foreign = [e for e in events if e.get("event") == "foreign_position"]
    if foreign:
        report.problems.append("account held {} contracts this run did not open (left alone)".format(
            foreign[-1].get("contracts")))
    report.demo_orders = sum(1 for e in events if e.get("event") == "demo_ack")
    report.demo_fills = sum(1 for e in events if e.get("event") == "demo_order" and e.get("state") == "filled")
    # A post_only that would cross is cancelled by the venue on arrival, and the
    # demo book's ask often sits under production's (it is a different book), so
    # our later cancel of it is refused with 102068. Both eight-hour Tokyo runs
    # (2026-09-13) showed 15 such refusals; they are the demo book disagreeing
    # with the production price, not a fault in the order path.
    cancelled_ok = {e.get("cid") for e in events if e.get("event") == "demo_ack"
                    and e.get("kind") == "cancel" and e.get("ok")}
    crossed = {e.get("cid") for e in events if e.get("event") == "demo_order" and e.get("kind") == "post"
               and e.get("state") == "canceled" and e.get("cid") not in cancelled_ok}
    report.demo_crossed = len(crossed)
    rejected = [e for e in events if e.get("event") == "demo_ack" and not e.get("ok", True)
                and not (e.get("kind") == "cancel" and e.get("cid") in crossed)]
    if rejected:
        first = rejected[0]
        report.problems.append("{} demo orders rejected, first: {} code {} {}".format(
            len(rejected), first.get("kind", "?"), first.get("code", "?"),
            first.get("msg") or "(no message)"))
    orphans = [e for e in events if e.get("event") == "demo_orphan"]
    if orphans:
        report.problems.append("{} demo entries filled after the paper side cancelled (closed reduce-only)".format(
            len(orphans)))
    open_paper = [e for e in events if e.get("event") == "paper_fill"]
    report.open_fills = max(len(open_paper) - len(fills), 0)
    if report.open_fills and report.stopped:
        report.problems.append("{} paper fills never closed (run ended holding)".format(report.open_fills))
    elif report.open_fills:     # read mid-run, or the process died holding: `run_state` tells which
        report.problems.append("{} paper fills not closed at the last row".format(report.open_fills))
    return report


# ---- every run at once ------------------------------------------------------

# A quoting run writes a row at least once a minute (`quoter_stats`), and in
# practice more than once a second (feed lag samples: 1.3 rows/s on IOST, the
# quietest pair of 2026-09-13). Three minutes of nothing is not a quiet market.
SILENT_AFTER_MS = 180_000


def run_state(report: MonitorReport, now_ms: int, silent_after_ms: int = SILENT_AFTER_MS) -> str:
    """`stopped` (a stop row: the runner exited on its own terms, Ctrl-C included),
    `live` (written to recently), or `silent`: no stop row and nothing written,
    which is a killed process, a dead box, or a hung loop."""
    if report.stopped:
        return "stopped"
    if report.last_ms is not None and now_ms - report.last_ms <= silent_after_ms:
        return "live"
    return "silent"


def _net(mean: float, se: float, n: int) -> str:
    if not n:
        return "-"
    if n < 2 or se != se:
        return "{:+.2f}".format(mean)
    return "{:+.2f} ±{:.2f}".format(mean, se)


def _p50(stats: Dict[str, float]) -> str:
    return "{:.0f}".format(stats["p50"]) if stats else "-"


def _utc(ms: Optional[int], fmt: str = "%m-%d %H:%M") -> str:
    return time.strftime(fmt, time.gmtime(ms / 1000.0)) if ms is not None else "?"


def board(reports: Sequence[MonitorReport], now_ms: int,
          silent_after_ms: int = SILENT_AFTER_MS) -> List[str]:
    """The newest run of each pair (is it up, is it healthy), then every run
    pooled per pair and across pairs (the scoreboard).

    Pooled net is the mean over FILLS, not a mean of run means: an 8-hour run
    with 13 fills outweighs a 10-minute one with 1, and fills are the unit the
    S3 decision is taken in (STRATEGIES.md). The SE treats fills as
    independent, as `summarise` does; they are minutes apart.
    """
    by_pair: Dict[str, List[MonitorReport]] = {}
    for report in sorted(reports, key=lambda r: (r.inst_id, r.started_ms or 0, r.path)):
        by_pair.setdefault(report.inst_id, []).append(report)
    latest = [runs[-1] for runs in by_pair.values()]
    states = {id(r): run_state(r, now_ms, silent_after_ms) for r in reports}
    live = sum(1 for r in reports if states[id(r)] == "live")
    out = ["lead quote board, {} UTC: {} runs over {} pairs, {} live".format(
        _utc(now_ms, "%Y-%m-%d %H:%M:%S"), len(reports), len(by_pair), live), ""]

    row = "{:<13} {:<11} {:<5} {:<7} {:>5} {:>5} {:>5} {:>13} {:>5} {:>4} {:>4} {:>6} {:>4}"
    out.append("NEWEST RUN PER PAIR")
    out.append(row.format("PAIR", "STARTED", "MODE", "STATE", "HOURS", "POSTS", "FILLS",
                          "NET bps/fill", "MAKER", "GAPS", "LEAD", "FOLLOW", "ACK"))
    for r in latest:
        state = states[id(r)]
        if state == "live" and r.open_fills:
            state = "holding"
        out.append(row.format(
            r.inst_id, _utc(r.started_ms), "paper" if r.dry_run else "demo", state,
            "{:.1f}".format(r.hours), r.posts, r.paper_fills, _net(r.net_bps, r.net_se, r.paper_fills),
            "{:.0%}".format(r.passive_share) if r.paper_fills else "-",
            "-" if r.gap_episodes is None else r.gap_episodes,
            _p50(r.leader_lag_ms), _p50(r.follower_lag_ms), _p50(r.ack_ms)))

    row = "{:<13} {:>4} {:>6} {:>5} {:>5} {:>5} {:>13} {:>5} {:>8} {:>5}"
    out += ["", "EVERY RUN, POOLED"]
    out.append(row.format("PAIR", "RUNS", "HOURS", "POSTS", "FILLS", "/DAY", "NET bps/fill", "t",
                          "SUM bps", "MAKER"))

    def pooled(name: str, runs: Sequence[MonitorReport]) -> str:
        nets = [n for r in runs for n in r.fill_nets]
        hours = sum(r.hours for r in runs)
        mean, se = mean_se(nets)
        maker = sum(r.passive_share * r.paper_fills for r in runs if r.paper_fills)
        return row.format(
            name, len(runs), "{:.1f}".format(hours), sum(r.posts for r in runs), len(nets),
            "{:.0f}".format(len(nets) / hours * 24.0) if hours > 0 else "-",
            _net(mean, se, len(nets)),
            "{:+.1f}".format(mean / se) if len(nets) > 1 and se > 1e-9 else "-",
            "{:+.1f}".format(sum(nets)) if nets else "-",
            "{:.0%}".format(maker / len(nets)) if nets else "-")

    for name, runs in by_pair.items():
        out.append(pooled(name, runs))
    out.append(pooled("ALL", list(reports)))

    flagged = []
    for r in latest:
        state = states[id(r)]
        if state == "silent":
            flagged.append("{} {}: silent since {} UTC, no stop row (killed, or hung)".format(
                r.inst_id, _utc(r.started_ms), _utc(r.last_ms, "%H:%M:%S")))
        flagged += ["{} {}: {}".format(r.inst_id, _utc(r.started_ms), p) for p in r.problems
                    if not (state == "live" and p.endswith("not closed at the last row"))]  # "holding" says it
    if flagged:
        out += ["", "PROBLEMS (newest runs)"] + ["  " + line for line in flagged]
    out += ["",
            "GAPS: leader moves past the edge (quoter_stats). LEAD / FOLLOW: feed lag p50 ms. "
            "ACK: demo order ack p50 ms.",
            "t: pooled net over its SE. SUM: net bps summed over fills, on one order's notional."]
    return out


# ---- following the logs as they are written ---------------------------------

class LogTail:
    """Rows appended to run logs since the last poll.

    Reads the JSONL rather than the runner's console: `RunLog` flushes every
    row, and Python block-buffers stdout that is redirected to a file. On the
    Tokyo box (2026-09-13) `lq_SUI-USDT.out` held only the SDK's stderr lines
    eight hours into a run with six paper fills; the `paper fill:` prints were
    still in the process's buffer. A row caught mid-write is held back until
    its newline arrives. Files appearing later (a new run) are read from the
    start.
    """

    def __init__(self, discover: Callable[[], Iterable[Path]]):
        self.discover = discover
        self.offsets: Dict[Path, int] = {}
        self.partial: Dict[Path, bytes] = {}

    def poll(self) -> List[Tuple[Path, Dict]]:
        rows: List[Tuple[Path, Dict]] = []
        for path in sorted(set(self.discover())):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            offset = self.offsets.get(path, 0)
            if size < offset:                   # replaced or truncated: read it again
                offset = 0
                self.partial.pop(path, None)
            if size == offset:
                continue
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read(size - offset)
            self.offsets[path] = offset + len(chunk)
            pieces = (self.partial.pop(path, b"") + chunk).split(b"\n")
            if pieces[-1]:
                self.partial[path] = pieces[-1]
            for piece in pieces[:-1]:
                try:
                    row = json.loads(piece)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append((path, row))
        rows.sort(key=lambda item: item[1].get("t", 0))     # stable: a file's own order is kept
        return rows


@dataclass
class _Run:
    inst_id: str
    last_ms: Optional[int] = None
    stopped: bool = False
    silent: bool = False
    crossed: Set[str] = field(default_factory=set)
    rejections: Dict[Tuple[str, str], int] = field(default_factory=dict)


def _side(side) -> str:
    return "long" if side == 1 else "short"


def _price(value) -> str:
    try:
        return "{:.10g}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


class Narrator:
    """Log rows in, the lines worth watching out: `(level, text)`.

    Levels are `win`, `loss`, `fill`, `warn` and `info`, for the caller to
    colour. Keeps a running tally of closed paper fills per pair so every exit
    line carries the scoreboard it just moved. Nothing here reads a clock:
    silence is judged against the `now_ms` the caller passes.
    """

    def __init__(self, show_intents: bool = False, silent_after_ms: int = SILENT_AFTER_MS):
        self.show_intents = show_intents
        self.silent_after_ms = silent_after_ms
        self.runs: Dict[Path, _Run] = {}
        self.nets: Dict[str, List[float]] = {}

    def _run(self, path: Path, row: Dict) -> _Run:
        run = self.runs.get(path)
        if run is None:
            run = self.runs[path] = _Run(row.get("inst_id") or Path(path).parent.parent.name)
        return run

    def tally(self, inst_id: Optional[str] = None) -> str:
        nets = self.nets.get(inst_id, []) if inst_id else [n for v in self.nets.values() for n in v]
        mean, se = mean_se(nets)
        return "{} {} fills {}".format(inst_id or "all", len(nets), _net(mean, se, len(nets)))

    def feed(self, path: Path, row: Dict) -> List[Tuple[str, str]]:
        run = self._run(path, row)
        if "t" in row:
            run.last_ms = int(row["t"])
        out: List[Tuple[str, str]] = []

        def say(level: str, text: str) -> None:
            out.append((level, "{}  {:<13} {}".format(_utc(run.last_ms, "%H:%M:%S"), run.inst_id, text)))

        if run.silent:
            run.silent = False
            say("info", "writing again")
        event = row.get("event")
        if event == "paper_fill":
            say("fill", "FILL  {} @ {}, {:.1f} s on the book".format(
                _side(row.get("side")), _price(row.get("entry")), float(row.get("wait_ms") or 0) / 1000.0))
        elif event == "paper_fill_closed":
            net = float(row.get("net_bps") or 0.0)
            self.nets.setdefault(run.inst_id, []).append(net)
            say("win" if net > 0 else "loss", "EXIT  {} {} -> {}  {:+.2f} bps  {}, held {:.1f} s  | {} | {}".format(
                _side(row.get("side")), _price(row.get("entry")), _price(row.get("exit")), net,
                "maker exit" if row.get("passive_exit") else "crossed out",
                float(row.get("hold_ms") or 0) / 1000.0, self.tally(run.inst_id), self.tally()))
        elif event == "demo_order":
            if row.get("kind") == "post" and row.get("state") == "canceled":
                run.crossed.add(str(row.get("cid")))
            if row.get("state") == "filled":
                say("info", "demo {} filled: {} @ {}, fee {}".format(
                    "entry" if row.get("kind") == "post" else row.get("kind"), row.get("filled_size"),
                    _price(row.get("avg_price")), _price(row.get("fee"))))
        elif event == "demo_ack" and not row.get("ok", True):
            if row.get("kind") == "cancel" and str(row.get("cid")) in run.crossed:
                pass            # the venue already cancelled a post_only that would cross the demo book
            else:
                key = (str(row.get("kind")), str(row.get("code")))
                count = run.rejections[key] = run.rejections.get(key, 0) + 1
                if count == 1 or count % 10 == 0:
                    say("warn", "demo {} rejected{}: code {} {}".format(
                        key[0], "" if count == 1 else " ({} times)".format(count), key[1],
                        row.get("msg") or "(no message)"))
        elif event == "demo_orphan":
            say("warn", "demo entry filled after the paper cancel: closing {} contracts reduce-only".format(
                row.get("contracts")))
        elif event == "foreign_position":
            say("warn", "demo account holds {} contracts this run did not open (left alone)".format(
                row.get("contracts")))
        elif event == "plan" and not row.get("ok", True):
            say("warn", "plan refused: " + "; ".join(row.get("reasons") or []))
        elif event == "start":
            say("info", "run started ({})".format("paper only" if row.get("dry_run") else "demo mirror"))
        elif event == "stop":
            run.stopped = True
            say("info", "run stopped: {} posts, {} paper fills".format(row.get("posts"), row.get("paper_fills")))
        elif event == "intent" and self.show_intents:
            say("info", "{} {} @ {} ({})".format(row.get("kind"), _side(row.get("side")),
                                                _price(row.get("price")), row.get("reason")))
        return out

    def check_silence(self, now_ms: int) -> List[Tuple[str, str]]:
        """Runs that went quiet without a stop row, each said once."""
        out = []
        for run in self.runs.values():
            if run.stopped or run.silent or run.last_ms is None:
                continue
            if now_ms - run.last_ms > self.silent_after_ms:
                run.silent = True
                out.append(("warn", "{}  {:<13} SILENT: nothing written for {:.1f} min and no stop row "
                                    "(killed, or hung)".format(_utc(now_ms, "%H:%M:%S"), run.inst_id,
                                                               (now_ms - run.last_ms) / 60000.0)))
        return out
