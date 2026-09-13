"""Read a lead-quote run back from its log and score it. READ ONLY.

The runner writes one JSON object per line to `data/<INST>/lead_quote/`:
feed latency samples, every intent, every paper fill with its exit, and the
demo order acknowledgements when orders were mirrored. This turns that into
the numbers step 9ae was measured in - fills, net bps per fill, passive-exit
share - beside the two numbers the backtest could only assume: the leader
and follower feed lag on this host, and the order round trip to the venue.

No broker, no network. Nothing here can send anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
    demo_orders: int = 0
    demo_fills: int = 0
    problems: List[str] = field(default_factory=list)

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
        for name, stats in (("leader feed lag", self.leader_lag_ms),
                            ("follower feed lag", self.follower_lag_ms),
                            ("order ack round trip", self.ack_ms),
                            ("probe post+cancel round trip", self.probe_ms)):
            if stats:
                out.append("  {}: p50 {:.0f} ms, p90 {:.0f}, p99 {:.0f} (n {})".format(
                    name, stats["p50"], stats["p90"], stats["p99"], int(stats["n"])))
        if self.demo_orders:
            out.append("  demo: {} orders sent, {} reported filled".format(self.demo_orders, self.demo_fills))
        for problem in self.problems:
            out.append("  PROBLEM: " + problem)
        return out


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)), "n": float(len(arr))}


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
    if not events:
        report.problems.append("empty log")
        return report
    times = [e["t"] for e in events if "t" in e]
    if times:
        report.hours = (max(times) - min(times)) / 3.6e6
    report.posts = sum(1 for e in events if e.get("event") == "intent" and e.get("kind") == "post")
    fills = [e for e in events if e.get("event") == "paper_fill_closed"]
    report.paper_fills = len(fills)
    if fills:
        nets = np.array([e["net_bps"] for e in fills], dtype=float)
        report.net_bps = float(nets.mean())
        report.net_se = float(nets.std(ddof=1) / np.sqrt(len(nets))) if len(nets) > 1 else float("nan")
        report.passive_share = float(np.mean([1.0 if e.get("passive_exit") else 0.0 for e in fills]))
        report.fills_per_day = len(fills) / max(report.hours, 1e-9) * 24.0
    report.leader_lag_ms = _stats([e["lag_ms"] for e in events if e.get("event") == "lag" and e.get("feed") == "leader"])
    report.follower_lag_ms = _stats([e["lag_ms"] for e in events if e.get("event") == "lag" and e.get("feed") == "follower"])
    report.ack_ms = _stats([e["rtt_ms"] for e in events if e.get("event") == "demo_ack" and "rtt_ms" in e])
    report.probe_ms = _stats([e["rtt_ms"] for e in events if e.get("event") == "demo_ack"
                              and e.get("kind") in ("probe_post", "probe_cancel") and e.get("ok")])
    foreign = [e for e in events if e.get("event") == "foreign_position"]
    if foreign:
        report.problems.append("account held {} contracts this run did not open (left alone)".format(
            foreign[-1].get("contracts")))
    report.demo_orders = sum(1 for e in events if e.get("event") == "demo_ack")
    report.demo_fills = sum(1 for e in events if e.get("event") == "demo_order" and e.get("state") == "filled")
    rejected = [e for e in events if e.get("event") == "demo_ack" and not e.get("ok", True)]
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
    if len(open_paper) > len(fills):
        report.problems.append("{} paper fills never closed (run ended holding)".format(
            len(open_paper) - len(fills)))
    return report
