"""Whether an instrument and a connection can carry the lead quote at all. SENDS NOTHING.

The gates are the two ways step 9ae's result died. ADA, whose tick is 4.8
bps, lost -3 a fill on every day: one tick is a whole spread there, and
"fair rounded to the tick" is not a price. And at 500 ms of feed latency the
alts went negative while 150 ms paid, so the leader's lag as measured on
THIS machine is a gate, not a footnote. Both are refusals with the number
that failed; a lag between the two thresholds is a warning, because
measuring it is the point of running the paper quoter at all.

Every reason is returned, not the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

MAX_TICK_BPS = 2.5
LAG_WARN_MS = 200.0
LAG_REFUSE_MS = 350.0


@dataclass
class QuotePlan:
    inst_id: str
    tick_bps: float = float("nan")
    spread_bps: float = float("nan")
    leader_lag_ms: Optional[float] = None
    follower_lag_ms: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


def plan_quote(inst_id: str, *, tick: float, bid: float, ask: float,
               leader_lag_ms: Optional[float] = None,
               follower_lag_ms: Optional[float] = None) -> QuotePlan:
    plan = QuotePlan(inst_id)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    if mid <= 0 or ask <= bid:
        plan.reasons.append("no usable quote (bid {} ask {})".format(bid, ask))
        return plan
    plan.tick_bps = tick / mid * 1e4
    plan.spread_bps = (ask - bid) / mid * 1e4
    if plan.tick_bps > MAX_TICK_BPS:
        plan.reasons.append(
            "tick is {:.2f} bps of price, above {:.1f}: one tick is a whole spread here, "
            "and 9ae lost on ADA at 4.8".format(plan.tick_bps, MAX_TICK_BPS))
    if plan.spread_bps < 2 * plan.tick_bps:
        plan.warnings.append(
            "spread {:.2f} bps is under two ticks; the order is rarely alone at its level".format(
                plan.spread_bps))
    for name, lag in (("leader (Binance)", leader_lag_ms), ("follower (BloFin)", follower_lag_ms)):
        if lag is None:
            plan.warnings.append(name + " feed lag not measured yet")
        elif lag > LAG_REFUSE_MS:
            plan.reasons.append(
                "{} feed lag {:.0f} ms, above {:.0f}: 9ae was negative at 500 ms; "
                "this host is too far from the venues".format(name, lag, LAG_REFUSE_MS))
        elif lag > LAG_WARN_MS:
            plan.warnings.append(
                "{} feed lag {:.0f} ms, above {:.0f}: the study paid at 150".format(name, lag, LAG_WARN_MS))
        plan.leader_lag_ms = leader_lag_ms
        plan.follower_lag_ms = follower_lag_ms
    return plan
