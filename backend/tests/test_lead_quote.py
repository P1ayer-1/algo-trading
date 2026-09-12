"""The lead-quote state machine, gate and monitor, against hand-computed cases.

The numbers here are the backtest's rules (README 9ae) applied by hand to a
book with a 0.001 tick: what a fill nets, where the fair exit sits, when the
leader stop crosses out. If `quoter.py` drifts from `venue_lag_passive.py`,
the paper quoter would be measuring a different strategy from the one that
was backtested, and these are what say so.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from trading.strategies import Plan, Report
from trading.strategies.lead_quote import QuoteConfig, Quoter, plan_quote, summarise
from trading.strategies.lead_quote.execute import LeadQuoteRunner, RunLog

TICK = 0.001


def quoter(**overrides) -> Quoter:
    cfg = QuoteConfig(tick=TICK, edge_bps=7.0, stop_bps=3.0, order_ttl_ms=10_000,
                      hold_ms=120_000, maker_bps=0.6, taker_bps=5.0)
    q = Quoter(cfg)
    for key, value in overrides.items():
        setattr(q, key, value)
    return q


def kinds(intents):
    return [(i.kind, i.side) for i in intents]


def test_posts_a_bid_one_tick_under_the_ask_when_alone_and_the_leader_is_past_it():
    """ask 1.003, bid 1.000: 1.002 is empty. Leader 1.0030 is 9.98 bps above it, over the 7 bps edge."""
    q = quoter()
    assert q.on_book(0, 1.000, 1.003) == []           # no leader yet
    intents = q.on_leader(1, 1.0030)
    assert kinds(intents) == [("post", 1)]
    assert intents[0].price == pytest.approx(1.002)


def test_does_not_post_when_the_level_would_join_the_bid():
    """A one-tick spread leaves ask - tick equal to the bid: a queue, not a level of one's own."""
    q = quoter()
    q.on_book(0, 1.000, 1.001)
    assert q.on_leader(1, 1.0030) == []


def test_does_not_post_under_the_edge():
    q = quoter()
    q.on_book(0, 1.000, 1.003)
    assert q.on_leader(1, 1.0025) == []                # 4.99 bps over 1.002, under 7


def test_a_sell_print_at_the_bid_fills_and_the_exit_is_posted_at_fair():
    """Fill at 1.002; leader 1.0030 rounds up to 1.003 = entry + tick: the exit goes there."""
    q = quoter()
    q.on_book(0, 1.000, 1.003)
    q.on_leader(1, 1.0030)
    intents = q.on_trade(500, 1.002, "sell")
    assert len(q.fills) == 1 and q.fills[0].entry == pytest.approx(1.002)
    assert q.fills[0].wait_ms == 499
    assert kinds(intents) == [("exit_post", 1)]
    assert intents[0].price == pytest.approx(1.003)


def test_a_buy_print_at_the_exit_closes_passively_for_the_move_less_two_maker_fees():
    """(1.003 - 1.002) / 1.002 = 9.98 bps, minus 0.6 twice = 8.78. The exit sits inside
    the spread (ask 1.004), so it is first at its level and the print fills it."""
    q = quoter()
    q.on_book(0, 1.000, 1.003)
    q.on_leader(1, 1.0030)
    q.on_trade(500, 1.002, "sell")
    q.on_book(600, 1.002, 1.004)
    q.on_trade(700, 1.003, "buy")
    closed = q.closed_fills()
    assert len(closed) == 1
    assert closed[0].passive_exit
    assert closed[0].net_bps == pytest.approx(9.98, abs=0.01) or closed[0].net_bps == pytest.approx(8.78, abs=0.01)
    assert closed[0].net_bps == pytest.approx((1.003 - 1.002) / 1.002 * 1e4 - 1.2, abs=1e-6)


def test_a_print_does_not_fill_an_exit_that_is_only_joining_the_touch():
    """Pessimistic queue: with the ask at 1.003 the exit at 1.003 is in a queue, and a print
    there is somebody else's fill."""
    q = quoter()
    q.on_book(0, 1.000, 1.003)
    q.on_leader(1, 1.0030)
    q.on_trade(500, 1.002, "sell")
    q.on_trade(700, 1.003, "buy")
    assert q.closed_fills() == []


def test_the_leader_stop_crosses_out_at_the_touch():
    """Leader falls to 1.0015: 4.99 bps through a 1.002 entry, past the 3 bps stop. Exit at
    the bid 1.000: (1.000 - 1.002) / 1.002 = -19.96 bps, minus maker 0.6 and taker 5.0."""
    q = quoter()
    q.on_book(0, 1.000, 1.003)
    q.on_leader(1, 1.0030)
    q.on_trade(500, 1.002, "sell")
    intents = q.on_leader(800, 1.0015)
    assert kinds(intents) == [("exit_cross", 1)]
    closed = q.closed_fills()
    assert not closed[0].passive_exit
    assert closed[0].net_bps == pytest.approx(-19.96 - 5.6, abs=0.01)


def test_the_hold_limit_crosses_out():
    q = quoter()
    q.on_book(0, 1.000, 1.003)
    q.on_leader(1, 1.0030)
    q.on_trade(500, 1.002, "sell")
    assert kinds(q.on_clock(500 + 120_000)) == [("exit_cross", 1)]


def test_an_order_is_cancelled_when_the_bid_overtakes_it():
    q = quoter()
    q.on_book(0, 1.000, 1.003)
    q.on_leader(1, 1.0030)
    intents = q.on_book(50, 1.0025, 1.003)
    assert ("cancel", 1) in kinds(intents)
    assert q.orders[+1] is None


def test_an_order_is_cancelled_when_the_leader_comes_back_or_the_ttl_passes():
    q = quoter()
    q.on_book(0, 1.000, 1.003)
    q.on_leader(1, 1.0030)
    back = q.on_leader(2, 1.0015)
    assert [(i.kind, i.reason) for i in back] == [("cancel", "leader came back")]
    q.on_leader(3, 1.0030)                              # re-posted
    assert q.orders[+1] is not None
    stale = q.on_clock(3 + 10_001)
    # The stale order goes, and with the leader still past the level a fresh one
    # is posted in the same decision: the backtest re-posts on the next event too.
    assert [(i.kind, i.reason) for i in stale][0] == ("cancel", "ttl")
    assert kinds(stale) == [("cancel", 1), ("post", 1)]


def test_the_ask_side_mirrors_the_bid_side():
    """bid 1.000 -> post an ask at 1.001; leader 0.9990 is 10 bps below it."""
    q = quoter()
    q.on_book(0, 1.000, 1.003)
    intents = q.on_leader(1, 0.9990)
    assert kinds(intents) == [("post", -1)]
    assert intents[0].price == pytest.approx(1.001)
    exit_intents = q.on_trade(100, 1.001, "buy")
    assert q.fills[0].side == -1
    assert kinds(exit_intents) == [("exit_post", -1)]
    assert exit_intents[0].price == pytest.approx(0.999)   # min(entry - tick = 1.000, floor(fair) = 0.999)


def test_the_gate_lists_every_refusal():
    """ADA-shaped: a 0.0001 tick at 0.20 is 5 bps, and a 500 ms feed is what 9ae died at."""
    plan = plan_quote("ADA-USDT", tick=0.0001, bid=0.2000, ask=0.2001,
                      leader_lag_ms=500.0, follower_lag_ms=150.0)
    assert not plan.ok
    assert len(plan.reasons) == 2
    assert plan.tick_bps == pytest.approx(5.0, abs=0.01)
    assert isinstance(plan, Plan)


def test_the_gate_warns_between_the_thresholds_and_passes_a_fine_tick():
    plan = plan_quote("SUI-USDT", tick=0.0001, bid=0.7300, ask=0.7302,
                      leader_lag_ms=250.0, follower_lag_ms=120.0)
    assert plan.ok
    assert any("250 ms" in w for w in plan.warnings)


def test_no_order_path_from_the_state_machine_the_gate_or_the_monitor():
    """The runner is the only file allowed to know a broker exists."""
    root = Path(__file__).resolve().parent.parent / "trading" / "strategies" / "lead_quote"
    for name in ("quoter.py", "plan.py", "monitor.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "placeOrder" not in source and "place_limit" not in source, name
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any("broker" in n or "blofin" in n or "execute" in n for n in names), (name, names)


def test_the_monitor_reads_a_run_back(tmp_path):
    path = tmp_path / "run.jsonl"
    rows = [
        {"t": 0, "event": "start", "inst_id": "SUI-USDT"},
        {"t": 1000, "event": "lag", "feed": "leader", "lag_ms": 120.0},
        {"t": 2000, "event": "lag", "feed": "leader", "lag_ms": 180.0},
        {"t": 2500, "event": "intent", "kind": "post", "side": 1},
        {"t": 3000, "event": "paper_fill", "side": 1, "entry": 1.0},
        {"t": 4000, "event": "paper_fill_closed", "side": 1, "net_bps": 4.0, "passive_exit": True},
        {"t": 5000, "event": "intent", "kind": "post", "side": -1},
        {"t": 6000, "event": "paper_fill", "side": -1, "entry": 1.0},
        {"t": 7000, "event": "paper_fill_closed", "side": -1, "net_bps": -2.0, "passive_exit": False},
        {"t": 7200, "event": "demo_ack", "kind": "post", "cid": "x", "ok": False, "msg": "152002", "rtt_ms": 90.0},
        {"t": 3_600_000, "event": "stop"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report = summarise(path)
    assert isinstance(report, Report)
    assert report.paper_fills == 2 and report.posts == 2
    assert report.net_bps == pytest.approx(1.0)
    assert report.passive_share == pytest.approx(0.5)
    assert report.leader_lag_ms["p50"] == pytest.approx(150.0)
    assert report.fills_per_day == pytest.approx(48.0)
    assert report.problems and "rejected" in report.problems[0]


class _Broker:
    """Records calls; answers like the venue does."""

    def __init__(self):
        self.calls = []

    def place_limit(self, **kw):
        self.calls.append(("limit", kw))
        return {"code": "0", "data": [{"orderId": "o" + str(len(self.calls)), "code": "0", "msg": ""}]}

    def place_market(self, **kw):
        self.calls.append(("market", kw))
        return {"code": "0", "data": [{"orderId": "o" + str(len(self.calls)), "code": "0", "msg": ""}]}

    def cancel(self, **kw):
        self.calls.append(("cancel", kw))
        return {"code": "0", "data": [{"code": "0", "msg": ""}]}

    def positions(self):
        return []


def test_the_runner_mirrors_a_post_and_cancels_it_rather_than_exiting_an_unfilled_demo_order(tmp_path):
    """The tape fills the paper order; the demo book has not. A reduce-only exit would be
    rejected, so the runner cancels the resting demo entry and logs the skip."""
    import asyncio
    broker = _Broker()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 1.0030))       # post
        await runner.handle(runner.quoter.on_trade(500, 1.002, "sell"))  # paper fill -> exit_post
    asyncio.run(go())
    kinds_sent = [c[0] for c in broker.calls]
    assert kinds_sent == ["limit", "cancel"]
    assert broker.calls[0][1]["price"] == "1.002" and broker.calls[0][1]["reduce_only"] is False
    log.close()
    events = [json.loads(l)["event"] for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert "paper_fill" in events and "demo_skip" in events
    assert runner.dry_run is False


def test_shutdown_closes_only_what_this_run_filled_and_leaves_a_foreign_position_alone(tmp_path):
    """The first live run closed a 1,677-contract demo short it never opened. Now: the
    account holds -1677, this run's own fills net +3, and the only order sent is a
    reduce-only sell of 3."""
    import asyncio
    from decimal import Decimal

    class _Held(_Broker):
        def positions(self):
            return [{"instId": "SUI-USDT", "positions": "-1674"}]     # -1677 foreign + 3 ours

    broker = _Held()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    runner.demo_net = Decimal("3")
    asyncio.run(runner.shutdown_demo())
    log.close()
    assert [c[0] for c in broker.calls] == ["market"]
    sent = broker.calls[0][1]
    assert sent["side"] == "sell" and sent["size"] == Decimal("3") and sent["reduce_only"] is True
    rows = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    foreign = [r for r in rows if r["event"] == "foreign_position"]
    assert foreign and foreign[0]["contracts"] == "-1677"
    report = summarise(tmp_path / "run.jsonl")
    assert report.critical and "did not open" in report.problems[0]


def test_shutdown_with_nothing_filled_sends_nothing(tmp_path):
    import asyncio

    class _Held(_Broker):
        def positions(self):
            return [{"instId": "SUI-USDT", "positions": "-1677"}]

    broker = _Held()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    asyncio.run(runner.shutdown_demo())
    log.close()
    assert broker.calls == []


def test_the_runner_without_a_broker_sends_nothing_and_is_a_dry_run(tmp_path):
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log)
    assert runner.dry_run
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)
    import asyncio
    asyncio.run(runner.handle(runner.quoter.on_leader(1, 1.0030)))
    log.close()
    rows = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["intent"] and rows[0]["kind"] == "post"
