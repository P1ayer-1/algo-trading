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


def test_the_quoter_counts_leader_moves_past_the_edge_whether_or_not_it_can_post():
    """XRP posted 0 times in 1.55 h (2026-09-13). Two explanations, told apart here: the
    leader never got 7 bps past ask - tick, or it did while the level was occupied.
    Book 1.000/1.003, leader 1.0030: gap (1.003 - 1.002)/1.002 = 9.98 bps, alone -> one
    episode, posted. Leader 1.0015: cancelled, flag clears. Leader 1.0025 is 4.99 bps over
    1.002, under the edge on both sides (the ask side's price is bid + tick). The bid then
    rises to 1.002 (one-tick spread) - both sides still 4.99 bps - and the leader returns to
    1.0030: a second episode, blocked because ask - tick == bid; the ask side's gap is 0.
    Leader 1.0025 clears it; book back to 1.000/1.003; leader 1.0030: a third, posted."""
    q = Quoter(QuoteConfig(tick=TICK))
    q.on_book(0, 1.000, 1.003)
    out = q.on_leader(1, 1.0030)
    assert [i.kind for i in out] == ["post"]
    assert (q.gap_episodes, q.gap_blocked) == (1, 0) and q.max_gap_bps == pytest.approx(9.98, abs=0.01)
    out = q.on_leader(2, 1.0015)
    assert [i.kind for i in out] == ["cancel"]
    assert q.on_leader(3, 1.0025) == [] and q.gap_episodes == 1
    assert q.on_book(4, 1.002, 1.003) == [] and q.gap_episodes == 1
    out = q.on_leader(5, 1.0030)
    assert out == [] and (q.gap_episodes, q.gap_blocked) == (2, 1) and q.posts == 1
    assert q.on_leader(6, 1.0025) == []
    assert q.on_book(7, 1.000, 1.003) == []
    out = q.on_leader(8, 1.0030)
    assert [i.kind for i in out] == ["post"] and (q.gap_episodes, q.gap_blocked) == (3, 1)


def test_a_leader_flicker_inside_flicker_ms_posts_nothing_and_a_held_move_posts_when_due():
    """17 of the first 137 "leader came back" cancels (2026-09-13) came within 2 ms of their
    post: a Binance quote that jumped and reverted. With flicker_ms 5, book 1.000/1.003:
    leader 1.0030 at t=10 is 9.98 bps past 1.002 - pending, due at 15. Back to 1.0015 at
    t=12, 2 ms in: one flicker, nothing sent. Past again at t=20: due 25; the clock at 24 is
    early, at 25 it posts 1.002. Both moves were episodes."""
    q = Quoter(QuoteConfig(tick=TICK, flicker_ms=5))
    q.on_book(0, 1.000, 1.003)
    assert q.on_leader(10, 1.0030) == [] and q.wake_at == 15 and q.posts == 0
    assert q.on_leader(12, 1.0015) == [] and q.flickers == 1 and q.wake_at is None
    assert q.on_leader(20, 1.0030) == [] and q.wake_at == 25
    assert q.on_clock(24) == []
    out = q.on_clock(25)
    assert kinds(out) == [("post", 1)] and out[0].price == pytest.approx(1.002)
    assert (q.flickers, q.gap_episodes, q.posts) == (1, 2, 1)


def test_a_print_beyond_a_resting_entry_cancels_it_before_the_book_batch_and_withholds_the_repost():
    """BloFin's book comes in 100 ms batches; its prints do not. Bid resting at 1.002 (book
    1.000/1.003, leader 1.0030). A SELL print at 1.003 means a bid stood above ours, so ours
    is no longer first: cancelled at once, and the repost at ask - tick = 1.002 is withheld
    because the print is through that level. The next book clears the print. The ask side
    mirrors it: ask resting at 1.001 (leader 0.9990), a BUY print at 1.000 cancels it. With
    trade_watch off - the backtest's rule - the same print does nothing."""
    q = Quoter(QuoteConfig(tick=TICK, trade_watch=True))
    q.on_book(0, 1.000, 1.003)
    q.on_leader(1, 1.0030)
    out = q.on_trade(50, 1.003, "sell")
    assert [(i.kind, i.side, i.reason) for i in out] == [("cancel", 1, "print beyond the level")]
    assert q.orders[+1] is None and (q.trade_cancels, q.trade_blocked) == (1, 1) and q.fills == []
    assert kinds(q.on_book(100, 1.000, 1.003)) == [("post", 1)]           # the batch disagrees: post again

    asks = Quoter(QuoteConfig(tick=TICK, trade_watch=True))
    asks.on_book(0, 1.000, 1.003)
    assert kinds(asks.on_leader(1, 0.9990)) == [("post", -1)]
    assert kinds(asks.on_trade(50, 1.000, "buy")) == [("cancel", -1)]

    backtest = quoter()
    backtest.on_book(0, 1.000, 1.003)
    backtest.on_leader(1, 1.0030)
    assert backtest.on_trade(50, 1.003, "sell") == [] and backtest.orders[+1].price == pytest.approx(1.002)


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
        {"t": 3_599_000, "event": "quoter_stats", "posts": 2, "gap_episodes": 5, "gap_blocked": 3, "max_gap_bps": 11.4},
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
    assert (report.gap_episodes, report.gap_blocked, report.max_gap_bps) == (5, 3, 11.4)
    assert "  leader past the edge 5 times (max gap 11.4 bps), 3 with the level occupied" in report.lines()


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

    def amend(self, **kw):
        self.calls.append(("amend", kw))       # data is one object for amend-order, per BloFin's docs
        return {"code": "0", "msg": "Order modified",
                "data": {"orderId": kw["order_id"], "code": "0", "msg": "Order modified"}}

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
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_trade(500, 1.002, "sell"))  # paper fill -> exit_post
        await runner.drain_mirror()
    asyncio.run(go())
    kinds_sent = [c[0] for c in broker.calls]
    assert kinds_sent == ["limit", "cancel"]
    assert broker.calls[0][1]["price"] == "1.002" and broker.calls[0][1]["reduce_only"] is False
    log.close()
    events = [json.loads(l)["event"] for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert "paper_fill" in events and "demo_skip" in events
    assert runner.dry_run is False


def test_a_demo_entry_that_fills_after_the_paper_cancel_is_closed_at_once(tmp_path):
    """Tokyo, 2026-09-13: the demo book filled a bid the tape never did, the paper
    side cancelled, the cancel was rejected as already filled, and one contract sat
    open for the rest of the run. Now: post, cancel (nothing filled yet), the fill
    notice arrives late - and a reduce-only market sell of exactly the filled size
    goes out, with a demo_orphan row in the log."""
    import asyncio
    from decimal import Decimal
    broker = _Broker()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK, order_ttl_ms=1000), log=log, broker=broker)
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 1.0030))       # post at 1.002
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_leader(1400, 1.0015))    # leader back -> cancel
        await runner.drain_mirror()
        cid = broker.calls[0][1]["client_order_id"]
        await runner.on_demo_row({"clientOrderId": cid, "orderId": "o1", "state": "filled",
                                  "filledSize": "1"})
    asyncio.run(go())
    log.close()
    assert [c[0] for c in broker.calls] == ["limit", "cancel", "market"]
    close = broker.calls[2][1]
    assert close["side"] == "sell" and close["size"] == Decimal("1") and close["reduce_only"] is True
    assert runner.demo_net == Decimal("1")      # the flatten's own fill notice has not arrived yet
    rows = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert [r for r in rows if r["event"] == "demo_orphan"][0]["contracts"] == "1"
    report = summarise(tmp_path / "run.jsonl")
    assert any("filled after the paper side cancelled" in p for p in report.problems)


def test_a_demo_entry_known_filled_before_the_paper_cancel_is_flattened_not_cancelled(tmp_path):
    """The other ordering: the fill notice arrives while the demo entry is still the
    live one, then the paper side cancels. No cancel is sent (it would be rejected);
    a reduce-only close is."""
    import asyncio
    broker = _Broker()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK, order_ttl_ms=1000), log=log, broker=broker)
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 1.0030))
        await runner.drain_mirror()
        cid = broker.calls[0][1]["client_order_id"]
        await runner.on_demo_row({"clientOrderId": cid, "orderId": "o1", "state": "filled",
                                  "filledSize": "1"})
        assert [c[0] for c in broker.calls] == ["limit"]          # live entry: nothing to close yet
        await runner.handle(runner.quoter.on_leader(1400, 1.0015))
        await runner.drain_mirror()
    asyncio.run(go())
    log.close()
    assert [c[0] for c in broker.calls] == ["limit", "market"]
    assert broker.calls[1][1]["reduce_only"] is True


def test_demo_orders_are_priced_on_the_demo_hosts_tick_and_never_closer_to_the_touch(tmp_path):
    """FIL-USDT, Tokyo, 2026-09-13: production's tick is 0.0001 and the demo host's 0.001, and
    36 of the first 44 demo posts came back 102016 "Precision does not match: 0.001" - every
    one whose price was off the 0.001 grid. The paper quote stays on production's grid; the
    demo order goes on demo's, a buy rounded down and a sell up. Rounding to nearest would
    have sent all three orders here one demo tick more aggressive than the quote.

    Long: book 0.9850 / 0.9868, leader 0.9875 is (0.9875 - 0.9867) / 0.9867 = 8.11 bps past
    ask - tick. Paper bid 0.9867 -> demo buy 0.986 (nearest 0.987). The demo book fills it,
    the leader eases to 0.9872 (still above the bid, so it rests), the tape sells at 0.9867:
    exit at max(entry + tick 0.9868, fair 0.9872) = 0.9872 -> demo sell 0.988 (nearest 0.987).
    Short: book 0.9853 / 0.9870, leader 0.9845 is (0.9854 - 0.9845) / 0.9854 = 9.13 bps under
    bid + tick. Paper ask 0.9854 -> demo sell 0.986 (nearest 0.985)."""
    import asyncio
    from decimal import Decimal
    fil = QuoteConfig(tick=0.0001)

    long_broker, short_broker = _Broker(), _Broker()
    long_log, short_log = RunLog(tmp_path / "long.jsonl"), RunLog(tmp_path / "short.jsonl")
    long_ = LeadQuoteRunner("FIL-USDT", fil, log=long_log, broker=long_broker, demo_tick=Decimal("0.001"))
    short = LeadQuoteRunner("FIL-USDT", fil, log=short_log, broker=short_broker, demo_tick=Decimal("0.001"))
    long_.quoter.on_book(0, 0.9850, 0.9868)
    short.quoter.on_book(0, 0.9853, 0.9870)

    async def go():
        await long_.handle(long_.quoter.on_leader(1, 0.9875))            # post bid
        await long_.drain_mirror()
        cid = long_broker.calls[0][1]["client_order_id"]
        await long_.on_demo_row({"clientOrderId": cid, "orderId": "o1", "state": "filled", "filledSize": "1"})
        assert long_.quoter.on_leader(2, 0.9872) == []
        await long_.handle(long_.quoter.on_trade(500, 0.9867, "sell"))  # paper fill -> exit_post
        await long_.drain_mirror()
        await short.handle(short.quoter.on_leader(1, 0.9845))            # post ask
        await short.drain_mirror()
    asyncio.run(go())
    long_log.close()
    short_log.close()

    paper = [json.loads(l) for l in (tmp_path / "long.jsonl").read_text().splitlines()
             if json.loads(l)["event"] == "intent"]
    assert [(r["kind"], r["price"]) for r in paper] == [("post", pytest.approx(0.9867)),
                                                        ("exit_post", pytest.approx(0.9872))]
    assert [(c[0], c[1]["side"], c[1]["price"], c[1]["reduce_only"]) for c in long_broker.calls] == [
        ("limit", "buy", "0.986", False), ("limit", "sell", "0.988", True)]
    assert [(c[0], c[1]["side"], c[1]["price"]) for c in short_broker.calls] == [("limit", "sell", "0.986")]


def test_the_start_row_names_the_demo_tick(tmp_path):
    """A FIL demo log read back later must say its orders were on a coarser grid than its
    paper quote, or its demo fills read as if they had rested at the paper price. Feeds are
    stubbed; with no quote the plan refuses and the run ends at once."""
    import asyncio
    from decimal import Decimal

    async def silent():
        return None

    runner = LeadQuoteRunner("FIL-USDT", QuoteConfig(tick=0.0001), log=RunLog(tmp_path / "run.jsonl"),
                             broker=_Broker(), warmup_seconds=0, demo_tick=Decimal("0.001"),
                             on_log=lambda line: None)
    runner.leader_feed = runner.follower_feed = silent
    asyncio.run(runner.run(minutes=0, measure_only=True))
    runner.log.close()
    start = json.loads((tmp_path / "run.jsonl").read_text().splitlines()[0])
    assert start["event"] == "start"
    assert (start["config"]["tick"], start["demo_tick"]) == (0.0001, "0.001")


def test_a_reprice_is_one_amend_and_a_reprice_to_the_resting_price_sends_nothing(tmp_path):
    """Nine processes on one key hit the demo post limit (2026-09-13). Book 1.000/1.004,
    leader 1.0130: bid posted at 1.003. The book moves to 1.004/1.006: 1.004 > 1.003 cancels
    it, and 1.005 is alone and (1.013 - 1.005) / 1.005 = 79.6 bps past - posted in the same
    decision. On demo that is one amend of o1 to 1.005, not a cancel and a post. At the ttl
    (posted t=100, 10 s) the paper side cancels and reposts 1.005: the demo order already
    rests there, so nothing is sent. The paper log keeps all five intents."""
    import asyncio
    broker = _Broker()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    runner.quoter.on_book(0, 1.000, 1.004)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 1.0130))
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_book(100, 1.004, 1.006))
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_clock(100 + 10_001))
        await runner.drain_mirror()
    asyncio.run(go())
    log.close()
    assert [(c[0], c[1].get("price"), c[1].get("order_id")) for c in broker.calls] == [
        ("limit", "1.003", None), ("amend", "1.005", "o1")]
    rows = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows if r["event"] == "intent"] == ["post", "cancel", "post", "cancel", "post"]
    assert [r["kind"] for r in rows if r["event"] == "demo_skip"] == ["amend"]
    report = summarise(tmp_path / "run.jsonl")
    assert (report.demo_amends, report.amend_skips) == (1, 1)


def test_a_refused_amend_falls_back_to_cancel_and_post(tmp_path):
    """If the venue refuses the amend (the order just filled or was cancelled on arrival),
    the resting order must still go and the new price must still be quoted."""
    import asyncio

    class _Refusing(_Broker):
        def amend(self, **kw):
            self.calls.append(("amend", kw))
            return {"code": "1", "msg": "refused (test)", "data": {"code": "1", "msg": "refused (test)"}}

    broker = _Refusing()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    runner.quoter.on_book(0, 1.000, 1.004)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 1.0130))
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_book(100, 1.004, 1.006))
        await runner.drain_mirror()
    asyncio.run(go())
    log.close()
    assert [(c[0], c[1].get("price")) for c in broker.calls] == [
        ("limit", "1.003"), ("amend", "1.005"), ("cancel", None), ("limit", "1.005")]
    assert broker.calls[2][1]["order_id"] == "o1"


def test_the_runner_wakes_itself_to_post_a_move_that_outlasted_flicker_ms(tmp_path):
    """The runner's clock ticks every 250 ms; a 5 ms confirmation must not wait for it or
    for the next message. One leader tick past the edge, then silence: the post must still
    be in the log a few ms later."""
    import asyncio
    from trading.strategies.lead_quote.execute import now_ms
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK, flicker_ms=5), log=log, on_log=lambda s: None)
    runner.quoting = True

    async def go():
        runner.quoter.on_book(now_ms(), 1.000, 1.003)
        await runner.handle(runner.quoter.on_leader(now_ms(), 1.0030))
        assert runner.quoter.posts == 0
        await asyncio.sleep(0.05)
    asyncio.run(go())
    log.close()
    rows = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert [(r["kind"], r["price"]) for r in rows if r["event"] == "intent"] == [("post", pytest.approx(1.002))]


def test_a_rejected_ack_keeps_the_envelope_message(tmp_path):
    """The Tokyo run's summary printed a rejection with an empty message because the
    text sat on the response envelope, not the data row."""
    import asyncio

    class _Rejecting(_Broker):
        def cancel(self, **kw):
            self.calls.append(("cancel", kw))
            return {"code": "152404", "msg": "Order has been filled", "data": []}

    broker = _Rejecting()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK, order_ttl_ms=1000), log=log, broker=broker)
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 1.0030))
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_leader(1400, 1.0015))
        await runner.drain_mirror()
    asyncio.run(go())
    log.close()
    report = summarise(tmp_path / "run.jsonl")
    assert report.problems == ["1 demo orders rejected, first: cancel code 152404 Order has been filled"]


def test_a_cancel_refused_because_the_venue_already_cancelled_the_post_only_is_not_a_rejection(tmp_path):
    """Both eight-hour Tokyo runs reported 15 rejected cancels: each was a post_only
    the demo book had cancelled on arrival because it would have crossed. Counted
    as such, not as a problem; a cancel refused for any other order still is."""
    path = tmp_path / "run.jsonl"
    rows = [
        {"t": 0, "event": "start", "inst_id": "SUI-USDT"},
        {"t": 100, "event": "demo_ack", "kind": "post", "cid": "a", "ok": True, "rtt_ms": 30.0},
        {"t": 130, "event": "demo_order", "cid": "a", "kind": "post", "state": "canceled", "filled_size": "0"},
        {"t": 900, "event": "demo_ack", "kind": "cancel", "cid": "a", "ok": False, "code": "102068",
         "msg": "Cancel failed as the order has been filled, triggered, canceled or does not exist.", "rtt_ms": 25.0},
        {"t": 2000, "event": "demo_ack", "kind": "post", "cid": "b", "ok": True, "rtt_ms": 30.0},
        {"t": 2900, "event": "demo_ack", "kind": "cancel", "cid": "b", "ok": True, "rtt_ms": 25.0},
        {"t": 2950, "event": "demo_order", "cid": "b", "kind": "post", "state": "canceled", "filled_size": "0"},
        {"t": 4000, "event": "demo_ack", "kind": "post", "cid": "c", "ok": True, "rtt_ms": 30.0},
        {"t": 4900, "event": "demo_ack", "kind": "cancel", "cid": "c", "ok": False, "code": "102068",
         "msg": "Cancel failed", "rtt_ms": 25.0},
        {"t": 3_600_000, "event": "stop"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report = summarise(path)
    assert report.demo_crossed == 1
    assert report.problems == ["1 demo orders rejected, first: cancel code 102068 Cancel failed"]
    assert any("1 post_only cancelled on arrival" in line for line in report.lines())


def test_the_mirror_is_queued_so_a_feed_never_waits_on_the_venue(tmp_path):
    """handle() must return without touching the broker; the worker sends in order."""
    import asyncio
    broker = _Broker()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 1.0030))          # post
        await runner.handle(runner.quoter.on_leader(1400, 1.0015))       # cancel
        assert broker.calls == []                                        # nothing sent yet
        worker = asyncio.create_task(runner.mirror_worker())
        async def sent():
            while len(broker.calls) < 2:          # two thread hops: wall time, not yields
                await asyncio.sleep(0.01)
        await asyncio.wait_for(sent(), 5)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    asyncio.run(go())
    log.close()
    assert [c[0] for c in broker.calls] == ["limit", "cancel"]


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


def test_a_probe_cancels_by_the_order_id_in_the_rest_ack(tmp_path):
    """The probe once read the order id only from the WS order stream, so when the REST
    ack came back first no cancel was sent and the order stayed resting. With no stream
    at all, two cycles must send limit, cancel(o1), limit, cancel(o3)."""
    import asyncio
    broker = _Broker()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    runner._bid = 1.000
    asyncio.run(runner.probe(2, pause_s=0))
    log.close()
    assert [c[0] for c in broker.calls] == ["limit", "cancel", "limit", "cancel"]
    assert broker.calls[1][1]["order_id"] == "o1" and broker.calls[3][1]["order_id"] == "o3"
    assert broker.calls[0][1]["price"] == "0.950"


def test_an_async_broker_is_awaited_on_the_loop_not_sent_to_a_thread(tmp_path):
    """The aiohttp broker's calls return coroutines. Handed to asyncio.to_thread they
    would come back un-awaited: no order sent and a dict lookup on a coroutine."""
    import asyncio

    class _Async(_Broker):
        is_async = True

        async def place_limit(self, **kw):
            return _Broker.place_limit(self, **kw)

        async def cancel(self, **kw):
            return _Broker.cancel(self, **kw)

    broker = _Async()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    runner._bid = 1.000
    asyncio.run(runner.probe(1, pause_s=0))
    log.close()
    assert [c[0] for c in broker.calls] == ["limit", "cancel"]
    acks = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert [(a["kind"], a["ok"]) for a in acks] == [("probe_post", True), ("probe_cancel", True)]


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


def _demo_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def test_a_live_row_after_canceled_does_not_revive_the_order_and_shutdown_cancels_nothing(tmp_path):
    """FIL-USDT, Tokyo, 2026-09-13, cid lqf954433497e4: post acked, `live`, `live`, cancel
    acked ok, `canceled` - then `live` again 39 ms and 390 ms later. The runner took the last
    row at its word, so at shutdown it cancelled the order a second time and the venue refused
    102068. Replayed in that order: the order stays canceled, the two late rows are logged as
    received and marked stale, shutdown sends nothing, and the run reports no problem."""
    import asyncio
    broker = _Broker()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("FIL-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 1.0030))        # post at 1.002
        await runner.drain_mirror()
        row = {"clientOrderId": broker.calls[0][1]["client_order_id"], "orderId": "o1", "filledSize": "0"}
        await runner.on_demo_row(dict(row, state="live", averagePrice="0.000000000000000000"))
        await runner.on_demo_row(dict(row, state="live", averagePrice="0"))
        await runner.handle(runner.quoter.on_leader(1400, 1.0015))     # leader back -> cancel, acked ok
        await runner.drain_mirror()
        await runner.on_demo_row(dict(row, state="canceled", averagePrice="0.000000000000000000"))
        await runner.on_demo_row(dict(row, state="live", averagePrice="0"))
        await runner.on_demo_row(dict(row, state="live", averagePrice="0"))
        await runner.shutdown_demo()
        return row["clientOrderId"]
    cid = asyncio.run(go())
    log.close()
    assert runner.demo_orders[cid].state == "canceled"
    assert [c[0] for c in broker.calls] == ["limit", "cancel"]
    assert [(r["state"], r.get("stale", False)) for r in _demo_rows(tmp_path / "run.jsonl")
            if r["event"] == "demo_order"] == [
        ("live", False), ("live", False), ("canceled", False), ("live", True), ("live", True)]
    assert summarise(tmp_path / "run.jsonl").problems == []


def test_a_late_live_row_sends_no_amend_and_no_cancel_to_an_order_the_venue_already_cancelled(tmp_path):
    """The amend path and the skip-cancel rule both read `order.state`. Book 1.000/1.004,
    leader 1.0130: a bid is posted at 1.003 and the demo book cancels it on arrival, then a
    late `live` row follows. Book 1.004/1.006: the paper side reprices to 1.005, folded into
    an amend. With the order revived, that amend went to an order that no longer exists and,
    refused, fell back to a cancel of it and a post: four requests against the post limit
    instead of two. Kept canceled: no amend, no cancel, one post at 1.005."""
    import asyncio

    class _Refusing(_Broker):
        def amend(self, **kw):
            self.calls.append(("amend", kw))
            return {"code": "1", "msg": "refused (test)", "data": {"code": "1", "msg": "refused (test)"}}

    broker = _Refusing()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker)
    runner.quoter.on_book(0, 1.000, 1.004)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 1.0130))
        await runner.drain_mirror()
        row = {"clientOrderId": broker.calls[0][1]["client_order_id"], "orderId": "o1", "filledSize": "0"}
        await runner.on_demo_row(dict(row, state="canceled"))
        await runner.on_demo_row(dict(row, state="live"))
        await runner.handle(runner.quoter.on_book(100, 1.004, 1.006))
        await runner.drain_mirror()
    asyncio.run(go())
    log.close()
    assert [(c[0], c[1].get("price")) for c in broker.calls] == [("limit", "1.003"), ("limit", "1.005")]


def test_an_accepted_cancel_is_terminal_before_the_stream_says_so_but_a_later_fill_still_lands(tmp_path):
    """Two orderings. (1) The cancel is acknowledged and a late `live` row arrives before any
    `canceled` row: the order is gone, so shutdown must not cancel it again. (2) UNI-USDT,
    2026-09-13 22:08 UTC, cid lqb51783dfb7ca: the cancel was acknowledged ok and 27 ms later
    the stream reported the ask filled, 0.1 contracts, and the short was real. A terminal rule
    that swallowed that fill would lose it from demo_net and never close it. Ask side, book
    1.000/1.003: leader 0.9990 posts 1.001, leader 1.0015 cancels it, twice. The second
    entry's fill (-0.1) is closed by a reduce-only buy of 0.1; that close's own fill brings
    demo_net back to 0, and shutdown then sends nothing."""
    import asyncio
    from decimal import Decimal
    broker = _Broker()
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("UNI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker, on_log=lambda s: None)
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 0.9990))
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_leader(1400, 1.0015))
        await runner.drain_mirror()
        first = broker.calls[0][1]["client_order_id"]
        await runner.on_demo_row({"clientOrderId": first, "orderId": "o1", "state": "live", "filledSize": "0"})
        await runner.handle(runner.quoter.on_leader(1500, 0.9990))
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_leader(1600, 1.0015))
        await runner.drain_mirror()
        second = broker.calls[2][1]["client_order_id"]
        await runner.on_demo_row({"clientOrderId": second, "orderId": "o3", "state": "filled", "filledSize": "0.1"})
        assert runner.demo_net == Decimal("-0.1")
        close = broker.calls[4][1]
        await runner.on_demo_row({"clientOrderId": close["client_order_id"], "orderId": "o5",
                                  "state": "filled", "filledSize": "0.1"})
        await runner.shutdown_demo()
        return first, second
    first, second = asyncio.run(go())
    log.close()
    assert (runner.demo_orders[first].state, runner.demo_orders[second].state) == ("canceled", "filled")
    assert [c[0] for c in broker.calls] == ["limit", "cancel", "limit", "cancel", "market"]
    assert (broker.calls[4][1]["side"], broker.calls[4][1]["size"], broker.calls[4][1]["reduce_only"]) == (
        "buy", Decimal("0.1"), True)
    assert runner.demo_net == Decimal("0")


class _ApiError(Exception):
    """The SDK's `BlofinAPIException` as the requests transport raises a refused call."""

    def __init__(self, message, code=None, status_code=None):
        super().__init__(message)
        self.code, self.status_code = code, status_code


class _RateLimited(_Broker):
    """Refuses the first `refusals[kind]` calls of each kind with 429, as the demo host did
    at ~4 posts a second across nine processes on one key. `raise_cancel` refuses cancels the
    way the requests transport does: an exception carrying the code."""

    def __init__(self, raise_cancel=False, **refusals):
        super().__init__()
        self.refusals, self.raise_cancel = dict(refusals), raise_cancel

    def _refused(self, kind, kw):
        if self.refusals.get(kind, 0) <= 0:
            return False
        self.refusals[kind] -= 1
        self.calls.append((kind, kw))
        return True

    def place_limit(self, **kw):
        if self._refused("limit", kw):
            return {"code": "429", "msg": "rate limit exceeded"}
        return _Broker.place_limit(self, **kw)

    def place_market(self, **kw):
        if self._refused("market", kw):
            return {"code": "429", "msg": "rate limit exceeded"}
        return _Broker.place_market(self, **kw)

    def cancel(self, **kw):
        if self._refused("cancel", kw):
            if self.raise_cancel:
                raise _ApiError("API request failed: rate limit exceeded", code="429", status_code=200)
            return {"code": "429", "msg": "rate limit exceeded"}
        return _Broker.cancel(self, **kw)


def _acks(path, kind):
    return [(r["ok"], r.get("code"), r.get("retry_in_ms"), r.get("attempt"))
            for r in _demo_rows(path) if r["event"] == "demo_ack" and r["kind"] == kind]


def test_a_flatten_refused_429_is_sent_again_and_closes_the_orphan(tmp_path):
    """UNI-USDT, Tokyo, 2026-09-13 22:08 UTC, cid lq4874c03e39b6: an ask filled after the
    paper cancel, the reduce-only market buy that should have closed it came back 429 "rate
    limit exceeded", nothing sent it again, and a 0.1 contract short stayed open. Replayed
    with the first flatten refused: the same order goes again, same clientOrderId (the
    refused one does not exist), one demo_ack row per attempt, and its fill takes demo_net
    from -0.1 back to 0. The report counts one retry and no rejection."""
    import asyncio
    from decimal import Decimal
    broker = _RateLimited(market=1)
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("UNI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker,
                             on_log=lambda s: None, retry_429_s=(0, 0, 0))
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 0.9990))
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_leader(1400, 1.0015))
        await runner.drain_mirror()
        entry = broker.calls[0][1]["client_order_id"]
        await runner.on_demo_row({"clientOrderId": entry, "orderId": "o1", "state": "filled", "filledSize": "0.1"})
        await runner.on_demo_row({"clientOrderId": broker.calls[3][1]["client_order_id"], "orderId": "o4",
                                  "state": "filled", "filledSize": "0.1"})
    asyncio.run(go())
    log.close()
    assert [c[0] for c in broker.calls] == ["limit", "cancel", "market", "market"]
    assert broker.calls[2][1] == broker.calls[3][1]
    assert (broker.calls[3][1]["side"], broker.calls[3][1]["size"], broker.calls[3][1]["reduce_only"]) == (
        "buy", Decimal("0.1"), True)
    assert _acks(tmp_path / "run.jsonl", "flatten") == [(False, "429", 0, None), (True, "0", None, 2)]
    assert runner.demo_net == Decimal("0")
    report = summarise(tmp_path / "run.jsonl")
    assert report.demo_retried == 1
    assert report.problems == ["1 demo entries filled after the paper side cancelled (closed reduce-only)"]
    assert "  demo rate limit: 1 closes/cancels refused 429 and sent again" in report.lines()


def test_cancel_and_exit_cross_are_retried_on_429_and_an_entry_post_is_not(tmp_path):
    """A late entry is stale: the gap it was priced on is gone by the time a slot frees up,
    so a refused post is logged and dropped. A cancel and a crossing exit take risk off and go
    again. hold_ms 0 makes the paper fill cross out in the same decision (hold limit), so the
    exit is an exit_cross with no exit_post before it. Book 1.000/1.003, leader 1.0030 posts
    1.002 and 1.0015 cancels it: post refused (nothing to cancel after); post accepted, its
    cancel refused - raised, the requests transport's shape - and accepted; post accepted,
    the demo book fills it, the tape sells at 1.002, exit_cross refused and accepted."""
    import asyncio
    from decimal import Decimal
    broker = _RateLimited(raise_cancel=True, limit=1, cancel=1, market=1)
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("SUI-USDT", QuoteConfig(tick=TICK, hold_ms=0), log=log, broker=broker,
                             on_log=lambda s: None, retry_429_s=(0, 0, 0))
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        for t, leader in ((1, 1.0030), (1400, 1.0015), (1500, 1.0030), (1600, 1.0015), (1700, 1.0030)):
            await runner.handle(runner.quoter.on_leader(t, leader))
            await runner.drain_mirror()
        assert [c[0] for c in broker.calls] == ["limit", "limit", "cancel", "cancel", "limit"]
        await runner.on_demo_row({"clientOrderId": broker.calls[4][1]["client_order_id"], "orderId": "o5",
                                  "state": "filled", "filledSize": "1"})
        intents = runner.quoter.on_trade(2000, 1.002, "sell")
        assert kinds(intents) == [("exit_cross", 1)]
        await runner.handle(intents)
        await runner.drain_mirror()
    asyncio.run(go())
    log.close()
    assert [c[0] for c in broker.calls] == ["limit", "limit", "cancel", "cancel", "limit", "market", "market"]
    assert broker.calls[2][1] == broker.calls[3][1] and broker.calls[3][1]["order_id"] == "o2"
    assert broker.calls[5][1] == broker.calls[6][1] and broker.calls[6][1]["reduce_only"] is True
    path = tmp_path / "run.jsonl"
    assert _acks(path, "post") == [(False, "429", None, None), (True, "0", None, None), (True, "0", None, None)]
    assert _acks(path, "cancel") == [(False, None, 0, None), (True, "0", None, 2)]
    assert _acks(path, "exit_cross") == [(False, "429", 0, None), (True, "0", None, 2)]
    assert runner.demo_net == Decimal("1")          # the exit's fill notice has not arrived
    report = summarise(path)
    assert report.demo_retried == 2
    assert report.problems == ["1 demo orders rejected, first: post code 429 rate limit exceeded"]


def test_a_close_refused_on_every_attempt_waits_250_500_1000_ms_and_is_left_to_shutdown(tmp_path, monkeypatch):
    """Bounded: at the defaults a close is tried four times with 0.25, 0.5 and 1.0 s between,
    1.75 s in all. If all four come back 429 the orphan stays open, but demo_net - which only
    fill rows move - still holds it, and the entry's filled_size was zeroed before the first
    attempt so a duplicate fill row cannot start a second close. Shutdown then closes it
    (its 5th market call is accepted), and the last refusal is reported as a rejection."""
    import asyncio
    from decimal import Decimal
    from trading.strategies.lead_quote import execute

    waits, duplicate = [], {}

    async def no_wait(seconds):
        waits.append(seconds)
        if len(waits) == 1:                             # the stream repeats the fill during the backoff
            await runner.on_demo_row(duplicate)

    monkeypatch.setattr(execute.asyncio, "sleep", no_wait)
    broker = _RateLimited(market=4)
    broker.positions = lambda: [{"instId": "UNI-USDT", "positions": "-0.1"}]
    log = RunLog(tmp_path / "run.jsonl")
    runner = LeadQuoteRunner("UNI-USDT", QuoteConfig(tick=TICK), log=log, broker=broker, on_log=lambda s: None)
    runner.quoting = True
    runner.quoter.on_book(0, 1.000, 1.003)

    async def go():
        await runner.handle(runner.quoter.on_leader(1, 0.9990))
        await runner.drain_mirror()
        await runner.handle(runner.quoter.on_leader(1400, 1.0015))
        await runner.drain_mirror()
        duplicate.update(clientOrderId=broker.calls[0][1]["client_order_id"], orderId="o1",
                         state="filled", filledSize="0.1")
        await runner.on_demo_row(dict(duplicate))
        assert [c[0] for c in broker.calls] == ["limit", "cancel", "market", "market", "market", "market"]
        assert runner.demo_net == Decimal("-0.1") and waits == [0.25, 0.5, 1.0]
        await runner.shutdown_demo()
    asyncio.run(go())
    log.close()
    assert [c[0] for c in broker.calls] == ["limit", "cancel"] + ["market"] * 5
    assert (broker.calls[6][1]["side"], broker.calls[6][1]["size"]) == ("buy", Decimal("0.1"))
    assert _acks(tmp_path / "run.jsonl", "flatten") == [
        (False, "429", 250, None), (False, "429", 500, 2), (False, "429", 1000, 3), (False, "429", None, 4),
        (True, "0", None, None)]
    report = summarise(tmp_path / "run.jsonl")
    assert report.demo_retried == 3
    assert report.problems[0] == "1 demo orders rejected, first: flatten code 429 rate limit exceeded"
