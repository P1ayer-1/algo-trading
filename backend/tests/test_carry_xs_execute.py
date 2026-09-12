"""The executor: the guarantees that make order-sending code safe to have.

Every test here names a way a twelve-leg rebalance can leave the account
holding something nobody chose. The expensive ones are silent: a flip sent as
one order that half-fills and leaves the old side on, a repair that opens
instead of reducing, an interleaving that spends the middle of the operation
long half the book, and a dry run that is not dry.
"""

from decimal import Decimal

import pytest

from trading.strategies import Outcome
from trading.strategies.carry_xs import BookConfig, Candidate, plan_book
from trading.strategies.carry_xs.broker import Rules
from trading.strategies.carry_xs.execute import (
    BookExecutor,
    ExecutionResult,
    Order,
    current_book,
    plan_orders,
    target_book,
)


class FakeBroker:
    """Records every call. Rejects whatever `reject` names."""

    def __init__(self, positions=None, quotes=None, reject=(), raises=(),
                 contract_values=None, marks=None, lot="0.01",
                 min_size="0.01"):
        self._positions = dict(positions or {})
        self._quotes = dict(quotes or {})
        self._marks = dict(marks or {})
        self._lot = Decimal(lot)
        self._min_size = Decimal(min_size)
        # Default 1 base unit per contract, so a test that does not care about
        # the multiplier reads in plain dollars.
        self._contract_values = dict(contract_values
                                     or {"A": Decimal("1"), "B": Decimal("1"),
                                         "C": Decimal("1"),
                                         "A-USDT": Decimal("1")})
        for name in list(self._positions):
            self._contract_values.setdefault(name, Decimal("1"))
        for index in range(30):
            self._contract_values.setdefault(
                "C{:02d}-USDT".format(index), Decimal("1"))
        self.reject = set(reject)
        self.raises = set(raises)
        self.sent = []
        self.leverage_set = []

    def margin_mode(self):
        return "cross"

    def set_leverage(self, inst_id, leverage):
        self.leverage_set.append((inst_id, leverage))
        return {"code": "0"}

    def place_perp(self, *, inst_id, side, size, client_order_id,
                   reduce_only=False):
        self.sent.append({"instId": inst_id, "side": side, "size": size,
                          "reduceOnly": reduce_only,
                          "clientOrderId": client_order_id})
        if inst_id in self.raises:
            raise RuntimeError("network went away")
        if inst_id in self.reject:
            return {"code": "0", "data": [{"code": "103", "msg": "rejected"}]}
        delta = size if side == "buy" else -size
        self._positions[inst_id] = self._positions.get(inst_id, Decimal("0")) + delta
        return {"code": "0", "data": [{"code": "0", "orderId": "1"}]}

    def positions(self):
        return [{"instId": inst_id, "positions": str(size),
                 "markPrice": str(self._marks.get(inst_id, Decimal("100")))}
                for inst_id, size in self._positions.items() if size != 0]

    def instrument_rules(self):
        return {name: Rules(contract_value=value,
                            lot_size=self._lot, min_size=self._min_size)
                for name, value in self._contract_values.items()}

    def quote(self, inst_id):
        return self._quotes.get(inst_id, {"bid": Decimal("99.99"),
                                          "ask": Decimal("100.01"),
                                          "last": Decimal("100")})


def candidate(name, carry, *, price="100"):
    bid = Decimal(price)
    return Candidate(inst_id=name, carry_bps_per_day=carry, bid=bid,
                     ask=bid + Decimal("0.02"), volume_usd_24h=2e7,
                     vol_30d_bps=300.0, contract_value=Decimal("1"),
                     lot_size=Decimal("0.01"), min_size=Decimal("0.01"),
                     max_leverage=Decimal("20"), funding_days=200)


def a_plan(n=10, notional="6000"):
    from trading.risk import RiskLimits

    names = [candidate("C{:02d}-USDT".format(i), (i - (n - 1) / 2) * 0.6)
             for i in range(n)]
    return plan_book(names, BookConfig(gross_notional_usd=Decimal(notional)),
                     RiskLimits(max_notional=Decimal("100000")))


# ---------------------------------------------------------------------------
# The deltas
# ---------------------------------------------------------------------------


UNIT_USD = {"A": Decimal("100"), "B": Decimal("100"), "C": Decimal("100")}


def test_opening_from_flat_is_not_reduce_only():
    orders = plan_orders({}, {"A": Decimal("5")}, UNIT_USD)
    assert len(orders) == 1
    assert orders[0].side == "buy" and not orders[0].reduce_only


def test_closing_to_flat_is_reduce_only():
    """`reduce_only` on every close is what stops an exit becoming an entry.

    Without it, a stale size read or a position that moved between the read and
    the send turns a close into a position on the other side.
    """
    orders = plan_orders({"A": Decimal("5")}, {}, UNIT_USD)
    assert len(orders) == 1
    assert orders[0].side == "sell" and orders[0].reduce_only
    assert orders[0].contracts == Decimal("5")


def test_trimming_a_position_is_reduce_only_and_growing_one_is_not():
    trim = plan_orders({"A": Decimal("5")}, {"A": Decimal("3")}, UNIT_USD)
    assert trim[0].side == "sell" and trim[0].reduce_only
    assert trim[0].contracts == Decimal("2")

    grow = plan_orders({"A": Decimal("3")}, {"A": Decimal("5")}, UNIT_USD)
    assert grow[0].side == "buy" and not grow[0].reduce_only
    assert grow[0].contracts == Decimal("2")


def test_a_flip_is_split_into_a_close_and_an_open():
    """One order cannot flip a position safely.

    `reduce_only` is rejected past the crossing point, and a plain order that
    is rejected half way leaves the position still on the OLD side - the book
    then holds the opposite of what the plan wanted and nothing says so. Split
    in two, a failed second half leaves the name flat, which is recoverable.
    """
    orders = plan_orders({"A": Decimal("5")}, {"A": Decimal("-3")}, UNIT_USD)
    assert len(orders) == 2
    close = [o for o in orders if o.reduce_only][0]
    open_ = [o for o in orders if not o.reduce_only][0]
    assert close.side == "sell" and close.contracts == Decimal("5")
    assert open_.side == "sell" and open_.contracts == Decimal("3")


def test_a_position_already_correct_produces_no_order():
    assert plan_orders({"A": Decimal("5")}, {"A": Decimal("5")}, UNIT_USD) == []


def test_shorts_are_read_from_the_sign_not_the_magnitude():
    """BloFin reports direction in the SIGN of `positions`.

    Reading the magnitude and assuming a side is how a short gets doubled
    instead of closed, so this pins that a negative position closes with a BUY.
    """
    broker = FakeBroker(positions={"A": Decimal("-4")})
    assert current_book(broker) == {"A": Decimal("-4")}
    orders = plan_orders(current_book(broker), {}, UNIT_USD)
    assert orders[0].side == "buy" and orders[0].reduce_only


# ---------------------------------------------------------------------------
# The directional window
# ---------------------------------------------------------------------------


def test_orders_are_interleaved_so_the_running_net_stays_small():
    """Six buys then six sells would spend the middle of the operation holding
    exactly the exposure the strategy exists to avoid.

    Interleaving bounds the worst intermediate net by ONE leg. Here six legs of
    $100 a side: all-buys-first would peak at $600, interleaved it peaks at
    $100.
    """
    target = {}
    for i in range(6):
        target["L{}".format(i)] = Decimal("1")
        target["S{}".format(i)] = Decimal("-1")
    prices = {name: Decimal("100") for name in target}
    orders = plan_orders({}, target, prices)

    running = Decimal("0")
    peak = Decimal("0")
    for order in orders:
        running += order.signed * Decimal("100")
        peak = max(peak, abs(running))
    assert peak <= Decimal("100")
    assert len(orders) == 12


def test_interleaving_balances_on_notional_not_contract_count():
    """A contract is a different amount of money on every instrument, so
    balancing counts would balance nothing."""
    target = {"BIG": Decimal("1"), "SMALL1": Decimal("-1"), "SMALL2": Decimal("-1")}
    prices = {"BIG": Decimal("1000"), "SMALL1": Decimal("500"),
              "SMALL2": Decimal("500")}
    orders = plan_orders({}, target, prices)
    running = Decimal("0")
    peak = Decimal("0")
    for order in orders:
        running += order.signed * prices[order.inst_id]
        peak = max(peak, abs(running))
    assert peak <= Decimal("1000")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_a_dry_run_sends_absolutely_nothing():
    broker = FakeBroker()
    result = BookExecutor(broker, dry_run=True).reconcile(a_plan())
    assert broker.sent == []
    assert result.dry_run
    assert result.orders and all(not o.sent for o in result.orders)


def test_a_refused_plan_is_never_executed():
    """The plan's gates are the executor's gates. A planner that says no and an
    executor that sends anyway would make every refusal decorative."""
    from trading.risk import RiskLimits

    plan = plan_book([candidate("A-USDT", 1.0)],
                     BookConfig(), RiskLimits())
    assert not plan.ok
    broker = FakeBroker()
    result = BookExecutor(broker, dry_run=False, settle_seconds=0).reconcile(plan)
    assert broker.sent == []
    assert not result.ok
    assert "refused" in result.problems[0]


def test_the_result_is_an_outcome():
    assert isinstance(ExecutionResult(), Outcome)


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------


def test_a_live_run_sends_every_leg_and_lands_neutral():
    broker = FakeBroker()
    plan = a_plan()
    result = BookExecutor(broker, dry_run=False, settle_seconds=0).reconcile(plan)
    assert result.ok, result.problems
    assert result.reconciled
    assert len(broker.sent) == len([leg for leg in plan.legs if leg.ok])
    assert abs(result.net_after_usd) / result.gross_after_usd < Decimal("0.02")


def test_running_it_twice_sends_nothing_the_second_time():
    """Twelve legs is twelve chances to be interrupted, so the answer to "what
    if it dies halfway" has to be "run it again". Every decision comes from
    positions read back from the exchange, so a completed book is a no-op."""
    broker = FakeBroker()
    plan = a_plan()
    executor = BookExecutor(broker, dry_run=False, settle_seconds=0)
    executor.reconcile(plan)
    count = len(broker.sent)
    second = executor.reconcile(plan)
    assert len(broker.sent) == count
    assert second.already_correct


def test_an_interrupted_run_is_finished_by_the_next_one():
    broker = FakeBroker()
    plan = a_plan()
    executor = BookExecutor(broker, dry_run=False, settle_seconds=0)

    # Half the legs never make it.
    legs = [leg.inst_id for leg in plan.legs if leg.ok]
    broker.reject = set(legs[: len(legs) // 2])
    executor.reconcile(plan)
    assert not current_book(broker).keys() >= set(legs)

    broker.reject = set()
    finished = executor.reconcile(plan)
    assert finished.ok, finished.problems
    assert set(current_book(broker)) == set(target_book(plan))


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


def test_a_rejected_leg_is_a_problem_not_a_silent_gap():
    broker = FakeBroker()
    plan = a_plan()
    first = [leg.inst_id for leg in plan.legs if leg.ok][0]
    broker.reject = {first}
    result = BookExecutor(broker, dry_run=False, settle_seconds=0).reconcile(plan)
    assert not result.ok
    assert any(first in problem for problem in result.problems)
    assert result.failed


def test_a_failed_leg_does_not_unwind_the_other_eleven():
    """`strategies/carry` unwinds a half-open carry; this must NOT inherit it.

    Unwinding a twelve-leg rebalance means closing eleven positions the plan
    explicitly wants because the twelfth was rejected - a full round trip to
    undo a book that is 92% correct. The imbalance from one missing leg is a
    fraction of gross; the cure would cost more than the disease.
    """
    broker = FakeBroker()
    plan = a_plan()
    legs = [leg.inst_id for leg in plan.legs if leg.ok]
    broker.reject = {legs[0]}
    result = BookExecutor(broker, dry_run=False, settle_seconds=0).reconcile(plan)
    held = current_book(broker)

    # The invariant is about MONEY, not about the number of rows. When a whole
    # leg is rejected the imbalance IS a whole leg, and the neutral book is
    # then the one with its counterpart gone - so a repair that restores
    # neutrality legitimately closes one position. What it must never do is
    # take off more than the imbalance, which is what unwinding means.
    removed = result.gross_before_usd + sum(
        order.notional_usd for order in result.orders
        if order.reason in ("open", "increase") and order.sent
    ) - result.gross_after_usd
    imbalance = sum(order.notional_usd for order in result.orders
                    if order.reason == "repair")
    assert removed <= imbalance + Decimal("1")
    assert len(held) >= 4, "the book was unwound rather than repaired"
    assert result.gross_after_usd > result.gross_before_usd


def test_an_exception_mid_batch_does_not_abandon_the_rest():
    """A network failure on one leg leaves the book more directional, not less,
    so stopping there is the wrong instinct: the remaining legs are what bring
    it back toward neutral."""
    broker = FakeBroker()
    plan = a_plan()
    legs = [leg.inst_id for leg in plan.legs if leg.ok]
    broker.raises = {legs[0]}
    result = BookExecutor(broker, dry_run=False, settle_seconds=0).reconcile(plan)
    # Every leg was attempted. Not an exact send count: the missing leg leaves
    # the book directional, so the repair legitimately sends more.
    assert set(legs) <= {order["instId"] for order in broker.sent}
    assert any("network went away" in problem for problem in result.problems)


# ---------------------------------------------------------------------------
# The spread gate
# ---------------------------------------------------------------------------


def test_a_leg_whose_spread_blew_out_is_skipped_and_reported():
    """A plan is minutes old and the spread on a thin alt is the largest term
    in its cost. Crossing one that has doubled turns a positive trade negative
    silently."""
    plan = a_plan()
    wide = [leg.inst_id for leg in plan.legs if leg.ok][0]
    broker = FakeBroker(quotes={wide: {"bid": Decimal("99"), "ask": Decimal("101"),
                                       "last": Decimal("100")}})
    result = BookExecutor(broker, dry_run=False, settle_seconds=0,
                          max_spread_bps=30.0).reconcile(plan)
    assert wide not in [order["instId"] for order in broker.sent]
    assert any("spread" in problem for problem in result.problems)


def test_the_spread_gate_never_blocks_getting_out():
    """Refusing to shrink risk because it is expensive is how a book stays
    broken. A reduce is exempt from the gate, always."""
    broker = FakeBroker(
        positions={"A-USDT": Decimal("5")},
        quotes={"A-USDT": {"bid": Decimal("90"), "ask": Decimal("110"),
                           "last": Decimal("100")}})
    orders = plan_orders({"A-USDT": Decimal("5")}, {}, {"A-USDT": Decimal("100")})
    executor = BookExecutor(broker, dry_run=False, settle_seconds=0,
                            max_spread_bps=1.0)
    result = ExecutionResult(dry_run=False)
    assert executor._spread_gate(orders[0], result) is True
    assert result.problems == []


# ---------------------------------------------------------------------------
# The repair
# ---------------------------------------------------------------------------


def test_a_directional_book_is_repaired_by_reducing_never_by_opening():
    """The repair can shrink risk and can never open a position nobody planned.

    That property is what makes it safe to run automatically after a partially
    failed batch - the worst case of a bad repair is a smaller book.
    """
    broker = FakeBroker()
    plan = a_plan()
    # Drop the whole short side, so the filled book is long-heavy.
    shorts = {leg.inst_id for leg in plan.legs if leg.ok and leg.side == "sell"}
    broker.reject = shorts
    result = BookExecutor(broker, dry_run=False, settle_seconds=0).reconcile(plan)

    repairs = [order for order in result.orders if order.reason == "repair"]
    assert repairs, "a long-only book was not repaired"
    assert all(order.reduce_only for order in repairs)
    assert all(order.side == "sell" for order in repairs)


def test_a_book_still_directional_after_repair_says_so_loudly():
    broker = FakeBroker()
    plan = a_plan()
    shorts = {leg.inst_id for leg in plan.legs if leg.ok and leg.side == "sell"}
    longs = {leg.inst_id for leg in plan.legs if leg.ok and leg.side == "buy"}
    broker.reject = shorts
    executor = BookExecutor(broker, dry_run=False, settle_seconds=0)
    # Make the repair fail too, by rejecting everything once the opens are done.
    original = broker.place_perp

    def place(**kwargs):
        if kwargs.get("reduce_only"):
            return {"code": "0", "data": [{"code": "103", "msg": "no"}]}
        return original(**kwargs)

    broker.place_perp = place
    result = executor.reconcile(plan)
    assert not result.ok
    assert any("STILL" in problem for problem in result.problems)


def test_a_neutral_book_is_not_repaired():
    broker = FakeBroker()
    result = BookExecutor(broker, dry_run=False, settle_seconds=0).reconcile(a_plan())
    assert not result.repaired
    assert [order for order in result.orders if order.reason == "repair"] == []


# ---------------------------------------------------------------------------
# Contract value, which a live dry run caught being ignored
# ---------------------------------------------------------------------------


def test_notional_uses_contract_value_not_contract_count():
    """A contract is a different amount of money on every instrument.

    On BloFin one BTC contract is 0.001 BTC and one DOGE contract is 1000 DOGE
    - a factor of a million between two legs of the same book. A first version
    of this executor priced orders as `contracts x price`, and a dry run
    against the real account reported a BTC position as $68.9 MILLION and a TRX
    leg as $0.49. Both were wrong by exactly the multiplier.
    """
    from trading.strategies.carry_xs.execute import unit_values

    plan = a_plan(10)
    leg = [item for item in plan.legs if item.ok][0]
    leg.contract_value = Decimal("0.001")
    leg.price = Decimal("80000")
    units = unit_values(plan, {}, None)
    assert units[leg.inst_id] == Decimal("80")      # 0.001 BTC at $80,000

    orders = plan_orders({}, {leg.inst_id: Decimal("10")}, units)
    assert orders[0].notional_usd == Decimal("800")


def test_a_leftover_position_the_plan_never_mentions_is_still_priced():
    """A position from some earlier run has to be closable.

    It is not in the plan, so its price comes from the exchange's mark and its
    contract value from the instrument table - and without BOTH it would be
    priced on contract count and trimmed by the wrong amount.
    """
    from trading.strategies.carry_xs.execute import unit_values

    units = unit_values(a_plan(10), {"OLD-USDT": Decimal("1000")},
                        {"OLD-USDT": Decimal("0.1")})
    assert units["OLD-USDT"] == Decimal("100")


def test_interleaving_is_correct_when_contract_values_differ():
    """The multiplier must not leak into the balancing.

    One BTC-style contract worth $80 against DOGE-style contracts worth $0.08:
    balanced on notional the peak is one leg, balanced on count it would be
    absurd.
    """
    target = {"BTCISH": Decimal("10"), "DOGEISH": Decimal("10000")}
    units = {"BTCISH": Decimal("80"), "DOGEISH": Decimal("-0")}
    units = {"BTCISH": Decimal("80"), "DOGEISH": Decimal("0.08")}
    target["DOGEISH"] = Decimal("-10000")
    orders = plan_orders({}, target, units)
    running = Decimal("0")
    peak = Decimal("0")
    for order in orders:
        running += order.signed * units[order.inst_id]
        peak = max(peak, abs(running))
    assert peak == Decimal("800")     # one leg, not the sum of both


def test_the_interleave_starts_from_the_exposure_the_account_already_has():
    """Balancing the ORDER FLOW is not balancing the BOOK.

    They are the same thing only when the account begins flat, and on a weekly
    rebalance it never does. A book already long $1,000 needs sells first; a
    counter that starts at zero sends a buy and makes the exposure worse before
    it makes it better. Found by running the real dry run against an account
    holding a leftover position.
    """
    units = {"OLD": Decimal("1000"), "NEW": Decimal("1000")}
    orders = plan_orders({"OLD": Decimal("1")}, {"NEW": Decimal("1")}, units)
    assert orders[0].inst_id == "OLD" and orders[0].side == "sell"
    assert orders[0].net_after == Decimal("0")
    assert orders[1].net_after == Decimal("1000")


def test_net_after_tracks_the_book_not_the_orders():
    units = {"A": Decimal("100")}
    orders = plan_orders({"A": Decimal("5")}, {}, units)
    # Closing a $500 long ends at zero exposure, not at -$500.
    assert orders[0].net_after == Decimal("0")


# ---------------------------------------------------------------------------
# Size rules, both of which a live run rejected
# ---------------------------------------------------------------------------


def test_every_order_is_rounded_to_the_lot_it_will_be_validated_against():
    """`152002 Parameter size error`, four times, on the first live run.

    The repair sized itself as `excess / unit` and sent the result raw -
    `0.2276622802836378615352929061` contracts. Rounding where a size is
    COMPUTED is not enough, because the next code path that computes one has
    to remember; it happens on the way out instead.
    """
    from trading.strategies.carry_xs.execute import round_to_lot

    assert round_to_lot(Decimal("0.2276622802836378615"), Decimal("0.01")) \
        == Decimal("0.22")
    assert round_to_lot(Decimal("1.99"), Decimal("0.1")) == Decimal("1.9")
    assert round_to_lot(Decimal("4.31"), Decimal("1")) == Decimal("4")
    # No lot rule known: pass it through rather than inventing one.
    assert round_to_lot(Decimal("4.31"), Decimal("0")) == Decimal("4.31")


def test_a_repair_order_is_rounded_and_dropped_when_it_rounds_below_minimum():
    broker = FakeBroker(lot="0.1", min_size="0.1")
    plan = a_plan()
    shorts = {leg.inst_id for leg in plan.legs if leg.ok and leg.side == "sell"}
    broker.reject = shorts
    result = BookExecutor(broker, dry_run=False, settle_seconds=0).reconcile(plan)
    for order in result.orders:
        if order.reason != "repair":
            continue
        assert order.contracts % Decimal("0.1") == 0, order.contracts
        assert order.contracts >= Decimal("0.1")


def test_an_order_below_the_minimum_size_is_never_sent():
    """A leg too small to be legal is not a leg, and sending it just collects a
    rejection while leaving the book short by that amount anyway."""
    rules = {"A": Rules(contract_value=Decimal("1"), lot_size=Decimal("1"),
                        min_size=Decimal("5"))}
    orders = plan_orders({}, {"A": Decimal("3")}, {"A": Decimal("100")}, rules)
    assert orders == []


def test_a_close_is_exempt_from_the_minimum_size():
    """The exchange lets a position be closed whatever its size.

    Applying the opening minimum to a close would strand a small position
    forever - and small positions are exactly what a partially failed rebalance
    leaves behind.
    """
    rules = {"A": Rules(contract_value=Decimal("1"), lot_size=Decimal("0.01"),
                        min_size=Decimal("5"))}
    orders = plan_orders({"A": Decimal("3")}, {}, {"A": Decimal("100")}, rules)
    assert len(orders) == 1
    assert orders[0].reduce_only and orders[0].contracts == Decimal("3")


def test_sizes_are_rounded_down_so_a_leg_never_exceeds_its_target():
    rules = {"A": Rules(contract_value=Decimal("1"), lot_size=Decimal("0.1"),
                        min_size=Decimal("0.1"))}
    orders = plan_orders({}, {"A": Decimal("4.39")}, {"A": Decimal("100")}, rules)
    assert orders[0].contracts == Decimal("4.3")


def test_an_instrument_the_account_cannot_trade_is_dropped_and_reported():
    """Demo lists 87 instruments against production's 488.

    A plan built on production prices names instruments this account has never
    heard of. Sending them one at a time collects one rejection each and leaves
    the book quietly short; dropping them up front and saying so does not.
    """
    broker = FakeBroker()
    plan = a_plan()
    unlisted = [leg.inst_id for leg in plan.legs if leg.ok][0]
    broker._contract_values.pop(unlisted, None)
    result = BookExecutor(broker, dry_run=False, settle_seconds=0).reconcile(plan)
    assert unlisted not in [order["instId"] for order in broker.sent]
    assert any("not listed on this account's host" in problem
               for problem in result.problems)


# ---------------------------------------------------------------------------
# The repair aims at neutral, which the second live run proved it had not been
# ---------------------------------------------------------------------------


def test_the_repair_targets_zero_net_and_not_the_edge_of_the_tolerance():
    """The exact book the second live run failed on, to the cent.

    Measured 2026-09-12 on demo: $139.91 net long on $3,377.27 gross, 4.14%
    against a 2% tolerance. Aiming at the tolerance EDGE sized one $71.28 trim,
    which landed at $68.62 net on $3,305.98 gross - 2.08% - and reported "STILL
    directional". It could not have succeeded: trimming shrinks gross, so the
    boundary being aimed at moves toward the book by `tolerance x trim`, while
    rounding the trim down stops it reaching even that.
    """
    from trading.strategies.carry_xs.execute import repair_trims

    # TRX is the heaviest long, so it is the one the repair reaches for - as it
    # did live. The rest are priced at $1 a contract so the book reads in
    # dollars: longs $1,758.59, shorts $1,618.68, which is the measured
    # $139.91 net on $3,377.27 gross.
    unit = Decimal("339.428571428571")          # TRX: $ per contract
    book = {"TRX-USDT": Decimal("1.25"),
            "L2": Decimal("400"), "L3": Decimal("400"),
            "L4": Decimal("400"), "L5": Decimal("134.30"),
            "S1": Decimal("-600"), "S2": Decimal("-600"),
            "S3": Decimal("-418.68")}
    unit_usd = {name: Decimal("1") for name in book}
    unit_usd["TRX-USDT"] = unit
    lot = Rules(contract_value=Decimal("1"), lot_size=Decimal("0.01"),
                min_size=Decimal("0.01"))
    rules = {name: lot for name in book}
    signed = sum(size * unit_usd[name] for name, size in book.items())
    gross = sum(abs(size) * unit_usd[name] for name, size in book.items())
    assert round(signed, 2) == Decimal("139.91")
    assert round(gross, 2) == Decimal("3377.27")

    # The floor the executor passes: a quarter of the tolerance band, below
    # which further trims are chasing rounding dust at a taker fee each.
    floor = Decimal("0.02") * gross / 4
    trims = repair_trims(book, unit_usd, rules, signed, floor)
    assert len(trims) == 1
    trim = trims[0]
    assert trim.inst_id == "TRX-USDT" and trim.side == "sell"
    assert trim.reduce_only

    # The old code sized 0.21 contracts, worth $71.28 against a $139.91
    # imbalance - it aimed at the edge and then rounded away from it.
    assert trim.contracts == Decimal("0.41")
    assert Decimal("139") < trim.notional_usd < Decimal("140")

    left = abs(signed - trim.notional_usd)
    assert left / (gross - trim.notional_usd) < Decimal("0.005"), \
        "the repair left the book outside a quarter of the tolerance"


def test_a_repair_trims_the_nearest_lot_rather_than_giving_up_below_one():
    """Rounding a trim DOWN cannot remove an imbalance smaller than one lot.

    Those are exactly the imbalances that survive to be a problem - a big one
    gets trimmed on the first pass. Nearest overshoots by at most half a lot's
    notional and leaves the book closer to neutral than doing nothing, which is
    what rounding down does here.
    """
    from trading.strategies.carry_xs.execute import repair_trims

    # $60 of imbalance on an instrument whose lot is worth $50.
    book = {"A": Decimal("10"), "B": Decimal("-9.4")}
    unit_usd = {"A": Decimal("100"), "B": Decimal("100")}
    rules = {"A": Rules(contract_value=Decimal("1"), lot_size=Decimal("0.5"),
                        min_size=Decimal("0.5")),
             "B": Rules(contract_value=Decimal("1"), lot_size=Decimal("0.5"),
                        min_size=Decimal("0.5"))}
    signed = Decimal("60")
    trims = repair_trims(book, unit_usd, rules, signed)
    assert len(trims) == 1
    # 60 / 50 = 1.2 steps -> nearest is 1 step = 0.5 contracts = $50.
    assert trims[0].contracts == Decimal("0.5")
    assert abs(signed - trims[0].notional_usd) < signed


def test_a_repair_never_trims_more_than_the_position_it_is_trimming():
    """`reduce_only` would reject the excess anyway, but a size the exchange
    refuses is a repair that silently does nothing at all."""
    from trading.strategies.carry_xs.execute import repair_trims

    book = {"A": Decimal("0.3")}
    unit_usd = {"A": Decimal("100")}
    rules = {"A": Rules(contract_value=Decimal("1"), lot_size=Decimal("0.1"),
                        min_size=Decimal("0.1"))}
    trims = repair_trims(book, unit_usd, rules, Decimal("5000"))
    assert len(trims) == 1
    assert trims[0].contracts == Decimal("0.3")


def test_the_rehearsal_and_the_live_repair_size_the_same_trims():
    """A dry run that priced trims differently from the run it rehearses is
    worse than no dry run: it is a rehearsal of something else.

    They shared a shape but not an implementation until the two copies of the
    sizing were collapsed onto `repair_trims`.
    """
    import trading.strategies.carry_xs.execute as ex

    lopsided = {"A-USDT": Decimal("40"), "B-USDT": Decimal("-10")}
    broker = FakeBroker(positions=dict(lopsided))

    dry = BookExecutor(broker, dry_run=True, settle_seconds=0).repair_only()
    sizes = [(o.inst_id, o.side, o.contracts) for o in dry.orders]
    assert sizes, "the rehearsal priced no trims on a lopsided book"

    seen = []
    original = ex.repair_trims

    def spy(book, unit_usd, rules, signed, stop_usd=Decimal("0")):
        trims = original(book, unit_usd, rules, signed, stop_usd)
        seen.append([(o.inst_id, o.side, o.contracts) for o in trims])
        return trims

    ex.repair_trims = spy
    try:
        live = FakeBroker(positions=dict(lopsided))
        BookExecutor(live, dry_run=False, settle_seconds=0).repair_only()
    finally:
        ex.repair_trims = original
    assert seen and seen[0] == sizes


def test_nothing_to_do_says_which_nothing():
    """Three different situations used to print the same sentence.

    An empty account, a book already inside tolerance, and a book that matches
    the plan all end in `already_correct`, and reporting all three as "the book
    already matches the plan" tells the operator the opposite of the truth in
    the first case - a repair that found no positions is not a book in good
    order, it is an account that may have been closed out from under it.
    """
    empty = BookExecutor(FakeBroker(), dry_run=False,
                         settle_seconds=0).repair_only()
    assert empty.already_correct
    assert "no open positions" in empty.summary.lower()

    balanced = BookExecutor(
        FakeBroker(positions={"A-USDT": Decimal("10"), "B-USDT": Decimal("-10")}),
        dry_run=False, settle_seconds=0).repair_only()
    assert balanced.already_correct
    assert "tolerance" in balanced.summary.lower()
    assert "no open positions" not in balanced.summary.lower()


def test_a_repair_stops_once_the_book_is_neutral_enough_to_stop_paying_fees():
    """Aiming at zero without a floor walks the whole heavy side.

    The first run of the fixed sizing took $139.17 off with one trim and then
    sized a SECOND order for the remaining $0.74 - 0.02% of gross, at a taker
    fee. The floor is a stopping rule and never a target; that distinction is
    the bug this whole section exists for.
    """
    from trading.strategies.carry_xs.execute import repair_trims

    book = {"A": Decimal("100"), "B": Decimal("60"), "C": Decimal("-159")}
    unit_usd = {name: Decimal("1") for name in book}
    rule = Rules(contract_value=Decimal("1"), lot_size=Decimal("1"),
                 min_size=Decimal("1"))
    rules = {name: rule for name in book}

    without = repair_trims(book, unit_usd, rules, Decimal("1"), Decimal("0"))
    assert [order.inst_id for order in without] == ["A"]

    # A $1 imbalance on $319 of gross is 0.3%, well inside a 2% tolerance.
    floor = Decimal("0.02") * Decimal("319") / 4
    assert repair_trims(book, unit_usd, rules, Decimal("1"), floor) == []
