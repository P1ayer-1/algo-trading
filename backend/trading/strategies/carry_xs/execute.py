"""Moves the book from whatever is on the exchange to what the plan wants.

One verb, `reconcile`, not an open and a close. That is the substantive
difference from `strategies/carry`, and it follows from the strategy rather
than from taste: this book is REBALANCED weekly, so "open" is the special case
where the exchange holds nothing and "close" is the one where the target is
empty. Writing those as separate verbs would give three code paths for one
operation and two of them would be exercised once a month.

It also makes the operation idempotent, which matters more here than in a
two-leg carry. Twelve legs is twelve chances to be interrupted, and the answer
to "what happens if it dies halfway" has to be "run it again". Every decision
is taken from positions read back from the EXCHANGE, never from a record of
what this process believed it had done.

The directional window, and how it is bounded
---------------------------------------------
A dollar-neutral book of twelve legs is not neutral between the first fill and
the last. The whole point of the strategy is that it carries no market
exposure, so an ordering that sends six buys and then six sells would spend
the middle of the operation holding exactly the position the strategy exists
to avoid.

So orders are interleaved: at each step the next order comes from whichever
side the filled book is currently short of, largest first within a side. That
bounds the running net exposure by the size of ONE leg rather than by half the
book - with twelve equal legs, a sixth of gross rather than a whole side of it.
`max_net_seen` is reported so the bound is a measurement and not a promise.

What happens when a leg fails, and why it is not the carry's answer
-------------------------------------------------------------------
`strategies/carry` unwinds: a half-on carry is a naked directional position
nobody decided to take, and it closes the leg that filled. This must NOT
inherit that.

Unwinding a twelve-leg rebalance means closing eleven positions the plan
explicitly wants, because the twelfth was rejected - paying a full round trip
to undo a book that is 92% correct. The imbalance from one missing leg is a
fraction of gross; the cure would be worse than the disease.

Instead, a failed leg leaves an imbalance that is REPAIRED: positions are read
back, the realised net exposure computed, and if it exceeds tolerance the
heavier side is REDUCED - `reduce_only`, so the repair can only ever shrink
risk and can never open a position nobody planned. Then it says so, loudly. A
book left directional beyond tolerance is reported as a problem whether or not
the repair succeeded.

Dry run by default
------------------
`dry_run=True` unless a caller explicitly says otherwise. Every step reports
what it would send and returns the same result shape, so the difference between
a rehearsal and the real thing is one flag and nothing else.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .plan import BookPlan

ZERO = Decimal("0")


class Broker(Protocol):
    """The exchange operations a reconcile needs. Implemented for real in
    `broker.py`, and faked in the tests, so none of the logic below has ever
    needed a network to be exercised."""

    def margin_mode(self) -> str: ...

    def set_leverage(self, inst_id: str, leverage: Decimal) -> Dict[str, Any]: ...

    def place_perp(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str,
                   reduce_only: bool = False) -> Dict[str, Any]: ...

    def positions(self) -> List[Dict[str, Any]]: ...

    def instrument_rules(self) -> Dict[str, Any]: ...

    def quote(self, inst_id: str) -> Optional[Dict[str, Decimal]]: ...


@dataclass
class Order:
    """One order to send, or one that would have been sent."""

    inst_id: str
    side: str                     # "buy" / "sell"
    contracts: Decimal
    reduce_only: bool
    reason: str                   # "open" / "increase" / "reduce" / "repair"
    notional_usd: Decimal = ZERO
    client_order_id: str = ""
    sent: bool = False
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    net_after: Decimal = ZERO     # running net exposure once this one fills

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def signed(self) -> Decimal:
        return self.contracts if self.side == "buy" else -self.contracts


@dataclass
class ExecutionResult:
    """What happened, or what would have. Conforms to the `Outcome` Protocol."""

    inst_id: str = "carry_xs"
    dry_run: bool = True
    orders: List[Order] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    reconciled: bool = False
    already_correct: bool = False
    max_net_usd: Decimal = ZERO           # worst directional exposure en route
    gross_before_usd: Decimal = ZERO
    gross_after_usd: Decimal = ZERO
    net_after_usd: Decimal = ZERO
    repaired: bool = False

    def add(self, order: Order) -> Order:
        self.orders.append(order)
        return order

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def failed(self) -> List[Order]:
        return [order for order in self.orders if not order.ok]


def _decimal(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return Decimal(default)


def _filled(response: Optional[Dict[str, Any]]) -> bool:
    """Did the order actually reach the book?

    BloFin returns a per-order `code` INSIDE `data`, so a transport-level
    success carrying a rejected order looks like success at the top level. That
    is the failure mode worth catching, and it is the same check
    `strategies/carry` makes for the same reason.
    """
    if not isinstance(response, dict):
        return False
    if str(response.get("code")) not in ("0", "None", ""):
        return False
    rows = response.get("data")
    if isinstance(rows, list) and rows:
        first = rows[0]
        if isinstance(first, dict) and str(first.get("code", "0")) not in ("0", ""):
            return False
    return True


def current_book(broker: Broker) -> Dict[str, Decimal]:
    """`{inst_id: signed contracts}` from the exchange.

    BloFin reports a net position's direction in the SIGN of `positions`, and a
    `positionSide` of "net". Reading the magnitude and assuming a side is how a
    short gets doubled instead of closed, so the sign is taken from the number
    itself and nothing else.
    """
    book: Dict[str, Decimal] = {}
    for row in broker.positions():
        inst_id = str(row.get("instId") or "")
        if not inst_id:
            continue
        size = _decimal(row.get("positions"))
        if size != 0:
            book[inst_id] = book.get(inst_id, ZERO) + size
    return book


def target_book(plan: BookPlan) -> Dict[str, Decimal]:
    """`{inst_id: signed contracts}` the plan wants. Only legs that sized."""
    target: Dict[str, Decimal] = {}
    for leg in plan.legs:
        if leg.ok and leg.contracts > 0:
            target[leg.inst_id] = (leg.contracts if leg.side == "buy"
                                   else -leg.contracts)
    return target


def round_to_lot(contracts: Decimal, lot: Decimal) -> Decimal:
    """Down to a whole lot. Anything sent to the exchange goes through here.

    The first live run rejected four repair orders with `152002 Parameter size
    error` because the repair sized itself as `excess / unit` and sent the
    result raw - `0.2276622802836378615352929061` contracts. Rounding at the
    point of sizing is not enough; it has to happen on the way OUT, because
    every future code path that computes a size will otherwise have to remember.
    """
    if lot <= 0:
        return contracts
    return (contracts / lot).to_integral_value(rounding=ROUND_DOWN) * lot


def unit_values(plan: BookPlan, contract_values: Dict[str, Decimal],
                marks: Optional[Dict[str, Decimal]] = None) -> Dict[str, Decimal]:
    """`{inst_id: USD per contract}` = contract value x price.

    This is the only unit anything here may size, balance or repair on. A
    CONTRACT is a different amount of money on every instrument - on BloFin
    0.001 BTC against 1000 DOGE, a factor of a million between two legs of the
    same book - so an interleaving that balanced contract counts would balance
    nothing, and a repair sized in contracts would trim the wrong amount by
    that factor.

    The plan's own legs come first because their price is the one the plan was
    costed at. Positions the plan does not mention - a leftover from an earlier
    run - are priced from `marks`, so they can still be closed.
    """
    out: Dict[str, Decimal] = {}
    for inst_id, price in (marks or {}).items():
        value = contract_values.get(inst_id)
        if value and price > 0:
            out[inst_id] = value * price
    for leg in plan.legs:
        if leg.price > 0 and leg.contract_value > 0:
            out[leg.inst_id] = leg.contract_value * leg.price
    return out


def plan_orders(current: Dict[str, Decimal], target: Dict[str, Decimal],
                unit_usd: Dict[str, Decimal],
                rules: Optional[Dict[str, Any]] = None) -> List[Order]:
    """The deltas, interleaved so the running net exposure stays small.

    An order is `reduce_only` whenever it moves a position TOWARD zero without
    crossing it, which is every close and every trim. Flipping a position from
    long to short cannot be reduce_only - it would be rejected past the
    crossing point - so a flip is split into a reduce_only close and a fresh
    open, which is also the only way to be sure a rejected second half leaves
    the book flat in that name rather than still long.
    """
    raw: List[Order] = []
    for inst_id in sorted(set(current) | set(target)):
        have, want = current.get(inst_id, ZERO), target.get(inst_id, ZERO)
        if have == want:
            continue
        unit = unit_usd.get(inst_id, ZERO)

        rule = (rules or {}).get(inst_id)

        def make(delta: Decimal, reduce_only: bool, reason: str) -> None:
            if delta == 0:
                return
            if rule is not None:
                # A CLOSE is exempt from the minimum: the exchange lets a
                # position be closed whatever its size, and refusing to shut
                # one because it is small would strand it forever.
                size = round_to_lot(abs(delta), rule.lot_size)
                closing = reduce_only and abs(delta) == abs(have)
                if size <= 0 or (size < rule.min_size and not closing):
                    return
                delta = size if delta > 0 else -size
            raw.append(Order(
                inst_id=inst_id, side="buy" if delta > 0 else "sell",
                contracts=abs(delta), reduce_only=reduce_only, reason=reason,
                notional_usd=abs(delta) * unit))

        if have == 0:
            make(want, False, "open")
        elif want == 0:
            make(-have, True, "reduce")
        elif (have > 0) != (want > 0):
            # A flip cannot be one order: `reduce_only` is rejected past the
            # crossing point, and a plain order that is rejected half way
            # leaves the position still on the old side. Two orders make the
            # failure mode "flat in this name", which is the safe one.
            make(-have, True, "reduce")
            make(want, False, "open")
        elif abs(want) < abs(have):
            make(want - have, True, "reduce")
        else:
            make(want - have, False, "increase")

    starting_net = sum((size * unit_usd.get(inst_id, ZERO)
                        for inst_id, size in current.items()), ZERO)
    return _interleave(raw, starting_net)


def _interleave(orders: Sequence[Order], starting_net: Decimal = ZERO,
                ) -> List[Order]:
    """Order the sends so the BOOK's net exposure stays near zero.

    At each step take the next order from whichever side the book is currently
    long of, largest notional first within a side. With twelve roughly equal
    legs this bounds the worst intermediate exposure at about a sixth of gross;
    sending all the buys and then all the sells would take it to half.

    `starting_net` is the exposure the account ALREADY carries, and the running
    total starts there rather than at zero. Starting at zero balances the order
    flow instead of the book, which is the same thing only when the account
    begins flat - and on a weekly rebalance it never does. A book already long
    $69,000 needs sells first; a counter that starts at zero would cheerfully
    send a buy.

    Notional, not contracts: a contract is a different amount of money on every
    instrument, so balancing contract counts would balance nothing.
    """
    buys = sorted([o for o in orders if o.side == "buy"],
                  key=lambda o: o.notional_usd, reverse=True)
    sells = sorted([o for o in orders if o.side == "sell"],
                   key=lambda o: o.notional_usd, reverse=True)

    out: List[Order] = []
    running = starting_net
    while buys or sells:
        if not sells or (buys and running <= 0):
            order = buys.pop(0)
            running += order.notional_usd
        else:
            order = sells.pop(0)
            running -= order.notional_usd
        order.net_after = running
        out.append(order)
    return out


class BookExecutor:
    """Turns a `BookPlan` into orders. Dry by default."""

    def __init__(self, broker: Broker, *, dry_run: bool = True,
                 on_log: Optional[Callable[[str], Any]] = None,
                 settle_seconds: float = 1.0,
                 sleep: Callable[[float], Any] = time.sleep,
                 max_spread_bps: float = 30.0,
                 net_tolerance_frac: float = 0.02):
        self.broker = broker
        self.dry_run = dry_run
        self.on_log = on_log or (lambda message: None)
        self.settle_seconds = settle_seconds
        self.sleep = sleep
        self.max_spread_bps = max_spread_bps
        self.net_tolerance_frac = net_tolerance_frac

    def log(self, message: str) -> None:
        self.on_log(message)

    # -- checks ------------------------------------------------------------

    def _spread_gate(self, order: Order, result: ExecutionResult) -> bool:
        """Refuse a leg whose spread has blown out since the plan was made.

        A plan is minutes old by the time it executes and the spread on a thin
        alt is the largest term in its cost - the live run in step 9s found
        NEAR quoting 16.9 bps against a 5 bps fee. Crossing a spread that has
        doubled since the plan priced it turns a positive expected trade into a
        negative one silently, so the quote is re-read immediately before
        sending. A REDUCE is exempt: getting out is not optional, and refusing
        to shrink risk because it is expensive is how a book stays broken.
        """
        if order.reduce_only:
            return True
        quote = self.broker.quote(order.inst_id)
        if quote is None:
            order.error = "no quote available"
            result.problems.append(order.inst_id + ": no quote, leg skipped")
            return False
        mid = (quote["bid"] + quote["ask"]) / 2
        spread_bps = float((quote["ask"] - quote["bid"]) / mid) * 10_000.0
        if spread_bps > self.max_spread_bps:
            order.error = "spread {:.1f} bps over the {:.1f} limit".format(
                spread_bps, self.max_spread_bps)
            result.problems.append(
                "{}: spread {:.1f} bps exceeds the {:.1f} bps limit; leg skipped "
                "and the book is short by ${:,.2f}".format(
                    order.inst_id, spread_bps, self.max_spread_bps,
                    order.notional_usd))
            return False
        return True

    # -- the verb ----------------------------------------------------------

    def reconcile(self, plan: BookPlan, *, book_tag: Optional[str] = None,
                  ) -> ExecutionResult:
        """Move the exchange's positions to the plan's. Idempotent."""
        result = ExecutionResult(dry_run=self.dry_run)
        tag = book_tag or uuid.uuid4().hex[:8]

        if not plan.ok:
            result.problems.append(
                "the plan was refused, so there is nothing to send: "
                + "; ".join(plan.reasons))
            return result

        current, marks = current_book(self.broker), {}
        rows = self.broker.positions()
        for row in rows:
            mark = _decimal(row.get("markPrice"))
            if mark > 0:
                marks[str(row.get("instId") or "")] = mark
        target = target_book(plan)
        rules = self.broker.instrument_rules()
        unit_usd = unit_values(
            plan, {name: rule.contract_value for name, rule in rules.items()},
            marks)

        # An instrument the ACCOUNT'S host does not list cannot be traded on
        # it, whatever the plan says. Demo lists 87 against production's 488,
        # so a plan built on production prices will name instruments this
        # account has never heard of, and they would each fail one at a time.
        missing = sorted(name for name in target if name not in rules)
        for name in missing:
            target.pop(name, None)
            result.problems.append(
                name + " is not listed on this account's host, so it cannot be "
                "traded here; the book is short by that leg")

        result.gross_before_usd = sum(
            (abs(size) * unit_usd.get(inst_id, ZERO)
             for inst_id, size in current.items()), ZERO)

        orders = plan_orders(current, target, unit_usd, rules)
        if not orders:
            result.already_correct = True
            result.reconciled = True
            self.log("The book already matches the plan. Nothing to send.")
            return result

        for index, order in enumerate(orders):
            order.client_order_id = "xs{}{:02d}".format(tag, index)
        result.orders = list(orders)
        result.max_net_usd = max((abs(order.net_after) for order in orders),
                                 default=ZERO)

        self.log("{} orders; worst intermediate net exposure ${:,.2f}".format(
            len(orders), result.max_net_usd))

        for order in orders:
            if not self._spread_gate(order, result):
                continue
            detail = "{} {} {} contracts (${:,.2f}, {})".format(
                order.reason, order.side, order.contracts, order.notional_usd,
                "reduce_only" if order.reduce_only else "opening")
            if self.dry_run:
                self.log("  WOULD SEND  " + order.inst_id + ": " + detail)
                continue
            self.log("  SENDING     " + order.inst_id + ": " + detail)
            try:
                order.response = self.broker.place_perp(
                    inst_id=order.inst_id, side=order.side,
                    size=order.contracts, client_order_id=order.client_order_id,
                    reduce_only=order.reduce_only)
            except Exception as exc:               # noqa: BLE001
                order.error = str(exc)
                result.problems.append(order.inst_id + ": " + str(exc))
                continue
            if not _filled(order.response):
                order.error = "rejected: " + str(order.response)
                result.problems.append(
                    order.inst_id + " was rejected by the exchange")
                continue
            order.sent = True

        if self.dry_run:
            self.log("Dry run: nothing was sent.")
            return result

        if self.settle_seconds:
            self.sleep(self.settle_seconds)
        self._verify_and_repair(unit_usd, rules, result, tag)
        return result

    # -- after ------------------------------------------------------------

    def _verify_and_repair(self, unit_usd: Dict[str, Decimal],
                           rules: Dict[str, Any], result: ExecutionResult,
                           tag: str) -> None:
        """Read the book back and, if it is directional, shrink the heavy side.

        Verification is against the EXCHANGE rather than against the orders
        this process believes it sent, because a plan that agrees with itself
        has established nothing. The repair is `reduce_only` without exception:
        it can shrink risk and can never open a position nobody planned, which
        is the property that makes it safe to run automatically after a
        partially failed batch.
        """
        book = current_book(self.broker)
        signed = ZERO
        gross = ZERO
        for inst_id, size in book.items():
            unit = unit_usd.get(inst_id, ZERO)
            if unit <= 0:
                # A position the plan never mentioned and whose contract value
                # is unknown cannot be priced, so it cannot be counted toward
                # the net either. Saying so beats silently treating it as zero
                # exposure, which is what an unpriced leg looks like.
                result.warnings.append(
                    inst_id + " could not be priced and is excluded from the "
                    "net-exposure check")
                continue
            signed += size * unit
            gross += abs(size) * unit
        result.gross_after_usd = gross
        result.net_after_usd = signed

        if gross <= 0:
            result.reconciled = not result.problems
            return

        drift = abs(signed) / gross
        if drift <= Decimal(str(self.net_tolerance_frac)):
            result.reconciled = not result.problems
            return

        result.warnings.append(
            "book is ${:,.2f} net on ${:,.2f} gross ({:.2%}); reducing the "
            "heavy side".format(signed, gross, drift))
        self.log(result.warnings[-1])

        # Trim the heaviest positions on the over-weighted side until the net is
        # inside tolerance. Largest first, so the fewest orders do it.
        heavy_side_is_long = signed > 0
        excess = abs(signed) - Decimal(str(self.net_tolerance_frac)) * gross
        candidates = sorted(
            ((inst_id, size) for inst_id, size in book.items()
             if (size > 0) == heavy_side_is_long and size != 0),
            key=lambda item: abs(item[1]) * unit_usd.get(item[0], ZERO),
            reverse=True)

        for position, (inst_id, size) in enumerate(candidates):
            if excess <= 0:
                break
            unit = unit_usd.get(inst_id, ZERO)
            if unit <= 0:
                continue
            rule = rules.get(inst_id)
            trim = min(abs(size), excess / unit)
            if rule is not None:
                trim = round_to_lot(trim, rule.lot_size)
                # A trim below the minimum cannot be sent. Skipping it and
                # moving to the next position is right: the repair is a
                # best-effort shrink, and the final check below reports
                # whatever it could not fix rather than pretending it did.
                if trim < rule.min_size:
                    continue
            if trim <= 0:
                continue
            order = result.add(Order(
                inst_id=inst_id, side="sell" if size > 0 else "buy",
                contracts=trim, reduce_only=True, reason="repair",
                notional_usd=trim * unit,
                client_order_id="xr{}{:02d}".format(tag, position)))
            self.log("  REPAIR      {}: {} {} contracts".format(
                inst_id, order.side, trim))
            try:
                order.response = self.broker.place_perp(
                    inst_id=inst_id, side=order.side, size=trim,
                    client_order_id=order.client_order_id, reduce_only=True)
            except Exception as exc:               # noqa: BLE001
                order.error = str(exc)
                result.problems.append("repair of " + inst_id + ": " + str(exc))
                continue
            if not _filled(order.response):
                order.error = "rejected: " + str(order.response)
                result.problems.append("repair of " + inst_id + " was rejected")
                continue
            order.sent = True
            result.repaired = True
            excess -= trim * unit

        book = current_book(self.broker)
        signed = sum((size * unit_usd.get(inst_id, ZERO)
                      for inst_id, size in book.items()), ZERO)
        gross = sum((abs(size) * unit_usd.get(inst_id, ZERO)
                     for inst_id, size in book.items()), ZERO)
        result.net_after_usd = signed
        result.gross_after_usd = gross
        if gross > 0 and abs(signed) / gross > Decimal(str(self.net_tolerance_frac)):
            result.problems.append(
                "book is STILL ${:,.2f} net on ${:,.2f} gross after repair. It "
                "is carrying market exposure the strategy does not want; close "
                "it or fix it by hand.".format(signed, gross))
        result.reconciled = not result.problems
