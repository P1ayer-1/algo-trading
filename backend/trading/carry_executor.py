"""Opens a planned carry: perp leg first, spot leg second, unwind if either fails.

`carry.py` decides what the position should be. This puts it on, and its real
job is the two minutes in the middle where only one leg exists.

Order of operations, and why
----------------------------
**Perp first.** Between sending the two legs the position is directional, so
the question is which exposure you would rather be holding when something goes
wrong. Sending the perp first means a failed spot leg leaves you SHORT the
perp - a position that closes instantly, on the venue's deepest book, with a
`reduce_only` order the risk engine will always approve. Sending spot first
means a failed perp leg leaves you LONG spot, unwindable only by selling into
the wider book, and holding an asset rather than a contract.

The perp leg also carries the liquidation risk and the tighter spread, so it
is the leg whose fill price matters more and the one worth getting on while
the plan's prices are freshest.

**Unwind, never hope.** If the spot leg fails after the perp filled, this
closes the perp immediately rather than retrying the spot. A carry that is
half on is not a carry, it is a naked directional position nobody decided to
take, and the longer it lives the more it stops being an execution problem and
starts being a trading one.

**Verify against the exchange, not against the plan.** After both legs are on,
the actual position is read back and its liquidation price compared to the
planned one. That check exists because maintenance margin is TIERED and
instrument-specific - measured at 0.500% on SOL-USDT and 0.300% on BTC-USDT -
so the rate a plan assumed is a guess until the exchange prices the position
you actually opened. A real liquidation closer than planned is a reason to
know immediately, not at 3am.

Dry run by default
------------------
`dry_run=True` unless a caller explicitly says otherwise. Every step reports
what it would send and returns the same result shape, so the difference
between a rehearsal and the real thing is one flag and nothing else.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol

from .carry import CarryPlan

ZERO = Decimal("0")


class Broker(Protocol):
    """The exchange operations an open needs. Implemented for real in
    `run_carry.py`, and faked in the tests, so none of the logic below has
    ever needed a network to be exercised."""

    def transfer(self, *, currency: str, amount: Decimal,
                 from_account: str, to_account: str) -> Dict[str, Any]: ...

    def margin_mode(self) -> str: ...

    def set_leverage(self, inst_id: str, leverage: Decimal) -> Dict[str, Any]: ...

    def place_perp(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str,
                   reduce_only: bool = False) -> Dict[str, Any]: ...

    def place_spot(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str) -> Dict[str, Any]: ...

    def perp_position(self, inst_id: str) -> Optional[Dict[str, Any]]: ...

    def spot_balance(self, currency: str) -> Decimal: ...


@dataclass
class Step:
    """One thing done or one thing that would have been done."""

    name: str
    detail: str
    sent: bool = False
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ExecutionResult:
    inst_id: str
    dry_run: bool
    steps: List[Step] = field(default_factory=list)
    opened: bool = False
    unwound: bool = False
    problems: List[str] = field(default_factory=list)

    perp_position: Optional[Dict[str, Any]] = None
    planned_liquidation: Optional[Decimal] = None
    actual_liquidation: Optional[Decimal] = None
    actual_mmr: Optional[Decimal] = None
    spot_acquired: Optional[Decimal] = None

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    @property
    def ok(self) -> bool:
        return not self.problems


def _decimal(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return None


def _filled(response: Optional[Dict[str, Any]]) -> bool:
    """Did an order actually result in a position or balance change?

    BloFin returns a per-order `code` inside `data`; a transport-level success
    with a rejected order is the failure mode worth catching, because it looks
    like success at the top level.
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


class CarryExecutor:
    """Turns a `CarryPlan` into orders. Dry by default."""

    def __init__(self, broker: Broker, *, dry_run: bool = True,
                 on_log: Optional[Callable[[str], Any]] = None,
                 settle_seconds: float = 1.0,
                 sleep: Callable[[float], Any] = time.sleep):
        self.broker = broker
        self.dry_run = dry_run
        self.on_log = on_log or (lambda message: None)
        self.settle_seconds = settle_seconds
        self.sleep = sleep

    def log(self, message: str) -> None:
        self.on_log(message)

    def _do(self, result: ExecutionResult, name: str, detail: str,
            action: Callable[[], Dict[str, Any]]) -> Step:
        step = Step(name=name, detail=detail)
        self.log(f"  {'[dry] ' if self.dry_run else ''}{name}: {detail}")
        if self.dry_run:
            return result.add(step)
        try:
            step.response = action()
            step.sent = True
            if not _filled(step.response):
                step.error = f"rejected: {step.response}"
        except Exception as exc:  # noqa: BLE001 - reported, never raised through
            step.error = f"{type(exc).__name__}: {exc}"
        return result.add(step)

    def open(self, plan: CarryPlan, *, base_currency: str) -> ExecutionResult:
        """Put the planned carry on, or leave nothing behind trying."""
        result = ExecutionResult(inst_id=plan.inst_id, dry_run=self.dry_run,
                                 planned_liquidation=plan.liquidation_price)

        if not plan.ok:
            result.problems.append(
                "plan was refused; nothing to execute: "
                + "; ".join(plan.reasons))
            return result

        tag = uuid.uuid4().hex[:16]
        # Recorded before anything is sent, so the spot leg can be verified by
        # what the BALANCE did rather than by what the order response said.
        # `size` on a spot market order means base units or quote depending on
        # a parameter the venue lets you omit, so a fill that reports success
        # is not yet evidence the hedge is the right size.
        base_before = (ZERO if self.dry_run
                       else self.broker.spot_balance(base_currency))

        # ---- funding the wallets ------------------------------------------
        if plan.spot_transfer_usd > 0:
            self._do(result, "transfer",
                     f"${plan.spot_transfer_usd:.2f} USDT futures -> spot",
                     lambda: self.broker.transfer(
                         currency="USDT", amount=plan.spot_transfer_usd,
                         from_account="futures", to_account="spot"))
        if plan.futures_transfer_usd > 0:
            self._do(result, "transfer",
                     f"${plan.futures_transfer_usd:.2f} USDT spot -> futures",
                     lambda: self.broker.transfer(
                         currency="USDT", amount=plan.futures_transfer_usd,
                         from_account="spot", to_account="futures"))

        if any(not step.ok for step in result.steps):
            result.problems.append("wallet transfer failed; no orders sent")
            return result

        # ---- margin configuration -----------------------------------------
        # Margin mode is checked, never set. On BloFin it is an ACCOUNT-wide
        # setting, so flipping it here to suit one carry would silently
        # re-margin every other open position - the sort of side effect an
        # execution path must not have. If it is wrong, say so and stop.
        mode = self.broker.margin_mode()
        if mode != "isolated":
            result.problems.append(
                f"account margin mode is '{mode}', and the plan priced "
                f"isolated. Changing it is an ACCOUNT-wide switch that would "
                f"re-margin every other open position, so it is not done "
                f"here - switch it deliberately, then re-run.")
            return result
        self.log(f"  margin mode: {mode} (checked, not changed)")

        self._do(result, "leverage", f"{plan.inst_id} -> {plan.leverage}x",
                 lambda: self.broker.set_leverage(plan.inst_id, plan.leverage))

        if any(not step.ok for step in result.steps):
            result.problems.append(
                "could not set leverage; no orders sent")
            return result

        # ---- leg 1: the perp short ----------------------------------------
        perp_step = self._do(
            result, "perp",
            f"SELL {plan.perp_contracts} contracts, isolated {plan.leverage}x",
            lambda: self.broker.place_perp(
                inst_id=plan.inst_id, side="sell",
                size=plan.perp_contracts,
                client_order_id=f"carry{tag}p"))

        if not perp_step.ok:
            # Nothing is on. This is the cheap failure and the reason the perp
            # goes first.
            result.problems.append(f"perp leg failed: {perp_step.error}")
            return result

        # ---- leg 2: the spot buy ------------------------------------------
        spot_step = self._do(
            result, "spot", f"BUY {plan.spot_base} {base_currency}",
            lambda: self.broker.place_spot(
                inst_id=plan.inst_id, side="buy", size=plan.spot_base,
                client_order_id=f"carry{tag}s"))

        if not spot_step.ok:
            result.problems.append(f"spot leg failed: {spot_step.error}")
            self._unwind(result, plan)
            return result

        result.opened = True

        # ---- verification against the exchange -----------------------------
        if not self.dry_run:
            self.sleep(self.settle_seconds)
            self._verify(result, plan)
            self._verify_spot(result, plan, base_currency, base_before)
        return result

    def _unwind(self, result: ExecutionResult, plan: CarryPlan) -> None:
        """Close the perp leg, because half a carry is a directional bet.

        `reduce_only`, so the risk engine approves it even with the kill
        switch tripped - getting out is never the thing to block.
        """
        self.log("  spot leg failed - unwinding the perp leg")
        step = self._do(
            result, "unwind",
            f"BUY {plan.perp_contracts} contracts, reduce_only",
            lambda: self.broker.place_perp(
                inst_id=plan.inst_id, side="buy", size=plan.perp_contracts,
                client_order_id=f"carry{uuid.uuid4().hex[:12]}u",
                reduce_only=True))
        if step.ok:
            result.unwound = True
        else:
            result.problems.append(
                "UNWIND FAILED - a naked short perp is open on "
                f"{plan.inst_id}. Close it by hand now.")

    def _verify_spot(self, result: ExecutionResult, plan: CarryPlan,
                     base_currency: str, before: Decimal) -> None:
        """Did the spot leg actually buy the amount the hedge needs?

        Checked against the balance rather than the order response, because
        the failure this guards is a units mismatch: `targetCurrency` decides
        whether a market order's `size` is base or quote, and the wrong one
        fills successfully at the wrong size. An order response cannot show
        that; a balance can.
        """
        after = self.broker.spot_balance(base_currency)
        acquired = after - before
        result.spot_acquired = acquired

        if plan.spot_base <= 0:
            return
        drift = abs(acquired - plan.spot_base) / plan.spot_base
        if drift > Decimal("0.02"):
            result.problems.append(
                f"spot leg bought {acquired} {base_currency}, plan needed "
                f"{plan.spot_base} ({drift:.1%} off). The hedge is the wrong "
                f"size - check targetCurrency on the order.")

    def _verify(self, result: ExecutionResult, plan: CarryPlan) -> None:
        """Read the position back and price it the way the exchange does.

        The plan assumed a maintenance margin rate. MMR is tiered and
        instrument-specific, so that assumption is only a guess until a real
        position exists - at which point the exchange publishes both its own
        liquidation price and the maintenance margin it is charging.
        """
        position = self.broker.perp_position(plan.inst_id)
        result.perp_position = position
        if not position:
            result.problems.append(
                "perp order was accepted but no position is reported")
            return

        contracts = _decimal(position.get("positions")) or ZERO
        if contracts >= 0:
            result.problems.append(
                f"expected a short perp position, found {contracts}")

        actual = _decimal(position.get("liquidationPrice"))
        mark = _decimal(position.get("markPrice"))
        maintenance = _decimal(position.get("maintenanceMargin"))
        result.actual_liquidation = actual
        if maintenance and mark and contracts:
            notional = abs(contracts) * mark
            if notional > 0:
                result.actual_mmr = maintenance / notional

        if actual and plan.liquidation_price and mark:
            planned_distance = abs(plan.liquidation_price - mark) / mark
            actual_distance = abs(actual - mark) / mark
            if actual_distance < planned_distance * Decimal("0.95"):
                result.problems.append(
                    f"liquidation is CLOSER than planned: {actual_distance:.2%} "
                    f"actual vs {planned_distance:.2%} planned. The plan's "
                    f"maintenance margin rate was wrong for this instrument.")
