"""Size a cross-sectional carry book. SENDS NOTHING.

The trade, in one line: long the perps whose funding is lowest, short the ones
whose funding is highest, dollar-neutral, and rebalance weekly. Step 9q is the
evidence — +40.5 bps a week net of twice the taker fee over five years of
Binance, a funding leg positive in nine calendar years out of nine across two
venues, and 12 of 12 random halves of the universe positive.

How this differs from `strategies/carry`, which is also a funding trade
----------------------------------------------------------------------
That one is long spot and short the perp on ONE instrument. It has no price
exposure at all, and three constraints that come from its spot leg: no borrow,
so only positive funding is harvestable; a spot spread of 4-50 bps; and the
full notional tied up in spot.

This one is perps on both sides across MANY instruments. It harvests funding of
either sign, needs no spot leg, and needs margin rather than notional. What it
buys with that is a price exposure the other does not have: the legs are
different coins, so nothing cancels, and the book's weekly standard deviation
is about 240 bps. The worst week measured was −1,991 bps, in February 2024,
when the book was short SHIB, PEPE and BONK because they had the highest
funding and the three returned +256%, +288% and +180% in seven days. That is
the trade rather than a flaw in it, and the sizing here exists to make its
scale explicit before it is on.

Step 9v is why `max_weight_frac` is the risk control here and no covariance
model is: in the 90 days before that week those three names had residual
correlations of +0.12, −0.13 and +0.21 and fell into three different clusters.
The correlation that did the damage did not exist in the data beforehand, so a
cap that does not try to estimate it is worth more than one that does.

What this module will not do
----------------------------
Send an order. There is no import of a broker and no path from here to
`placeOrder`, which is checkable with a grep and is checked that way. It also
does not decide leg ordering: a book of thirty legs is directional between the
first fill and the last, and that is an executor's problem, stated here only as
a warning so it cannot be discovered later.

Refusals are plural. A book with four separate problems reports four, because
fixing one and re-running to find the next is a worse loop than being told.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from trading.risk import (
    RiskLimits,
    Side,
    liquidation_distance_pct,
    liquidation_price,
    to_decimal,
)

ZERO = Decimal("0")


@dataclass
class Candidate:
    """One instrument's inputs. Everything here is observable before trading.

    `carry_bps_per_day` is the SCORE, stated the way the research states it:
    negated trailing funding, so high means "expected to outperform" and the
    book goes long it. Passing raw funding here would invert the whole book,
    so the sign lives in the name.
    """

    inst_id: str
    carry_bps_per_day: float
    bid: Decimal
    ask: Decimal
    volume_usd_24h: float
    vol_30d_bps: float
    contract_value: Decimal = Decimal("1")
    lot_size: Decimal = Decimal("1")
    min_size: Decimal = Decimal("1")
    max_leverage: Decimal = Decimal("10")
    funding_days: int = 0

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else ZERO

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        if mid <= 0:
            return float("inf")
        return float((self.ask - self.bid) / mid) * 10_000.0


@dataclass
class LegPlan:
    """One leg. Conforms to the `Plan` Protocol structurally, like every other
    plan object here — a leg that cannot be sized is a refusal in its own
    right, not a silently dropped row."""

    inst_id: str
    ok: bool = False
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    side: str = ""
    contracts: Decimal = ZERO
    base: Decimal = ZERO
    price: Decimal = ZERO
    # USD per contract = contract_value * price. A contract is a different
    # amount of money on every instrument (BloFin: 0.001 BTC, 1000 DOGE), so
    # anything that sizes or balances on contract COUNT balances nothing.
    contract_value: Decimal = Decimal("1")
    notional_usd: Decimal = ZERO
    target_notional_usd: Decimal = ZERO
    carry_bps_per_day: float = 0.0
    spread_bps: float = 0.0
    liquidation_price: Optional[Decimal] = None
    liquidation_distance: Optional[Decimal] = None


@dataclass
class BookPlan:
    """The whole book, and every reason not to put it on.

    `inst_id` is the book's name rather than an instrument, which is what lets
    a multi-leg plan satisfy the same Protocol as a single-instrument one
    without pretending to be one.
    """

    inst_id: str = "carry_xs"
    ok: bool = False
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    legs: List[LegPlan] = field(default_factory=list)
    eligible: int = 0
    considered: int = 0

    gross_notional_usd: Decimal = ZERO
    net_notional_usd: Decimal = ZERO
    margin_usd: Decimal = ZERO
    leverage: Decimal = ZERO

    carry_spread_bps_per_day: float = 0.0
    expected_funding_bps: float = 0.0
    round_trip_bps: float = 0.0
    expected_net_bps: float = 0.0
    expected_net_usd: Decimal = ZERO
    breakeven_days: Optional[float] = None

    @property
    def long_legs(self) -> List[LegPlan]:
        return [leg for leg in self.legs if leg.side == "buy"]

    @property
    def short_legs(self) -> List[LegPlan]:
        return [leg for leg in self.legs if leg.side == "sell"]


@dataclass
class BookConfig:
    """The frozen specification from step 9q, as defaults.

    These are the numbers the evidence was produced at, not tuning knobs: the
    result was measured at a 7-day hold, top and bottom 30%, inverse-volatility
    sizing and a $5M liquidity floor. Changing one moves off the measurement,
    which is allowed and should be deliberate.
    """

    top_frac: float = 0.3
    hold_days: int = 7
    min_volume_usd: float = 5e6
    min_funding_days: int = 30
    min_names_per_side: int = 3
    gross_notional_usd: Decimal = Decimal("2000")
    leverage: Decimal = Decimal("2")
    vol_scale: bool = True
    max_weight_frac: float = 0.25
    delta_tolerance_frac: float = 0.02


def eligible_candidates(candidates: Sequence[Candidate], config: BookConfig,
                        ) -> Tuple[List[Candidate], List[str]]:
    """Who can be traded, and a note for everyone excluded.

    Exclusions are reported rather than silently applied. A book that quietly
    dropped half its universe for want of funding history would look like a
    concentrated book that somebody chose.
    """
    kept: List[Candidate] = []
    notes: List[str] = []
    for candidate in candidates:
        problems = []
        if candidate.volume_usd_24h < config.min_volume_usd:
            problems.append("${:,.0f}/24h under the ${:,.0f} floor".format(
                candidate.volume_usd_24h, config.min_volume_usd))
        if candidate.funding_days < config.min_funding_days:
            problems.append("only {} days of funding history".format(
                candidate.funding_days))
        if candidate.mid <= 0:
            problems.append("no usable quote")
        if candidate.ask <= candidate.bid:
            problems.append("crossed or locked quote")
        if not (candidate.vol_30d_bps > 0):
            problems.append("no volatility estimate")
        if problems:
            notes.append(candidate.inst_id + ": " + "; ".join(problems))
        else:
            kept.append(candidate)
    return kept, notes


def _side_weights(side: Sequence[Candidate], config: BookConfig,
                  ) -> Tuple[Dict[str, float], List[str]]:
    """Fractions of one side's notional, summing to 1.

    Inverse volatility if configured, then capped at `max_weight_frac` and
    renormalised. The cap is not in the backtest and is here deliberately: the
    February 2024 loss was three correlated meme perps on the same side of the
    book, and inverse-vol sizing made it WORSE rather than better, because it
    sizes on trailing volatility and those three were quiet right up until they
    were not. A cap is the one control that does not depend on having estimated
    the risk correctly.
    """
    warnings: List[str] = []
    if config.vol_scale:
        raw = {c.inst_id: 1.0 / c.vol_30d_bps for c in side}
    else:
        raw = {c.inst_id: 1.0 for c in side}

    total = sum(raw.values())
    weights = {key: value / total for key, value in raw.items()}

    cap = config.max_weight_frac
    if cap > 0 and len(side) > 1:
        # With few enough names the equal weight already exceeds the cap, and
        # capping then means nothing: the floor is what every name gets. A
        # tolerance below it stops a three-name side reporting all three as
        # "capped at 33% (uncapped 33%)", which is noise that trains the reader
        # to skip the warnings that matter.
        original = dict(weights)
        floor = 1.0 / len(side)
        effective = max(cap, floor)
        capped: Dict[str, None] = {}
        for _ in range(10):
            over = {k: v for k, v in weights.items()
                    if v > effective * (1.0 + 1e-9) + 1e-12}
            if not over:
                break
            for key in over:
                capped[key] = None
                weights[key] = effective
            spare = 1.0 - sum(weights.values())
            free = [k for k in weights if k not in capped]
            if not free or spare <= 0:
                break
            share = sum(weights[k] for k in free)
            for key in free:
                weights[key] += spare * (weights[key] / share if share > 0
                                         else 1.0 / len(free))
        # One line per capped name, against the weight it started with.
        # Redistribution pushes a name marginally back over the cap, so warning
        # inside the loop reported the same instrument several times and
        # compared it to its already-capped weight - "capped at 25% (uncapped
        # 25%)", which says nothing.
        for key in capped:
            if original[key] > effective * (1.0 + 1e-6):
                warnings.append(
                    "{} capped at {:.0%} of its side (inverse-vol weight was "
                    "{:.0%})".format(key, effective, original[key]))
    return weights, warnings


def _round_to_lot(contracts: Decimal, lot: Decimal) -> Decimal:
    if lot <= 0:
        return contracts
    return (contracts / lot).to_integral_value(rounding="ROUND_DOWN") * lot


def plan_book(candidates: Sequence[Candidate], config: Optional[BookConfig] = None,
              limits: Optional[RiskLimits] = None, *,
              taker_fee_bps: float = 5.0) -> BookPlan:
    """Size the book. Computes orders; cannot send them.

    Sizing order matters and is the reverse of the obvious one: each leg's
    target notional is set first, then rounded DOWN to a whole lot, and the
    residual is reported rather than absorbed. Rounding each side to hit a
    target exactly would leave the book quietly directional by whatever the lot
    sizes forced, which is the failure `plan_carry.py` was built around on two
    legs and is thirty times more likely on thirty.
    """
    config = config or BookConfig()
    limits = limits or RiskLimits()
    plan = BookPlan()
    plan.considered = len(candidates)

    kept, notes = eligible_candidates(candidates, config)
    plan.eligible = len(kept)
    plan.warnings.extend(notes)

    n_side = int(round(len(kept) * config.top_frac))
    if n_side < config.min_names_per_side:
        plan.reasons.append(
            "{} eligible instruments give {} a side at top {:.0%}, under the "
            "{} minimum. A book this narrow is a few idiosyncratic bets, not a "
            "cross-section.".format(len(kept), n_side, config.top_frac,
                                    config.min_names_per_side))
        return plan
    if 2 * n_side > len(kept):
        n_side = len(kept) // 2

    ranked = sorted(kept, key=lambda c: c.carry_bps_per_day)
    shorts, longs = ranked[:n_side], ranked[-n_side:]
    sides = (("buy", longs, Side.LONG), ("sell", shorts, Side.SHORT))

    def size_side(side: str, members: Sequence[Candidate], order_side: Side,
                  side_notional: Decimal) -> List[LegPlan]:
        weights, capped = _side_weights(members, config)
        plan.warnings.extend(capped)
        legs: List[LegPlan] = []
        for candidate in members:
            leg = LegPlan(inst_id=candidate.inst_id, side=side,
                          carry_bps_per_day=candidate.carry_bps_per_day,
                          spread_bps=candidate.spread_bps,
                          contract_value=candidate.contract_value)
            # Cross the spread: buy the ask, sell the bid. Pricing a plan at
            # the mid understates the round trip by half a spread per leg, and
            # on this venue a median spread of 2.86 bps makes that 2.86 bps of
            # imaginary edge per rebalance.
            leg.price = candidate.ask if side == "buy" else candidate.bid
            leg.target_notional_usd = side_notional * to_decimal(
                weights[candidate.inst_id])

            if leg.price <= 0 or candidate.contract_value <= 0:
                leg.reasons.append("no usable price or contract size")
                legs.append(leg)
                continue

            raw = leg.target_notional_usd / (leg.price * candidate.contract_value)
            leg.contracts = _round_to_lot(raw, candidate.lot_size)
            if leg.contracts < candidate.min_size or leg.contracts <= 0:
                leg.reasons.append(
                    "target ${:,.2f} is {} contracts, under the {} minimum; "
                    "the book's gross would need to be about ${:,.0f} for this "
                    "leg to exist".format(
                        leg.target_notional_usd, leg.contracts,
                        candidate.min_size,
                        float(config.gross_notional_usd)
                        * float(candidate.min_size) / max(float(raw), 1e-12)))
                legs.append(leg)
                continue

            leg.base = leg.contracts * candidate.contract_value
            leg.notional_usd = leg.base * leg.price

            liq = liquidation_price(
                entry_price=leg.price, leverage=config.leverage, side=order_side,
                maintenance_margin_rate=limits.maintenance_margin_rate,
                quantity_base=leg.base, fee_buffer_bps=limits.fee_buffer_bps)
            leg.liquidation_price = liq
            leg.liquidation_distance = liquidation_distance_pct(
                mark_price=leg.price, liq_price=liq)
            if (leg.liquidation_distance is not None
                    and leg.liquidation_distance < limits.min_liquidation_buffer_pct):
                leg.reasons.append(
                    "liquidation only {:.1%} away, needs {:.1%}".format(
                        leg.liquidation_distance, limits.min_liquidation_buffer_pct))
            if config.leverage > candidate.max_leverage:
                leg.reasons.append(
                    "leverage {} over the instrument's {} maximum".format(
                        config.leverage, candidate.max_leverage))
            leg.ok = not leg.reasons
            legs.append(leg)
        return legs

    def live_usd(legs: Sequence[LegPlan]) -> Decimal:
        return sum((leg.notional_usd for leg in legs if leg.ok), ZERO)

    # Sizing takes up to three passes, and each one exists for a different
    # failure.
    #
    # A leg that cannot meet its minimum size does not come back, so its side
    # is short by exactly that leg and the book is left directional. Legs fail
    # for reasons that correlate with the instrument - the smallest and least
    # liquid go first - so what is missing is never a random draw.
    #
    # Pass 1 finds the failures. Pass 2 re-sizes each side across its SURVIVORS
    # only, so the side can reach its target without the dead leg's share; a
    # pass that left the failed member in the weighting would hand its money to
    # nobody and the side would come up short again. Pass 3 trims both sides to
    # whatever the weaker one could actually fill, which is what lot rounding
    # leaves behind. Every step sizes DOWN, never up, so no leg is scaled back
    # into a lot boundary it has already been rounded against.
    target = config.gross_notional_usd / 2
    sized = {side: size_side(side, members, order_side, target)
             for side, members, order_side in sides}

    survivors = {side: [c for c in members
                        if any(leg.inst_id == c.inst_id and leg.ok
                               for leg in sized[side])]
                 for side, members, _ in sides}
    dropped = sum(len(members) - len(survivors[side])
                  for side, members, _ in sides)
    if dropped:
        plan.warnings.append(
            "{} leg(s) could not be sized and were removed; the remaining legs "
            "on their side were re-weighted to cover the gap.".format(dropped))
        failures = [leg for side in sized for leg in sized[side] if not leg.ok]
        plan.warnings = [w for w in plan.warnings if "capped at" not in w]
        sized = {side: size_side(side, survivors[side], order_side, target)
                 for side, _, order_side in sides}
        for side in sized:
            sized[side].extend(leg for leg in failures if leg.side == side)

    long_usd, short_usd = live_usd(sized["buy"]), live_usd(sized["sell"])
    if min(long_usd, short_usd) > 0 and long_usd != short_usd:
        achievable = min(long_usd, short_usd)
        if abs(long_usd - short_usd) / max(long_usd, short_usd) > to_decimal("0.001"):
            plan.warnings.append(
                "sides filled unevenly (long ${:,.2f} vs short ${:,.2f}); both "
                "re-sized down to ${:,.2f} a side so the book stays neutral. "
                "Gross is ${:,.2f}, not the ${:,.2f} asked for.".format(
                    long_usd, short_usd, achievable, achievable * 2,
                    config.gross_notional_usd))
            failures = [leg for side in sized for leg in sized[side] if not leg.ok]
            plan.warnings = [w for w in plan.warnings if "capped at" not in w]
            sized = {side: size_side(side, survivors[side], order_side, achievable)
                     for side, _, order_side in sides}
            for side in sized:
                sized[side].extend(leg for leg in failures if leg.side == side)

    plan.legs = sized["buy"] + sized["sell"]
    live = [leg for leg in plan.legs if leg.ok]
    plan.gross_notional_usd = sum((leg.notional_usd for leg in live), ZERO)
    plan.net_notional_usd = (
        sum((leg.notional_usd for leg in live if leg.side == "buy"), ZERO)
        - sum((leg.notional_usd for leg in live if leg.side == "sell"), ZERO))
    if config.leverage > 0:
        plan.margin_usd = plan.gross_notional_usd / config.leverage
    plan.leverage = config.leverage

    # Economics, from the book that can actually be put on rather than the one
    # that was asked for.
    long_carry = [leg.carry_bps_per_day for leg in live if leg.side == "buy"]
    short_carry = [leg.carry_bps_per_day for leg in live if leg.side == "sell"]
    if long_carry and short_carry:
        plan.carry_spread_bps_per_day = (
            sum(long_carry) / len(long_carry) - sum(short_carry) / len(short_carry))
        # Per unit of GROSS notional, matching how the research reports it: the
        # spread accrues on one dollar a side against two dollars gross.
        plan.expected_funding_bps = (
            plan.carry_spread_bps_per_day * config.hold_days / 2.0)
    if live:
        mean_half_spread = sum(leg.spread_bps for leg in live) / len(live) / 2.0
        # A full rebalance closes the old book and opens the new one, so every
        # leg is crossed twice per period in the worst case. Turnover measured
        # at this specification was 1.7 of a possible 4 units of notional on a
        # 2-gross book, so 2x per unit of gross is the honest bound and is what
        # the backtest charged.
        plan.round_trip_bps = 2.0 * (taker_fee_bps + mean_half_spread)
    plan.expected_net_bps = plan.expected_funding_bps - plan.round_trip_bps
    plan.expected_net_usd = (to_decimal(plan.expected_net_bps / 10_000.0)
                             * plan.gross_notional_usd)
    if plan.carry_spread_bps_per_day > 0:
        plan.breakeven_days = (plan.round_trip_bps * 2.0
                               / plan.carry_spread_bps_per_day)

    failed = [leg for leg in plan.legs if not leg.ok]
    if failed:
        plan.warnings.append(
            "{} of {} legs could not be sized; the book below is what is left, "
            "and it is no longer balanced by count".format(
                len(failed), len(plan.legs)))

    n_long = len([leg for leg in live if leg.side == "buy"])
    n_short = len([leg for leg in live if leg.side == "sell"])
    if min(n_long, n_short) < config.min_names_per_side:
        plan.reasons.append(
            "only {} long and {} short legs survived sizing, under the {} "
            "minimum a side".format(n_long, n_short, config.min_names_per_side))

    if plan.gross_notional_usd > 0:
        drift = abs(plan.net_notional_usd) / plan.gross_notional_usd
        if drift > to_decimal(config.delta_tolerance_frac):
            plan.reasons.append(
                "net exposure ${:,.2f} is {:.2%} of gross, over the {:.2%} "
                "tolerance. Lot rounding left the book directional, and a "
                "dollar-neutral strategy carrying {:.2%} of beta is a "
                "different strategy.".format(
                    plan.net_notional_usd, drift, config.delta_tolerance_frac,
                    drift))
    # Both the requested size and the realised one, because they differ and
    # each can be the one over the limit. When every leg fails to size, the
    # realised gross is zero and a check on it alone goes silent about a book
    # that asked for twenty times the notional cap.
    for label, amount in (("requested", config.gross_notional_usd),
                          ("gross", plan.gross_notional_usd)):
        if amount > limits.max_notional:
            plan.reasons.append(
                "{} ${:,.2f} over the ${:,.2f} notional limit".format(
                    label, amount, limits.max_notional))
            break
    if config.leverage > limits.max_leverage:
        plan.reasons.append("leverage {} over the {} limit".format(
            config.leverage, limits.max_leverage))
    if plan.expected_net_bps <= 0:
        plan.reasons.append(
            "expected funding {:+.1f} bps over {} days does not cover the "
            "{:.1f} bps round trip".format(
                plan.expected_funding_bps, config.hold_days, plan.round_trip_bps))

    plan.warnings.append(
        "This book has PRICE exposure: the legs are different coins and "
        "nothing cancels. Measured weekly standard deviation is about 240 bps "
        "of gross, and the worst week in five years was -1,991 bps. At {}x "
        "that is {:.0%} of margin.".format(
            config.leverage, 1991.0 / 10_000.0 * float(config.leverage)))
    plan.warnings.append(
        "Leg ordering is not decided here. A {}-leg book is directional between "
        "the first fill and the last, and that belongs to an executor.".format(
            len(live)))

    plan.ok = not plan.reasons and bool(live)
    return plan
