"""A trading range, and the three prices a fade of it needs. Pure arithmetic.

    LONG  near the bottom of the range, stop below it, target the middle
    SHORT near the top of the range,    stop above it, target the middle

The bet is that price keeps bouncing between its recent high and low. The stop
is the admission that sometimes it does not: a range that breaks is the start
of a move, and a fade is on the wrong side of it.

This module is the definition and nothing else. The backtest
(`analysis/range_backtest.py`) and anything that plans a live order both call
it, so the rule that was tested is the rule that would be traded. The
backtest's vectorised rolling range is asserted equal to `find_range` in the
tests, because two implementations of "the range" that drift apart produce a
backtest of a strategy nobody runs.

What "the range" means here
---------------------------
The highest high and lowest low of the last `lookback_minutes` of CLOSED bars.
The bar still forming is excluded: its extreme is not known until it closes,
and including it would let the range move out to meet the price testing it.

`trend_ratio` is the window's net move over its width,
|last close - first close| / (high - low). Near 1 the window went from one
edge to the other, which is a trend that happens to have a high and a low.
Near 0 it went nowhere. It is the whole of the "is this sideways?" judgement,
computed from the window alone, so it is available at the moment of entry
rather than in hindsight. What it cannot tell apart: a range, and a V - a
sharp move out and all the way back also nets to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ...risk import Side

BPS = 10_000.0


@dataclass(frozen=True)
class RangeParams:
    """One configuration of the fade. Fractions are of the range's WIDTH."""

    lookback_minutes: int = 1440
    # Rest the entry this far INSIDE the edge. 0 is the edge itself, which an
    # equal low touches and a real breakdown trades through.
    entry_frac: float = 0.10
    # Stop this far OUTSIDE the edge.
    stop_frac: float = 0.25
    # Take profit this far from the entry EDGE. 0.5 is the middle.
    target_frac: float = 0.50
    # Time stop. None means one lookback: a fade that has not worked within
    # the window that defined it is holding a range that no longer exists.
    hold_minutes: Optional[int] = None
    # A range narrower than this cannot pay for its round trip.
    min_width_bps: float = 50.0
    # None disables the regime filter.
    max_trend_ratio: Optional[float] = None

    @property
    def max_hold_minutes(self) -> int:
        if self.hold_minutes is not None:
            return self.hold_minutes
        return self.lookback_minutes

    def problems(self) -> List[str]:
        """Every reason this configuration is not a coherent strategy."""
        reasons: List[str] = []
        if self.lookback_minutes < 2:
            reasons.append("lookback must cover at least 2 bars")
        if not 0 <= self.entry_frac < 0.5:
            reasons.append(
                f"entry_frac {self.entry_frac:g} must be in [0, 0.5) - past "
                "the middle the long and short entries cross")
        if self.target_frac <= self.entry_frac:
            reasons.append(
                f"target_frac {self.target_frac:g} must be beyond entry_frac "
                f"{self.entry_frac:g}, or the target sits at or behind the entry")
        if self.target_frac > 1:
            reasons.append(
                f"target_frac {self.target_frac:g} is past the opposite edge")
        if self.stop_frac <= 0:
            reasons.append(
                f"stop_frac {self.stop_frac:g} must be positive - a stop at or "
                "inside the edge is hit by the range doing what ranges do")
        if self.max_hold_minutes < 1:
            reasons.append("hold must be at least one bar")
        if self.min_width_bps < 0:
            reasons.append("min_width_bps cannot be negative")
        if self.max_trend_ratio is not None and self.max_trend_ratio <= 0:
            reasons.append(
                "max_trend_ratio must be positive, or None to disable the filter")
        return reasons

    def label(self) -> str:
        trend = "any" if self.max_trend_ratio is None else f"<={self.max_trend_ratio:g}"
        return (f"{self.lookback_minutes / 60:g}h e{self.entry_frac:g} "
                f"s{self.stop_frac:g} t{self.target_frac:g} trend{trend}")


@dataclass(frozen=True)
class Range:
    high: float
    low: float
    first_close: float
    last_close: float

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def width_bps(self) -> float:
        return self.width / self.mid * BPS if self.mid > 0 else 0.0

    @property
    def trend_ratio(self) -> float:
        if self.width <= 0:
            return float("inf")
        return abs(self.last_close - self.first_close) / self.width


def find_range(highs: Sequence[float], lows: Sequence[float],
               closes: Sequence[float]) -> Optional[Range]:
    """The range over exactly the bars given, oldest first.

    The caller chooses the bars: the last `lookback_minutes` that have CLOSED,
    never the one still forming.
    """
    if not highs or len(highs) != len(lows) or len(highs) != len(closes):
        return None
    return Range(high=max(highs), low=min(lows),
                 first_close=closes[0], last_close=closes[-1])


@dataclass(frozen=True)
class Bracket:
    side: Side
    entry: float
    stop: float
    target: float

    @property
    def stop_bps(self) -> float:
        return abs(self.entry - self.stop) / self.entry * BPS

    @property
    def target_bps(self) -> float:
        return abs(self.target - self.entry) / self.entry * BPS


def brackets(rng: Range, params: RangeParams) -> Tuple[Bracket, Bracket]:
    """(long, short). Both are always returned; `range_problems` says whether
    either should be placed."""
    width = rng.width
    long = Bracket(Side.LONG,
                   entry=rng.low + params.entry_frac * width,
                   stop=rng.low - params.stop_frac * width,
                   target=rng.low + params.target_frac * width)
    short = Bracket(Side.SHORT,
                    entry=rng.high - params.entry_frac * width,
                    stop=rng.high + params.stop_frac * width,
                    target=rng.high - params.target_frac * width)
    return long, short


def range_problems(rng: Range, params: RangeParams) -> List[str]:
    """Every reason this range should not be faded."""
    if rng.width <= 0:
        return ["the range has no width - every bar printed the same price"]
    reasons: List[str] = []
    if rng.width_bps < params.min_width_bps:
        reasons.append(
            f"range is {rng.width_bps:.1f} bps wide, needs "
            f"{params.min_width_bps:g} - too narrow to pay for the round trip")
    if params.max_trend_ratio is not None and rng.trend_ratio > params.max_trend_ratio:
        reasons.append(
            f"trend ratio {rng.trend_ratio:.2f} exceeds "
            f"{params.max_trend_ratio:g} - the window travelled edge to edge, "
            "which is a trend, not a range")
    return reasons
