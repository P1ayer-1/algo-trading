"""Range fade: buy near the bottom of a recent range, sell near the top, stop
beyond both edges for when it breaks.

    levels.py   the range and the three prices a fade needs - the one
                definition the backtest and any planner share

There is deliberately no plan.py, execute.py or monitor.py. The
plan/execute/monitor split exists so that code which can move money is asked
for after the evidence, and `analysis/range_backtest.py` returned it on
2026-09-11: 365 days, ten majors, 24 configurations, and the best in-sample
lost -9.0 bps per trade out of sample [-21.4, +2.9], 8.2 bps worse than a
shuffled-day random walk with the same days. At 5x every symbol lost 53-97%
of its capital. See README step 9k before building the other verbs.
"""

from .levels import (
    Bracket,
    Range,
    RangeParams,
    brackets,
    find_range,
    range_problems,
)

__all__ = [
    "Bracket",
    "Range",
    "RangeParams",
    "brackets",
    "find_range",
    "range_problems",
]
