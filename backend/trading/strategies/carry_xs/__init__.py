"""Cross-sectional funding carry: LONG the cheap perps, SHORT the dear ones.

    plan.py   sizes the whole book, prices the risk, says whether to do it

There is no `execute.py` and no `monitor.py` here, and that absence is
deliberate rather than unfinished. The repo's rule is that order-sending code
is asked for after the evidence, by name; the evidence for this book exists
(step 9q) and the order path does not, so nothing in this package can place a
trade. `plan.py` imports no broker.

What this is, against the other carry
-------------------------------------
`strategies/carry` is long spot and short the perp on ONE instrument, and has
no price exposure at all. It can only harvest POSITIVE funding, because there
is no borrow and the spot leg can only be long, and it pays a spot spread of
4-50 bps to do it.

This one is perps on both sides across MANY instruments. It harvests funding of
either sign, never touches spot, and needs margin rather than notional. The
price of that is real price exposure - the legs are different coins, so nothing
cancels - and the numbers are in `plan.py`'s docstring rather than left to be
discovered: about 240 bps of weekly standard deviation on gross notional, and a
worst measured week of -1,991.

Delta-neutral is not risk-neutral here either, for a reason the two-leg carry
does not have: thirty legs margin separately, and the book is directional
between the first fill and the last.
"""

from .plan import BookConfig, BookPlan, Candidate, LegPlan, plan_book

__all__ = ["BookConfig", "BookPlan", "Candidate", "LegPlan", "plan_book"]
