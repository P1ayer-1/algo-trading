"""Cross-sectional funding carry: LONG the cheap perps, SHORT the dear ones.

    plan.py     sizes the whole book, prices the risk, says whether to do it
    execute.py  moves the exchange from whatever it holds to that book
    broker.py   the REST calls execute.py is written against

`plan.py` still imports no broker and has no path to `placeOrder`, which is
checkable with a grep and is checked that way - the order path lives in
`execute.py` and nowhere else, so every sizing and margin question is settled
while the answer is still text.

There is one verb rather than an open and a close, and that follows from the
strategy: this book is REBALANCED weekly, so "open" is the case where the
exchange holds nothing and "close" is the case where the target is empty.
`reconcile` covers all three and is idempotent, because twelve legs is twelve
chances to be interrupted and the answer to "what if it dies halfway" has to be
"run it again".

There is no `monitor.py` yet. Nothing has been traded, so there is nothing to
read back.

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

from .execute import BookExecutor, ExecutionResult, Order
from .plan import BookConfig, BookPlan, Candidate, LegPlan, plan_book

__all__ = ["BookConfig", "BookExecutor", "BookPlan", "Candidate",
           "ExecutionResult", "LegPlan", "Order", "plan_book"]
