"""Lead quote: maker orders on BloFin timed by Binance's public feed (README 9ae).

    plan     `plan.plan_quote`      tick and latency gates, every refusal listed
    execute  `execute.LeadQuoteRunner`  production feeds, paper fills, demo orders only with confirm
    monitor  `monitor.summarise`    the log read back, read only

The state machine itself is `quoter.Quoter` and has no I/O.
"""

from .monitor import MonitorReport, summarise
from .plan import QuotePlan, plan_quote
from .quoter import Fill, Intent, QuoteConfig, Quoter

__all__ = ["Fill", "Intent", "MonitorReport", "QuoteConfig", "QuotePlan", "Quoter",
           "plan_quote", "summarise"]
