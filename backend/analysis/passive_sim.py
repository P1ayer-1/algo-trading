"""Does providing liquidity actually pay? Markout curves and bracketed fill rates.

    python backend\\analysis\\passive_sim.py --date 2026-09-01 --hours 2
    python backend\\analysis\\passive_sim.py --source raw --date 2026-09-07
    python backend\\analysis\\passive_sim.py --date 2026-09-01 --hours 6 \\
        --signals obi_1,tfi_5s,ret_5s

Roadmap step 9. Every result this project has produced died on execution cost
rather than on prediction, and the one effect that survived — short-horizon
cross-sectional reversal — is the return to *providing* liquidity to whatever
just moved. Both point at the same unanswered question: what does a passive
quote actually earn, after the people who fill it are done selecting against
you? That question is worth more right now than any further feature work, and
it is answerable offline from data already on disk.

The simulator places a hypothetical passive quote on both sides of the touch
every `--quote-interval` milliseconds, leaves it there until it fills or
`--timeout` elapses, and measures two things:

  **Fill rate** — what fraction of quotes ever trade.
  **Markout** — where the mid is 0s / 1s / 5s / ... after the fill, signed so
  that positive is money. At 0s this is the half-spread you captured. The rate
  at which it decays is adverse selection, priced in basis points.

Unconditional first, because a conditional number you cannot compare to
anything is not a number. The signal-conditioned section comes second and is
evaluated on a purged out-of-sample split, because choosing the best signal
bucket on the same rows you then report is how a backtest lies.

The hard part: queue position
-----------------------------
You cannot see how many orders sit ahead of yours at a price level, and you
cannot see how many of them cancel. Neither is in any market data feed. So
this does not try to model it — it **brackets** it, running the identical
markout machinery under two fill rules:

  **Pessimistic.** You join behind the entire visible size `Q` at that level
  and fill only once cumulative same-side aggressor volume at that price
  exceeds `Q`. Nobody ahead of you ever cancels.

  **Optimistic.** You fill the moment any trade occurs at your price — as if
  you were at the front of the queue.

Both are wrong in a known direction, and the truth is between them. It sits
closer to the optimistic end than the pessimistic bound suggests, because real
queues shrink by cancellation as well as by trading, and the pessimistic rule
counts only trades. So the report also measures, directly from the L2 tape,
what fraction of observed depletion at the touch was cancellation rather than
trading (see `touch_cancel_share`) — one number that says *how far* toward the
optimistic end to look, without pretending to simulate a queue.

**Plan with the pessimistic number.** Treat the gap between the two as the
uncertainty on every figure in the report, because that is what it is.

All three fill rules are the same function with a different queue to clear:
optimistic is the pessimistic rule with `Q = 0`. See `resolve_fill`.

What this does not answer
-------------------------
Your own order changes the book, and none of this models that: it assumes the
quote is small enough to be a price taker in queue terms and large enough to
matter to you. It also assumes the quote rests at a fixed price rather than
being requoted as the touch moves, which is the conservative choice — a
managed quote fills more often and is selected against harder.

And the data is Binance (via Tardis) or whatever the raw archive holds, not
necessarily BloFin. Fill rates are a property of a specific venue's queue.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.stats import effective_sample_size  # noqa: E402
from trading.features import FeatureEngine  # noqa: E402
from trading.orderbook import OrderBook  # noqa: E402
from trading.tape import TradeTape  # noqa: E402

try:  # Single source of truth for fees; see backend/config.py.
    from config import COST_MAKER_MAKER_BPS, MAKER_FEE_BPS

    MAKER_BPS = float(MAKER_FEE_BPS)
    ROUND_TRIP_MAKER_BPS = float(COST_MAKER_MAKER_BPS)
except Exception:  # pragma: no cover - keeps the analysis tools standalone
    MAKER_BPS = 0.6
    ROUND_TRIP_MAKER_BPS = 1.2

SIDES = ("bid", "ask")

# The three fill rules, in the order they are reported. "cancel-adj" is the
# pessimistic rule with the queue shrunk by the measured cancellation share —
# an interpolation between the bounds, not a third model.
MODELS = ("optimistic", "cancel-adj", "pessimistic")

# Book and trade prices come from separate files and are compared for
# equality. Both parse from the same decimal strings so exact comparison would
# work, but a relative epsilon costs nothing and survives a venue that rounds
# differently in the two feeds.
PRICE_EPS = 1e-9

# Quotes are not placed until the feature engine has this much history behind
# it, matching check_features.FEATURE_WARMUP_SECONDS. Before it, the slow
# features report padded zeros, and bucketing quotes on padding is bucketing
# on nothing.
WARMUP_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Market state, flattened
# ---------------------------------------------------------------------------


@dataclass
class Market:
    """The whole session as flat arrays, which is what makes this tractable.

    One row per book update and one per trade. Every question the simulator
    asks — "what traded between t0 and t1 at or through price P", "where was
    the mid 30 seconds after that" — becomes a `searchsorted` plus a slice,
    instead of a walk over millions of events per quote.
    """

    book_ts: np.ndarray       # int64 ms, non-decreasing
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray

    trade_ts: np.ndarray      # int64 ms, non-decreasing
    trade_px: np.ndarray
    trade_sz: np.ndarray
    trade_is_buy: np.ndarray  # True if the AGGRESSOR was a buyer

    def __post_init__(self) -> None:
        self.mid = (self.bid_px + self.ask_px) / 2.0

    def mid_at(self, ts: np.ndarray) -> np.ndarray:
        """Last observed mid at or before each timestamp.

        Held flat rather than interpolated: interpolation would invent a price
        between two book updates, and a markout is meant to be a price you
        could actually have transacted at.
        """
        index = np.searchsorted(self.book_ts, ts, side="right") - 1
        return self.mid[np.clip(index, 0, len(self.mid) - 1)]

    @property
    def span_seconds(self) -> float:
        if len(self.book_ts) < 2:
            return 0.0
        return float(self.book_ts[-1] - self.book_ts[0]) / 1000.0


@dataclass
class Quotes:
    """One hypothetical passive quote pair per decision time.

    `queue[side]` is the visible size resting at that price the instant the
    quote is placed — all of it, since a new order joins the back.
    """

    ts: np.ndarray
    price: Dict[str, np.ndarray]
    queue: Dict[str, np.ndarray]
    spread_bps: np.ndarray
    signals: Dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.ts)


# ---------------------------------------------------------------------------
# Pass one: replay events into arrays
# ---------------------------------------------------------------------------


def collect(
    events: Iterable[dict],
    *,
    quote_interval_ms: int,
    signals: Sequence[str],
    progress_every: int = 1_000_000,
) -> Tuple[Market, Quotes]:
    """Replay BloFin-shaped messages, recording the touch, the tape and quotes.

    The book, tape and feature engine are the live ones, so a signal value
    recorded here is the same number the running bot would have computed at
    that instant. `compute()` is called only at quote times rather than on
    every event — the recorder needs a row per sample, this needs a feature
    vector per decision, and the engine's incremental state (the OFI
    recursion, the mid history) is maintained by `on_book_event` either way.
    """
    book, tape, engine = OrderBook(), TradeTape(), FeatureEngine()

    book_ts: List[int] = []
    bid_px: List[float] = []
    bid_sz: List[float] = []
    ask_px: List[float] = []
    ask_sz: List[float] = []

    trade_ts: List[int] = []
    trade_px: List[float] = []
    trade_sz: List[float] = []
    trade_is_buy: List[bool] = []

    quote_ts: List[int] = []
    quote_price: Dict[str, List[float]] = {side: [] for side in SIDES}
    quote_queue: Dict[str, List[float]] = {side: [] for side in SIDES}
    quote_spread: List[float] = []
    quote_signals: Dict[str, List[float]] = {name: [] for name in signals}

    next_quote_at = 0
    processed = 0
    skipped_invalid = 0

    for message in events:
        if not isinstance(message, dict):
            continue
        channel = message.get("arg", {}).get("channel", "")
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"  {processed:,} events  ({len(quote_ts):,} quote pairs)")

        if channel in ("books", "books5"):
            book.apply(message)
            if not book.is_ready or book.is_crossed():
                continue
            engine.on_book_event(book)
            bid, bid_size, ask, ask_size = book.best_bid_ask_size()
            ts = book.ts or (trade_ts[-1] if trade_ts else 0)
            if ts <= 0:
                continue
            book_ts.append(ts)
            bid_px.append(bid)
            bid_sz.append(bid_size)
            ask_px.append(ask)
            ask_sz.append(ask_size)
        elif channel == "trades":
            data = message.get("data")
            if not tape.add_message(data):
                continue
            # Mirror TradeTape's own filtering exactly. If these two disagree
            # about which prints are real, the fill simulation and the signals
            # conditioning it are computed from different tapes.
            for item in (data if isinstance(data, list) else [data]):
                try:
                    price = float(item["price"])
                    size = float(item["size"])
                    stamp = int(item["ts"])
                except (KeyError, TypeError, ValueError):
                    continue
                if price <= 0 or size <= 0:
                    continue
                trade_ts.append(stamp)
                trade_px.append(price)
                trade_sz.append(size)
                trade_is_buy.append(str(item.get("side", "")).lower() == "buy")
            if not trade_ts:
                continue
            ts = trade_ts[-1]
        elif channel == "funding-rate":
            engine.on_funding(message.get("data"))
            continue
        else:
            continue

        # A quote can only be placed against a book we believe in. Everything
        # else — the pending decision time included — is deferred rather than
        # faked, so a desync produces missing quotes, never wrong ones.
        if ts < next_quote_at:
            continue
        if not book.is_ready or book.is_crossed():
            continue
        snapshot = engine.compute(book, tape)
        if not snapshot.is_valid or snapshot.mid is None:
            skipped_invalid += 1
            continue
        if snapshot.history_seconds < WARMUP_SECONDS:
            next_quote_at = ts + quote_interval_ms
            continue

        bid, bid_size, ask, ask_size = book.best_bid_ask_size()
        quote_ts.append(ts)
        quote_price["bid"].append(bid)
        quote_queue["bid"].append(bid_size)
        quote_price["ask"].append(ask)
        quote_queue["ask"].append(ask_size)
        quote_spread.append(snapshot.spread_bps)
        for name in signals:
            quote_signals[name].append(float(getattr(snapshot, name, np.nan)))
        next_quote_at = ts + quote_interval_ms

    if len(book_ts) < 2:
        raise SystemExit("No usable book updates — nothing to simulate.")
    if not trade_ts:
        raise SystemExit("No trades in this window; fills cannot be resolved.")
    if not quote_ts:
        raise SystemExit(
            "No quotes were placed. The book never stayed valid for "
            f"{WARMUP_SECONDS:g}s of feature history — check the feed."
        )
    if skipped_invalid:
        print(f"  skipped {skipped_invalid:,} decision times on an invalid book")

    books = _sorted_by_time(
        "book", np.asarray(book_ts, dtype=np.int64),
        [np.asarray(column, dtype=float)
         for column in (bid_px, bid_sz, ask_px, ask_sz)])
    tape_arrays = _sorted_by_time(
        "trade", np.asarray(trade_ts, dtype=np.int64),
        [np.asarray(trade_px, dtype=float), np.asarray(trade_sz, dtype=float),
         np.asarray(trade_is_buy, dtype=bool)])

    market = Market(
        book_ts=books[0], bid_px=books[1], bid_sz=books[2],
        ask_px=books[3], ask_sz=books[4],
        trade_ts=tape_arrays[0], trade_px=tape_arrays[1],
        trade_sz=tape_arrays[2], trade_is_buy=tape_arrays[3],
    )
    quotes = Quotes(
        ts=np.asarray(quote_ts, dtype=np.int64),
        price={side: np.asarray(quote_price[side], dtype=float) for side in SIDES},
        queue={side: np.asarray(quote_queue[side], dtype=float) for side in SIDES},
        spread_bps=np.asarray(quote_spread, dtype=float),
        signals={name: np.asarray(values, dtype=float)
                 for name, values in quote_signals.items()},
    )
    return market, quotes


def _sorted_by_time(what: str, timestamps: np.ndarray,
                    columns: List[np.ndarray]) -> List[np.ndarray]:
    """Restore time order, loudly, if the source did not already have it.

    Everything downstream is `searchsorted` against these timestamps, and
    `searchsorted` on an unsorted array does not fail — it returns a wrong
    index and the whole report comes out plausible and incorrect. The raw
    archive merges by *arrival* time, so an exchange timestamp arriving late
    can put the two out of step.

    Sorting is the right repair rather than an error: the book was already
    applied in arrival order, and these arrays are only ever used to answer
    "what was true at time T". A stable sort keeps same-millisecond events in
    the order they were seen.
    """
    if len(timestamps) < 2 or bool((np.diff(timestamps) >= 0).all()):
        return [timestamps, *columns]
    print(f"  WARNING: {what} timestamps were not in order; sorting "
          f"{len(timestamps):,} of them by exchange time.")
    order = np.argsort(timestamps, kind="stable")
    return [timestamps[order], *(column[order] for column in columns)]


def observable(quotes: Quotes, market: Market, *, timeout_ms: int,
               max_horizon_ms: int) -> Quotes:
    """Drop quotes whose fill and full markout curve run off the end of the data.

    Every horizon is then measured on the *same* quotes, so the markout curve
    is a curve rather than a row of unrelated means computed on shrinking
    samples. Same lookahead guard the recorder applies to labels, for the same
    reason.
    """
    deadline = market.book_ts[-1] - timeout_ms - max_horizon_ms
    keep = quotes.ts <= deadline
    dropped = int((~keep).sum())
    if dropped:
        print(f"  dropped {dropped:,} quote pairs with no observable future")
    if keep.sum() < 30:
        raise SystemExit(
            f"Only {int(keep.sum())} quotes have an observable future. Use a "
            "longer sample (--hours), a shorter --timeout, or shorter "
            "--horizons."
        )
    return _subset(quotes, keep)


# ---------------------------------------------------------------------------
# The fill rule
# ---------------------------------------------------------------------------


def resolve_fill(
    trade_ts: np.ndarray,
    trade_px: np.ndarray,
    trade_sz: np.ndarray,
    trade_is_buy: np.ndarray,
    *,
    side: str,
    price: float,
    queue: float,
    start_ts: int,
    deadline_ts: int,
) -> Optional[int]:
    """Index of the trade that fills a resting order, or None if it never does.

    One function for every queue assumption in this module, because they
    differ only in how much volume has to print before the fill reaches you:

        optimistic     queue = 0            any trade at your price fills you
        pessimistic    queue = Q            the whole visible level goes first
        cancel-adj     queue = Q x (1 - c)  c measured, see touch_cancel_share

    A resting **bid** is filled by *sell* aggressors, and only by those trading
    at or below its price — a trade strictly below means the level was swept
    and everything resting there, you included, is gone. Getting this aggressor
    convention backwards inverts every number downstream, so it is pinned by
    tests.

    Trades stamped exactly `start_ts` are excluded: the quote is placed after
    everything already carrying that millisecond has been applied.
    """
    low = int(np.searchsorted(trade_ts, start_ts, side="right"))
    high = int(np.searchsorted(trade_ts, deadline_ts, side="right"))
    if low >= high:
        return None

    prices = trade_px[low:high]
    tolerance = abs(price) * PRICE_EPS
    if side == "bid":
        qualifies = (~trade_is_buy[low:high]) & (prices <= price + tolerance)
    else:
        qualifies = trade_is_buy[low:high] & (prices >= price - tolerance)
    if not qualifies.any():
        return None

    # Cumulative qualifying volume only rises on a qualifying trade, so the
    # first crossing is necessarily one of them.
    cumulative = np.cumsum(np.where(qualifies, trade_sz[low:high], 0.0))
    filled = cumulative > queue
    if not filled.any():
        return None
    return low + int(np.argmax(filled))


def simulate(
    market: Market,
    quotes: Quotes,
    *,
    timeout_ms: int,
    cancel_share: Dict[str, float],
) -> Dict[Tuple[str, str], np.ndarray]:
    """Fill index per (side, model), with -1 for a quote that never filled."""
    results: Dict[Tuple[str, str], np.ndarray] = {}
    for side in SIDES:
        queues = {
            "optimistic": np.zeros(len(quotes)),
            "cancel-adj": quotes.queue[side] * (1.0 - cancel_share[side]),
            "pessimistic": quotes.queue[side],
        }
        for model in MODELS:
            indices = np.full(len(quotes), -1, dtype=np.int64)
            queue_sizes = queues[model]
            for position in range(len(quotes)):
                start = int(quotes.ts[position])
                hit = resolve_fill(
                    market.trade_ts, market.trade_px, market.trade_sz,
                    market.trade_is_buy,
                    side=side,
                    price=float(quotes.price[side][position]),
                    queue=float(queue_sizes[position]),
                    start_ts=start,
                    deadline_ts=start + timeout_ms,
                )
                if hit is not None:
                    indices[position] = hit
            results[(side, model)] = indices
    return results


# ---------------------------------------------------------------------------
# Where in the bracket the truth sits
# ---------------------------------------------------------------------------


def touch_cancel_share(market: Market, side: str) -> float:
    """Fraction of observed depletion at the touch that was NOT trading.

    The pessimistic bound assumes a queue only shrinks by trading. It also
    shrinks by cancellation, and unlike queue position that *is* observable:
    between two consecutive book updates at an unchanged touch price, the size
    fell by more than the volume that printed there. The excess was cancelled.

    Returns a fraction in [0, 1]. High means the pessimistic bound is very
    pessimistic — most of the queue ahead of you evaporates rather than
    trading, so real fills land far closer to the optimistic end.

    This is a measurement, not a model, and a rough one: intervals where
    someone *adds* size hide removals inside a smaller net change, which biases
    it down. Down is the safe direction — it makes the cancel-adjusted column
    more conservative, not less.
    """
    price = market.bid_px if side == "bid" else market.ask_px
    size = market.bid_sz if side == "bid" else market.ask_sz
    if len(price) < 2:
        return 0.0

    unchanged = price[1:] == price[:-1]
    change = size[1:] - size[:-1]
    depleted = np.where(unchanged & (change < 0), -change, 0.0)
    if depleted.sum() <= 0:
        return 0.0

    # Attribute each trade to the book interval it lands in, then keep only
    # those that could have consumed this level: right aggressor, and exactly
    # at the touch price that held across the interval.
    slot = np.searchsorted(market.book_ts, market.trade_ts, side="right") - 1
    inside = (slot >= 0) & (slot < len(depleted))
    slot = slot[inside]
    traded = np.zeros(len(depleted))
    if len(slot):
        level = price[slot]
        is_buy = market.trade_is_buy[inside]
        aggressor = ~is_buy if side == "bid" else is_buy
        at_level = np.abs(market.trade_px[inside] - level) <= np.abs(level) * PRICE_EPS
        keep = aggressor & at_level
        traded = np.bincount(slot[keep], weights=market.trade_sz[inside][keep],
                             minlength=len(depleted))[:len(depleted)]

    cancelled = np.maximum(0.0, depleted - traded)
    return float(cancelled.sum() / depleted.sum())


# ---------------------------------------------------------------------------
# Markout
# ---------------------------------------------------------------------------


def markout_bps(
    market: Market,
    fill_ts: np.ndarray,
    fill_price: np.ndarray,
    side: str,
    horizons_s: Sequence[float],
) -> np.ndarray:
    """(n_fills, n_horizons) markout in bps, signed so positive is money.

    For a filled bid: `(mid(t+h) - fill_price) / mid(t)`. At h = 0 this is the
    half-spread captured for providing the liquidity, and it is the only part
    of the curve that is yours for free. Everything after it is the market's
    opinion of what you just bought.
    """
    sign = 1.0 if side == "bid" else -1.0
    base = market.mid_at(fill_ts)
    out = np.empty((len(fill_ts), len(horizons_s)), dtype=float)
    for column, horizon in enumerate(horizons_s):
        forward = market.mid_at(fill_ts + int(horizon * 1000))
        out[:, column] = sign * (forward - fill_price) / base * 1e4
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fill_mask(indices: np.ndarray) -> np.ndarray:
    return indices >= 0


def _stderr(values: np.ndarray, effective_n: int) -> float:
    if len(values) < 2 or effective_n < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(effective_n))


def _subset(quotes: Quotes, mask: np.ndarray) -> Quotes:
    return Quotes(
        ts=quotes.ts[mask],
        price={side: quotes.price[side][mask] for side in SIDES},
        queue={side: quotes.queue[side][mask] for side in SIDES},
        spread_bps=quotes.spread_bps[mask],
        signals={name: values[mask] for name, values in quotes.signals.items()},
    )


def _subset_fills(fills: Dict[Tuple[str, str], np.ndarray],
                  mask: np.ndarray) -> Dict[Tuple[str, str], np.ndarray]:
    return {key: value[mask] for key, value in fills.items()}


def report_setup(market: Market, quotes: Quotes, *, quote_interval_ms: int,
                 timeout_ms: int, cancel_share: Dict[str, float]) -> None:
    print("\n" + "=" * 72)
    print("SETUP")
    print("=" * 72)
    print(f"  book updates          {len(market.book_ts):,}")
    print(f"  trades                {len(market.trade_ts):,}")
    print(f"  time span             {market.span_seconds / 3600:.2f} hours")
    print(f"  quote pairs           {len(quotes):,} "
          f"(one every {quote_interval_ms} ms, both sides)")
    print(f"  resting timeout       {timeout_ms / 1000:g}s")
    print(f"  median spread         {np.median(quotes.spread_bps):.3f} bps")
    for side in SIDES:
        print(f"  median {side} queue Q    "
              f"{np.median(quotes.queue[side]):.3f} contracts")

    print("\n  QUEUE DEPLETION AT THE TOUCH — where in the bracket to look")
    for side in SIDES:
        print(f"    {side}: {cancel_share[side]:.1%} of observed depletion was "
              "cancellation, not trading")
    print("    The pessimistic rule counts only the trading part, so it "
          "understates\n    fills by roughly that much. The cancel-adj column "
          "shrinks Q by it.")


def report_fill_rates(quotes: Quotes, market: Market,
                      fills: Dict[Tuple[str, str], np.ndarray],
                      *, timeout_ms: int) -> None:
    print("\n" + "=" * 72)
    print(f"FILL RATES  (within {timeout_ms / 1000:g}s, quote resting at a "
          "fixed price)")
    print("=" * 72)
    print(f"  {'':<12}" + "".join(f"{model:>14}" for model in MODELS))
    print("  " + "-" * 54)
    for side in SIDES:
        cells = [f"{_fill_mask(fills[(side, model)]).mean():>13.1%} "
                 for model in MODELS]
        print(f"  {side:<12}" + "".join(cells))

    print(f"\n  {'median wait':<12}" + "".join(f"{model:>14}" for model in MODELS))
    print("  " + "-" * 54)
    for side in SIDES:
        cells = []
        for model in MODELS:
            indices = fills[(side, model)]
            filled = _fill_mask(indices)
            if not filled.any():
                cells.append(f"{'-':>13} ")
                continue
            delay = (market.trade_ts[indices[filled]] - quotes.ts[filled]) / 1000.0
            cells.append(f"{np.median(delay):>12.1f}s ")
        print(f"  {side:<12}" + "".join(cells))

    print("\n  A quote that never fills costs nothing and earns nothing. The "
          "gap between\n  the optimistic and pessimistic columns is the "
          "uncertainty on every number\n  below — plan with the pessimistic one.")


def markout_table(
    market: Market,
    quotes: Quotes,
    fills: Dict[Tuple[str, str], np.ndarray],
    *,
    horizons_s: Sequence[float],
    sides: Sequence[str] = SIDES,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """Per model: (mean markout by horizon, standard error, number of fills).

    Bid and ask fills are pooled. Signed as they are, they are the same trade
    seen from opposite sides, and separating them halves the sample for no gain
    unless the venue is structurally asymmetric — worth checking once, not
    reporting forever.
    """
    table: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    for model in MODELS:
        rows: List[np.ndarray] = []
        gaps: List[float] = []
        for side in sides:
            indices = fills[(side, model)]
            filled = _fill_mask(indices)
            if not filled.any():
                continue
            fill_ts = market.trade_ts[indices[filled]]
            rows.append(markout_bps(market, fill_ts, quotes.price[side][filled],
                                    side, horizons_s))
            gaps.append(market.span_seconds / max(1, int(filled.sum())))
        if not rows:
            table[model] = (np.full(len(horizons_s), np.nan),
                            np.full(len(horizons_s), np.nan), 0)
            continue
        stacked = np.vstack(rows)
        mean_gap = float(np.mean(gaps))
        errors = np.array([
            _stderr(stacked[:, column],
                    effective_sample_size(len(stacked), horizons_s[column], mean_gap))
            for column in range(len(horizons_s))
        ])
        table[model] = (stacked.mean(axis=0), errors, len(stacked))
    return table


def report_markout(table: Dict[str, Tuple[np.ndarray, np.ndarray, int]],
                   horizons_s: Sequence[float]) -> None:
    print("\n" + "=" * 72)
    print("UNCONDITIONAL MARKOUT  (bps, gross; positive is money)")
    print("=" * 72)
    print(f"  {'horizon':>9}" + "".join(f"{model:>18}" for model in MODELS))
    print("  " + "-" * 63)
    for column, horizon in enumerate(horizons_s):
        cells = []
        for model in MODELS:
            mean, error, count = table[model]
            cells.append(f"{'-':>18}" if count == 0
                         else f"{mean[column]:>+10.3f} +-{error[column]:5.3f}")
        note = "  <- half spread captured" if horizon == 0 else ""
        print(f"  {horizon:>8.0f}s" + "".join(cells) + note)

    print("\n  fills    " + "".join(f"{table[model][2]:>18,}" for model in MODELS))
    print("\n  Standard errors use the EFFECTIVE sample size: consecutive fills "
          "share\n  most of their markout window, so the raw fill count "
          "overstates independence.")


def report_economics(table: Dict[str, Tuple[np.ndarray, np.ndarray, int]],
                     fills: Dict[Tuple[str, str], np.ndarray],
                     horizons_s: Sequence[float],
                     decision_horizon: float,
                     half_spread_bps: float) -> None:
    column = int(np.argmin(np.abs(np.asarray(horizons_s) - decision_horizon)))
    horizon = horizons_s[column]

    print("\n" + "=" * 72)
    print(f"ECONOMICS AT {horizon:g}s  (the number that decides)")
    print("=" * 72)

    # Do this arithmetic before reading anything below it. Providing liquidity
    # earns the half spread and nothing else; if that is smaller than the fee,
    # a flawless fill with zero adverse selection still loses, and no queue
    # assumption, signal or horizon can change it. On a one-tick-wide
    # instrument this is usually the whole story.
    starved = half_spread_bps < ROUND_TRIP_MAKER_BPS
    print(f"  median half spread             {half_spread_bps:>7.3f} bps  "
          "<- the most a passive fill can capture")
    print(f"  round trip, both legs passive  {ROUND_TRIP_MAKER_BPS:>7.3f} bps  "
          f"({MAKER_BPS:.2f} per leg)")
    print(f"  a flawless fill, marked out instantly, earns "
          f"{half_spread_bps - ROUND_TRIP_MAKER_BPS:+.3f} bps")
    if starved:
        print("  The spread does not cover the fee. Everything below is "
              "measuring how\n  much worse than that it gets.")

    print(f"\n  {'':<16}{'gross':>10}{'net of fees':>14}{'fill rate':>12}"
          f"{'per quote':>12}")
    print("  " + "-" * 64)

    net_by_model: Dict[str, float] = {}
    for model in MODELS:
        mean, _, count = table[model]
        if count == 0:
            print(f"  {model:<16}{'no fills':>10}")
            net_by_model[model] = float("nan")
            continue
        gross = float(mean[column])
        net = gross - ROUND_TRIP_MAKER_BPS
        rate = float(np.mean([_fill_mask(fills[(side, model)]).mean()
                              for side in SIDES]))
        print(f"  {model:<16}{gross:>+10.3f}{net:>+14.3f}{rate:>12.1%}"
              f"{net * rate:>+12.3f}")
        net_by_model[model] = net

    print("\n  'per quote' is fill rate x net markout — what one quote is worth "
          "before\n  you know whether it fills. It is not an hourly rate: "
          "quotes overlap.")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    pessimistic, optimistic = net_by_model["pessimistic"], net_by_model["optimistic"]
    if np.isnan(optimistic) or np.isnan(pessimistic):
        print("  NO FILLS — nothing to conclude. Widen --timeout, or check the feed.")
    elif pessimistic > 0:
        print("  PASSIVE ENTRY PAYS, EVEN AT THE BACK OF THE QUEUE")
        print(f"  Net {pessimistic:+.3f} bps under the pessimistic rule, "
              f"{optimistic:+.3f} optimistic.")
        print("  Confirm on a different date range before believing it, then "
              "size from the\n  pessimistic number and treat the rest as "
              "headroom, not budget.")
    elif optimistic > 0:
        print("  DEPENDS ENTIRELY ON QUEUE POSITION")
        print(f"  The bracket straddles zero: {pessimistic:+.3f} bps "
              f"pessimistic, {optimistic:+.3f} optimistic.")
        print("  The measured cancellation share above says how far toward the "
              "optimistic\n  end to look. This is the case where queue position "
              "IS the strategy — a\n  passive fill that arrives late is a "
              "different trade from one that arrives\n  early.")
    elif starved:
        print("  THE SPREAD NEVER COVERED THE FEE")
        print(f"  Net {optimistic:+.3f} bps even at the FRONT of the queue, and "
              f"{half_spread_bps - ROUND_TRIP_MAKER_BPS:+.3f} of\n  that was "
              "lost before a single fill was adversely selected. This is not a "
              "queue\n  problem or a signal problem — it is arithmetic.")
        print("  A one-tick spread on a heavily arbitraged instrument cannot "
              "pay a maker\n  fee. Look at a wider-spread venue or symbol, or "
              "at a fee tier with a maker\n  rebate, before looking at "
              "anything else.")
    else:
        print("  ADVERSE SELECTION EXCEEDS THE SPREAD")
        print(f"  Net {optimistic:+.3f} bps even at the FRONT of the queue, so "
              "no queue\n  assumption rescues it. Whoever fills you knows "
              "something at this horizon.")
        print("  Quoting unconditionally does not work here. The conditional "
              "section below\n  is the only remaining question: is there a "
              "state in which it does?")


# ---------------------------------------------------------------------------
# Conditional on the signal
# ---------------------------------------------------------------------------


def bucket_edges(values: np.ndarray, buckets: int) -> np.ndarray:
    """Interior quantile edges, fitted on the training rows only.

    Duplicates are collapsed. Several of these features are zero most of the
    time — `ret_5s` on a book that did not move is exactly 0.0 — and repeated
    edges would otherwise produce an empty bucket with a `[+0.000, +0.000)`
    range, which looks like a bug and reports statistics on nothing. Fewer,
    honest buckets is the right answer; the caller reads the count back off
    the returned array.
    """
    edges = np.quantile(values, np.linspace(0.0, 1.0, buckets + 1)[1:-1])
    return np.unique(edges)


def assign_buckets(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right")


def conditional_stats(
    market: Market,
    quotes: Quotes,
    fills: Dict[Tuple[str, str], np.ndarray],
    membership: np.ndarray,
    bucket: int,
    *,
    model: str,
    horizon: float,
) -> Tuple[float, float, int]:
    """(fill rate, mean gross markout, fills) for one signal bucket."""
    selected = membership == bucket
    if not selected.any():
        return float("nan"), float("nan"), 0

    rates: List[float] = []
    marks: List[np.ndarray] = []
    for side in SIDES:
        indices = fills[(side, model)][selected]
        filled = _fill_mask(indices)
        rates.append(float(filled.mean()))
        if filled.any():
            marks.append(markout_bps(
                market, market.trade_ts[indices[filled]],
                quotes.price[side][selected][filled], side, [horizon])[:, 0])
    if not marks:
        return float(np.mean(rates)), float("nan"), 0
    pooled = np.concatenate(marks)
    return float(np.mean(rates)), float(pooled.mean()), len(pooled)


def report_conditional(
    market: Market,
    quotes: Quotes,
    fills: Dict[Tuple[str, str], np.ndarray],
    *,
    signals: Sequence[str],
    buckets: int,
    horizon: float,
    model: str,
    train_fraction: float,
) -> None:
    print("\n" + "=" * 72)
    print(f"CONDITIONAL ON THE SIGNAL  ({model} queue, {horizon:g}s markout)")
    print("=" * 72)
    print("  A signal is only useful here if it improves markout WITHOUT "
          "collapsing the\n  fill rate. Quoting into a state nobody wants to "
          "trade against is a perfect\n  markout on no volume.")

    split = max(1, int(len(quotes) * train_fraction))
    for name in signals:
        values = quotes.signals.get(name)
        if values is None or not np.isfinite(values).all() or values.std() == 0:
            print(f"\n  {name}: constant or unavailable, skipped.")
            continue

        edges = bucket_edges(values[:split], buckets)
        membership = assign_buckets(values, edges)
        actual = len(edges) + 1

        print(f"\n  {name}")
        if actual < buckets:
            print(f"    ({buckets} buckets requested, {actual} distinct — this "
                  "feature is tied\n     across most quotes)")
        print(f"    {'bucket':<8}{'range':>24}{'fill rate':>12}"
              f"{'gross bps':>12}{'net bps':>10}")
        print("    " + "-" * 64)
        bounds = np.concatenate([[-np.inf], edges, [np.inf]])
        for bucket in range(actual):
            rate, gross, count = conditional_stats(
                market, quotes, fills, membership, bucket,
                model=model, horizon=horizon)
            label = f"[{bounds[bucket]:+.3f}, {bounds[bucket + 1]:+.3f})"
            rate_cell = "-" if not np.isfinite(rate) else f"{rate:.1%}"
            if count == 0:
                print(f"    {bucket + 1:<8}{label:>24}{rate_cell:>12}"
                      f"{'-':>12}{'-':>10}")
                continue
            print(f"    {bucket + 1:<8}{label:>24}{rate_cell:>12}"
                  f"{gross:>+12.3f}{gross - ROUND_TRIP_MAKER_BPS:>+10.3f}")

        _report_out_of_sample(market, quotes, fills, membership, actual,
                              split=split, model=model, horizon=horizon)


def _report_out_of_sample(
    market: Market,
    quotes: Quotes,
    fills: Dict[Tuple[str, str], np.ndarray],
    membership: np.ndarray,
    buckets: int,
    *,
    split: int,
    model: str,
    horizon: float,
) -> None:
    """Pick the best bucket in-sample, then report what that choice earned after.

    The table above is fitted and scored on the same rows, so its best bucket
    is optimistic by construction — with five buckets and a noisy metric, one
    of them looks good whether or not anything is there. The honest version
    picks on the first `train_fraction` of the session and reports the rest.
    The gap between the two lines is the cost of choosing.
    """
    train = np.zeros(len(quotes), dtype=bool)
    train[:split] = True
    test = ~train
    if test.sum() < 30:
        print("    (too little data after the split to score the choice)")
        return

    train_quotes, train_fills = _subset(quotes, train), _subset_fills(fills, train)
    best_bucket, best_value = None, -np.inf
    for bucket in range(buckets):
        _, gross, count = conditional_stats(
            market, train_quotes, train_fills, membership[train], bucket,
            model=model, horizon=horizon)
        if count >= 30 and np.isfinite(gross) and gross > best_value:
            best_bucket, best_value = bucket, gross

    if best_bucket is None:
        print("    (too few in-sample fills in any bucket to choose one)")
        return

    test_quotes, test_fills = _subset(quotes, test), _subset_fills(fills, test)
    rate, gross, count = conditional_stats(
        market, test_quotes, test_fills, membership[test], best_bucket,
        model=model, horizon=horizon)
    base_rate, base_gross, base_count = conditional_stats(
        market, test_quotes, test_fills,
        np.zeros(int(test.sum()), dtype=int), 0, model=model, horizon=horizon)

    print(f"    picked bucket {best_bucket + 1} on the first "
          f"{split / len(quotes):.0%} ({best_value:+.3f} bps in-sample)")
    if count == 0:
        print("    out-of-sample: no fills in that bucket.")
        return
    print(f"    out-of-sample:   {gross:+.3f} gross, "
          f"{gross - ROUND_TRIP_MAKER_BPS:+.3f} net, fill rate {rate:.1%}, "
          f"{count:,} fills")
    print(f"    unconditional:   {base_gross:+.3f} gross, "
          f"{base_gross - ROUND_TRIP_MAKER_BPS:+.3f} net, fill rate "
          f"{base_rate:.1%}, {base_count:,} fills")


# ---------------------------------------------------------------------------
# Event sources
# ---------------------------------------------------------------------------


def tardis_events(cache: Path, exchange: str, symbol: str, date: str,
                  depth: int, hours: Optional[float]) -> List[dict]:
    """Merged book+trade messages from the Tardis sample files.

    Reuses the importer's parsers rather than re-reading the CSVs here, so a
    schema fix lands in one place and the simulator can never disagree with
    the feature files about what the data said.
    """
    from analysis.tardis_import import book_events, trade_events

    book_path = cache / f"{exchange}-{symbol}-book_snapshot_{depth}-{date}.csv.gz"
    trade_path = cache / f"{exchange}-{symbol}-trades-{date}.csv.gz"
    for path in (book_path, trade_path):
        if not path.exists():
            raise SystemExit(
                f"\nMissing {path}.\nDownload it first:\n"
                f"  python backend\\analysis\\tardis_import.py --date {date} "
                f"--symbol {symbol} --hours 2"
            )

    limit_ms = int(hours * 3_600_000) if hours else None
    print("Parsing book snapshots...")
    books = book_events(book_path, limit_ms, depth=depth)
    print(f"  {len(books):,} book updates")
    print("Parsing trades...")
    trades = trade_events(trade_path, limit_ms)
    print(f"  {len(trades):,} trades")

    print("Merging...")
    merged = sorted(books + trades, key=lambda item: (item[0], item[1]))
    return [message for _, _, message in merged]


def raw_archive_events(raw_dir: Path, date: Optional[str]) -> Iterable[dict]:
    """Messages from the live raw archive — the real venue, in arrival order."""
    from analysis.replay import merged_events

    return (message for _, _, message in merged_events(raw_dir, date))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=("tardis", "raw"), default="tardis",
                        help="tardis = the free sample day; raw = data/raw.")
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD. Required for tardis; optional for raw.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--depth", type=int, default=25, choices=(5, 25))
    parser.add_argument("--hours", type=float, default=None,
                        help="Only simulate the first N hours. Start with 2.")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Tardis download dir (default data/tardis/raw).")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Raw archive root (default data/raw).")
    parser.add_argument("--quote-interval", type=int, default=1000,
                        help="Milliseconds between hypothetical quote pairs.")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Seconds a quote rests before being given up on.")
    parser.add_argument("--horizons", default="0,1,5,10,30,60,300",
                        help="Markout horizons in seconds, measured from the fill.")
    parser.add_argument("--decision-horizon", type=float, default=60.0,
                        help="The horizon the verdict is taken at.")
    parser.add_argument("--signals", default="obi_1,ofi_1s,tfi_5s,ret_5s",
                        help="FeatureSnapshot fields to condition on.")
    parser.add_argument("--buckets", type=int, default=5)
    parser.add_argument("--conditional-model", choices=MODELS, default="pessimistic",
                        help="Queue rule the conditional section uses. Defaults "
                             "to the one you should plan with.")
    parser.add_argument("--train-fraction", type=float, default=0.7,
                        help="Fraction of the session used to CHOOSE a bucket.")
    args = parser.parse_args(argv)

    horizons = tuple(float(part) for part in args.horizons.split(",") if part.strip())
    signals = tuple(name.strip() for name in args.signals.split(",") if name.strip())
    timeout_ms = int(args.timeout * 1000)
    max_horizon_ms = int(max(horizons) * 1000)

    if args.source == "tardis":
        if not args.date:
            raise SystemExit("--date is required for --source tardis.")
        cache = args.cache or repo_root / "data" / "tardis" / "raw"
        print(f"Passive fill simulation - Tardis {args.exchange} {args.symbol} "
              f"{args.date}")
        print("=" * 72)
        events: Iterable[dict] = tardis_events(
            cache, args.exchange, args.symbol, args.date, args.depth, args.hours)
    else:
        raw_dir = args.raw_dir or repo_root / "data" / "raw"
        print(f"Passive fill simulation - raw archive {args.date or 'all dates'}")
        print("=" * 72)
        events = raw_archive_events(raw_dir, args.date)

    print("\nReplaying events...")
    market, quotes = collect(
        events, quote_interval_ms=args.quote_interval, signals=signals)
    quotes = observable(quotes, market, timeout_ms=timeout_ms,
                        max_horizon_ms=max_horizon_ms)

    cancel_share = {side: touch_cancel_share(market, side) for side in SIDES}
    report_setup(market, quotes, quote_interval_ms=args.quote_interval,
                 timeout_ms=timeout_ms, cancel_share=cancel_share)

    print("\nResolving fills...")
    fills = simulate(market, quotes, timeout_ms=timeout_ms,
                     cancel_share=cancel_share)

    report_fill_rates(quotes, market, fills, timeout_ms=timeout_ms)
    table = markout_table(market, quotes, fills, horizons_s=horizons)
    report_markout(table, horizons)
    report_economics(table, fills, horizons, args.decision_horizon,
                     float(np.median(quotes.spread_bps)) / 2.0)
    report_conditional(market, quotes, fills, signals=signals,
                       buckets=args.buckets, horizon=args.decision_horizon,
                       model=args.conditional_model,
                       train_fraction=args.train_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
