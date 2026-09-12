"""The same coin, two venues, two funding rates. Is the difference harvestable?

    python backend\\analysis\\funding_dispersion.py
    python backend\\analysis\\funding_dispersion.py --hold-days 14 --top 10

The trade, and why it is not the carry of step 9e
-------------------------------------------------
Step 9e's cash-and-carry buys spot and shorts the perp against it. Three things
constrain it, and all three came from the spot leg: there is no borrow, so the
spot side can only be LONG and only POSITIVE funding is harvestable; the spot
spread runs 4-50 bps against a perp spread under 2, so the four-leg round trip
is days of carry; and it ties up the full notional in spot.

Replace the spot leg with the SAME COIN'S PERP ON ANOTHER VENUE and all three
go away. Long BloFin's perp and short Binance's is delta-neutral by
construction - it is the same underlying - it harvests the funding DIFFERENCE
whichever sign either leg has, and both legs are perps, which are the cheap
side of both venues' books.

What replaces them is a new risk, and it is measured here rather than assumed:
the two perps are not the same instrument, and their prices can diverge. The
pair's P&L is `(venue A price move - venue B price move) + funding collected`,
and the first term is computed from both panels' own closes at the same UTC
instant rather than asserted to be zero.

What this measures
------------------
For every date and every coin both panels carry, the daily funding difference
in bps. Then the strategy: every `--hold-days`, take the `--top` coins by
trailing funding difference, long the cheap venue's perp and short the dear
one, hold, collect what actually settled, pay the actual divergence, and charge
four legs of taker fee.

The honest bar, stated before the numbers
-----------------------------------------
A round trip is four taker legs - in and out, on two venues - so at VIP 1
futures taker rates it is about 20 bps before spreads, and BloFin's alt spreads
are 3-10 bps (step 9c). A 30 bps all-in round trip needs the funding difference
to accumulate past 30 bps over the hold, which at a typical 8h cadence means
the difference has to be both large and PERSISTENT. Persistence is the part
this file exists to measure: a difference that reverts before the position is
on is a spread you can see and cannot collect.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.factor_panel import Panel, block_bootstrap, load_panel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BINANCE_PANEL = REPO_ROOT / "data" / "panel" / "daily.csv"
BLOFIN_PANEL = REPO_ROOT / "data" / "panel" / "blofin-daily.csv"


@dataclass
class Aligned:
    """Two panels reduced to the dates and symbols both of them carry."""

    dates: List[str]
    symbols: List[str]
    funding_a: np.ndarray          # bps accrued that day, venue A
    funding_b: np.ndarray
    close_a: np.ndarray
    close_b: np.ndarray
    volume: np.ndarray             # the SMALLER of the two venues' volumes
    name_a: str
    name_b: str

    @property
    def difference(self) -> np.ndarray:
        """`funding_a - funding_b`, bps per day.

        Positive means venue A's longs pay more, so the harvest is short A and
        long B.
        """
        return self.funding_a - self.funding_b


QUOTES = ("USDT", "USDC", "USD")
# Contract multipliers. A perp on 1000 SHIB and one on 1 SHIB are the same
# underlying and quote the same funding RATE - the multiplier changes the size
# of a contract, not the percentage paid - so matching across them is correct.
# Venues spell the multiplier differently, hence both forms.
MULTIPLIERS = ("1000000", "1000", "100", "10", "k", "K", "M")


def canonical(symbol: str) -> str:
    """`1000BONKUSDT` and `kBONK` both become `BONK`.

    The three panels name the same coin three ways: Binance and BloFin use
    `<BASE><QUOTE>`, Hyperliquid uses the bare base, and each venue picks its
    own contract multiplier. Matching on the raw string silently drops every
    multiplied contract - which is most of the meme perps, and those are where
    the funding dispersion is largest, so the loss would not be random.
    """
    name = symbol.upper()
    for quote in QUOTES:
        if name.endswith(quote) and len(name) > len(quote):
            name = name[:-len(quote)]
            break
    for multiplier in MULTIPLIERS:
        prefix = multiplier.upper()
        if name.startswith(prefix) and len(name) > len(prefix):
            name = name[len(prefix):]
            break
    return name


def align(panel_a: Panel, panel_b: Panel, *, name_a: str, name_b: str) -> Aligned:
    """Intersect two panels on date and symbol.

    Both panels close their rows at 00:00 UTC, so the two closes are quotes on
    the same underlying at the same instant and their difference is a real
    basis rather than a timing artifact. That is the single assumption the
    divergence leg rests on, and it is why both panels are built to the same
    schema instead of being compared loosely.
    """
    dates = sorted(set(panel_a.dates) & set(panel_b.dates))
    index_a = {d: i for i, d in enumerate(panel_a.dates)}
    index_b = {d: i for i, d in enumerate(panel_b.dates)}

    # Match on the canonical base asset, not the raw ticker. A base that maps
    # from two tickers on one venue is dropped rather than guessed at: it means
    # that venue lists the coin twice (a USDT and a USDC contract, say) and
    # picking one silently would be a choice nobody made.
    def columns(panel: Panel) -> Dict[str, int]:
        seen: Dict[str, int] = {}
        clashes = set()
        for index, symbol in enumerate(panel.symbols):
            base = canonical(symbol)
            if base in seen:
                clashes.add(base)
            seen[base] = index
        for base in clashes:
            seen.pop(base, None)
        return seen

    col_a, col_b = columns(panel_a), columns(panel_b)
    symbols = sorted(set(col_a) & set(col_b))

    shape = (len(dates), len(symbols))
    funding_a, funding_b = np.full(shape, np.nan), np.full(shape, np.nan)
    close_a, close_b = np.full(shape, np.nan), np.full(shape, np.nan)
    volume = np.full(shape, np.nan)

    for d, date in enumerate(dates):
        ra, rb = index_a[date], index_b[date]
        for s, symbol in enumerate(symbols):
            ca, cb = col_a[symbol], col_b[symbol]
            funding_a[d, s] = panel_a.funding[ra, ca]
            funding_b[d, s] = panel_b.funding[rb, cb]
            close_a[d, s] = panel_a.close[ra, ca]
            close_b[d, s] = panel_b.close[rb, cb]
            volume[d, s] = min(panel_a.volume[ra, ca], panel_b.volume[rb, cb])

    return Aligned(dates, symbols, funding_a, funding_b, close_a, close_b,
                   volume, name_a, name_b)


def describe(data: Aligned) -> None:
    """How big is the difference, and does it persist?

    Size without persistence is not an opportunity. A funding difference that
    is large today and gone tomorrow is a spread you can see and cannot
    collect, because the position takes a round trip to put on and the round
    trip is paid whether or not the difference survives it.
    """
    difference = data.difference
    finite = difference[np.isfinite(difference)]
    print()
    print("Funding difference, " + data.name_a + " minus " + data.name_b
          + ", bps accrued per day")
    print("  observations      {:,} coin-days over {} coins, {} .. {}".format(
        len(finite), len(data.symbols), data.dates[0], data.dates[-1]))
    if not len(finite):
        return
    for label, value in (("mean", finite.mean()), ("median", np.median(finite)),
                         ("mean |diff|", np.abs(finite).mean()),
                         ("p90 |diff|", np.percentile(np.abs(finite), 90)),
                         ("p99 |diff|", np.percentile(np.abs(finite), 99))):
        print("  {:<17} {:+.3f}".format(label, value))

    print()
    print("  Persistence: correlation of a coin's difference with its own past")
    for lag in (1, 3, 7, 14, 30):
        pairs = []
        for s in range(len(data.symbols)):
            column = difference[:, s]
            usable = np.isfinite(column[lag:]) & np.isfinite(column[:-lag])
            if usable.sum() > 30:
                pairs.append(np.corrcoef(column[lag:][usable],
                                         column[:-lag][usable])[0, 1])
        if pairs:
            print("    lag {:>2}d   mean autocorrelation {:+.3f}  "
                  "(over {} coins)".format(lag, float(np.mean(pairs)), len(pairs)))


@dataclass
class PairResult:
    net: np.ndarray
    funding: np.ndarray
    divergence: np.ndarray
    dates: List[str]
    positions: int


def backtest(data: Aligned, *, hold_days: int, top: int, cost_bps: float,
             min_volume: float, start: int, lookback: int,
             shuffle_seed: Optional[int] = None) -> PairResult:
    """Hold the `top` widest trailing differences for `hold_days`, repeatedly.

    Each position is one dollar long the cheap venue's perp and one dollar
    short the dear one, so the book is `top` dollars a side. Returns are per
    unit of gross notional, matching `factor_panel.py`, and cost is four taker
    legs per full round trip charged as `2 * cost_bps` on entry and the same on
    exit - the pair is opened and closed on both venues.

    `shuffle_seed` picks the coins at random instead, keeping the position
    count, the hold and the cost: the control for "is the SELECTION doing
    anything, or is any pair of venues just structurally different".
    """
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    difference = data.difference
    n_dates = len(data.dates)

    nets, fundings, divergences, dates = [], [], [], []
    for entry in range(start, n_dates - hold_days, hold_days):
        window = difference[max(0, entry - lookback + 1):entry + 1]
        with np.errstate(invalid="ignore"):
            trailing = np.nanmean(window, axis=0)
            liquid = np.nanmedian(
                data.volume[max(0, entry - lookback + 1):entry + 1], axis=0)

        usable = (np.isfinite(trailing) & np.isfinite(liquid)
                  & (liquid >= min_volume)
                  & np.isfinite(data.close_a[entry]) & np.isfinite(data.close_b[entry])
                  & np.isfinite(data.close_a[entry + hold_days])
                  & np.isfinite(data.close_b[entry + hold_days]))
        index = np.flatnonzero(usable)
        if len(index) < 3:
            continue

        if rng is not None:
            chosen = rng.choice(index, size=min(top, len(index)), replace=False)
        else:
            chosen = index[np.argsort(-np.abs(trailing[index]))][:top]

        # Short the venue whose longs pay more; long the other.
        side = -np.sign(trailing[chosen])
        side[side == 0] = 1.0

        # Funding actually settled over the hold, on days entry+1..entry+hold.
        #
        # `side` is the position's sign in venue A, so side = -1 is short A and
        # long B. A short receives the funding its longs pay, so that position
        # collects +(funding_A - funding_B) = +difference. The collected amount
        # is therefore `-side * difference`, not `+side * difference`: with the
        # wrong sign the book takes exactly the losing side of a spread it
        # correctly identified, which reads as a strategy that fails rather
        # than as a bug.
        block = difference[entry + 1:entry + hold_days + 1, chosen]
        collected = -side * np.nansum(np.nan_to_num(block, nan=0.0), axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            move_a = np.log(data.close_a[entry + hold_days, chosen]
                            / data.close_a[entry, chosen]) * 10_000.0
            move_b = np.log(data.close_b[entry + hold_days, chosen]
                            / data.close_b[entry, chosen]) * 10_000.0
        # side = -1 means short A and long B, so the price term is
        # side * (move_a - move_b): the venues' divergence, not the coin's move.
        divergence = side * (move_a - move_b)

        ok = np.isfinite(collected) & np.isfinite(divergence)
        if ok.sum() == 0:
            continue
        # Per unit of gross notional: each pair is 2 dollars gross, and both
        # legs are opened and closed, so 4 taker legs on 2 dollars = 2x cost.
        gross = float(np.mean(collected[ok] + divergence[ok])) / 2.0
        nets.append(gross - 2.0 * cost_bps)
        fundings.append(float(np.mean(collected[ok])) / 2.0)
        divergences.append(float(np.mean(divergence[ok])) / 2.0)
        dates.append(data.dates[entry])

    return PairResult(np.array(nets), np.array(fundings), np.array(divergences),
                      dates, top)


def report(result: PairResult, control: PairResult, *, hold_days: int,
           cost_bps: float) -> None:
    if len(result.net) < 3:
        print("\nToo few holding periods to say anything.")
        return
    periods_per_year = 365.0 / hold_days
    mean = float(result.net.mean())
    std = float(result.net.std(ddof=1))
    low, high = block_bootstrap(result.net)
    print()
    print("Pair backtest: {} non-overlapping {}-day holds, {} pairs a time, "
          "{:.1f} bps a leg".format(len(result.net), hold_days,
                                    result.positions, cost_bps))
    print("  funding collected   {:+8.1f} bps per period".format(
        float(result.funding.mean())))
    print("  venue divergence    {:+8.1f}".format(float(result.divergence.mean())))
    print("  round trip          {:+8.1f}  (four taker legs on two dollars)".format(
        -2.0 * cost_bps))
    print("  net                 {:+8.1f}  [{:+.1f}, {:+.1f}]".format(mean, low, high))
    print("  Sharpe              {:8.2f}   annualised {:+.1f}%".format(
        mean / std * math.sqrt(periods_per_year) if std > 0 else 0.0,
        mean * periods_per_year / 100.0))
    print("  hit rate            {:8.0%}   worst period {:+.0f}".format(
        float((result.net > 0).mean()), float(result.net.min())))
    if len(control.net) >= 3:
        print("  random-pair control {:+8.1f} bps per period".format(
            float(control.net.mean())))
        print("      The control takes the SAME trade on randomly chosen coins, "
              "so it earns\n      whatever is structural about the venue pair "
              "and nothing from selection.\n      Read the gap, not the level: "
              "if the control is most of the return, the\n      trade is "
              "'short this venue's alts' rather than 'pick the wide ones'.")

    # A spread between two venues is a fact about how they differ today, and
    # two venues can converge. Per-year is the cheapest check that the premium
    # is not one episode, and the one most likely to be omitted.
    by_year: Dict[str, List[float]] = {}
    for date, value in zip(result.dates, result.net):
        by_year.setdefault(date[:4], []).append(float(value))
    print()
    print("  net by year: " + "  ".join(
        "{} {:+.0f} ({})".format(year, float(np.mean(values)), len(values))
        for year, values in sorted(by_year.items())))
    print()
    if low > 0:
        print("VERDICT: the funding difference clears four taker legs.")
    else:
        print("VERDICT: does not clear. The difference is visible and not "
              "collectable at this cost.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel-a", type=Path, default=BLOFIN_PANEL)
    parser.add_argument("--panel-b", type=Path, default=BINANCE_PANEL)
    parser.add_argument("--name-a", default="BloFin")
    parser.add_argument("--name-b", default="Binance")
    parser.add_argument("--hold-days", type=int, default=7)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=7)
    parser.add_argument("--cost-bps", type=float, default=None,
                        help="per leg; default = the config taker fee")
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--start", type=int, default=40)
    args = parser.parse_args(argv)

    cost_bps = args.cost_bps
    if cost_bps is None:
        cost_bps = float(config.TAKER_FEE_BPS)

    data = align(load_panel(args.panel_a), load_panel(args.panel_b),
                 name_a=args.name_a, name_b=args.name_b)
    if not data.symbols:
        raise SystemExit("The two panels share no symbols.")
    describe(data)

    result = backtest(data, hold_days=args.hold_days, top=args.top,
                      cost_bps=cost_bps, min_volume=args.min_volume,
                      start=args.start, lookback=args.lookback)
    control = backtest(data, hold_days=args.hold_days, top=args.top,
                       cost_bps=cost_bps, min_volume=args.min_volume,
                       start=args.start, lookback=args.lookback, shuffle_seed=5)
    report(result, control, hold_days=args.hold_days, cost_bps=cost_bps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
