"""Where the touch-probability formula stops being usable, measured.

Step 9o left a calibrated instrument:

    P(price touches a level within H) = 2 * Phi(-distance / (sigma * sqrt(H)))

which is the reflection principle, and it works because the market is a
near-random walk at hourly-and-longer scales (VR 1.00-1.13). The obvious next
move is to point it at the thing in this repo that makes money and replace a
static risk buffer with a probability - "P(mark touches liquidation before you
exit) = 0.4%" instead of "liquidation must be 15% away".

That proposal has a hole in it, and this file is the hole. The reflection
principle needs Brownian motion, which is a stronger assumption than a random
walk: not just uncorrelated increments but GAUSSIAN ones. The variance-ratio
test that justified the formula only checked the first. Crypto returns are
uncorrelated and fat-tailed at the same time, and every risk question worth
asking - liquidation, ruin, how bad can one week be - lives in the tail the
variance ratio never looked at.

So the question is not "is the formula right" but "out to what distance", and
that is an empirical question with a number for an answer.

What this measures
------------------
Non-overlapping returns, standardised, pooled, counted against what a Gaussian
predicts at each distance. Two populations, because they answer different
questions:

- SINGLE ASSETS: 62 symbols, 10,298 weekly returns. This is the population the
  formula was validated on and the one `strategies/carry` would use it for.
- THE BOOK: the cross-sectional carry factor's own period returns, pooled over
  six grid cells so the answer is not one specification's luck.

Measured 2026-09-12 on the Binance daily panel, 7-day non-overlapping holds,
as `actual exceedances / Gaussian prediction`:

    distance      single assets        the book
      1.0 sd            0.8x             0.7x
      2.0 sd            0.9x             0.9x
      2.5 sd            1.7x             2.2x
      3.0 sd            3.2x             7.6x
      4.0 sd           24.5x           173.2x
      5.0 sd            0.0x        16,748.9x

Two findings, and the second is the one that matters.

**The formula is a moderate-distance instrument.** Out to about 2 sigma it is
calibrated for both populations, which is exactly the regime step 9o validated
it in. Past 2.5 sigma it degrades, and by 4 sigma it is wrong by one to two
orders of magnitude. It is not a tail model and was never tested as one.

**The book's tail is much fatter than its legs'.** At 4 sigma the book is
overconfident by 173x against single assets' 24.5x - the cross-section is
SEVEN TIMES worse than the instruments it is built from. That is the opposite
of what diversification is supposed to do, and step 9v already named the
mechanism: in February 2024 the book was short SHIB, PEPE and BONK because they
had the highest funding, they returned +256%, +288% and +180% in a week, and in
the 90 days beforehand their residual correlations were +0.12, -0.13 and +0.21.
The correlation that did the damage did not exist in the data yet. A book
sized on a model that assumes it away is sized on a fiction.

What follows for sizing
-----------------------
The worst observed period was -1,988 bps. Gaussian at the same sigma calls that
a 1-in-6,142,475-period event; it happened once in 243. For a $2,000 loss
budget:

    Gaussian at the 1/n quantile       -532 bps  ->  $37,599 gross
    empirical 1st percentile           -444 bps  ->  $45,086 gross
    worst actually observed            -976 bps  ->  $20,500 gross

The Gaussian would let you run 1.8x too big, and the empirical worst is itself
one draw from 243 periods, so even $20,500 is not a bound.
"""

from __future__ import annotations

import sys
from math import erfc, sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DISTANCES: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)

# Where "calibrated" stops. Wide on purpose: with a few dozen expected
# exceedances the count itself is noisy, and a band that calls 1.9x a failure
# would be reporting sampling error as a finding.
CALIBRATED = (0.5, 2.0)


def gaussian_tail(k: float) -> float:
    """P(Z < -k) for a standard normal, via erfc so the far tail keeps its
    precision. `1 - Phi(k)` loses every significant digit past about 8 sigma,
    which is exactly where this is being asked."""
    return 0.5 * erfc(k / sqrt(2.0))


def standardise(returns: Sequence[float]) -> np.ndarray:
    """Zero mean, unit sd. Returns an empty array when that is undefined.

    Standardising per series before pooling is what makes symbols with
    different volatilities comparable - pooling raw returns would just measure
    the spread of volatilities across the universe.
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.array([])
    sd = values.std(ddof=1)
    if sd <= 0:
        return np.array([])
    return (values - values.mean()) / sd


def exceedance_table(z: np.ndarray,
                     distances: Sequence[float] = DISTANCES,
                     ) -> List[Dict[str, float]]:
    """How often the LOSS tail was reached, against what a Gaussian predicts.

    The loss side only. A symmetric test would average an overshoot on one side
    against an undershoot on the other and report a fit that neither side has.
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    rows: List[Dict[str, float]] = []
    for k in distances:
        predicted = gaussian_tail(k) * n
        actual = int((z < -k).sum())
        rows.append({
            "k": float(k),
            "predicted": float(predicted),
            "actual": float(actual),
            "ratio": (actual / predicted) if predicted > 0 else float("inf"),
        })
    return rows


def calibrated_to(rows: Sequence[Dict[str, float]]) -> float:
    """The largest distance the Gaussian is still usable at.

    Scans outward and stops at the first failure rather than reporting the
    furthest passing distance, because the property wanted is "trustworthy up
    to here" and one passing cell beyond a failure is noise, not a reprieve.
    """
    best = 0.0
    for row in rows:
        low, high = CALIBRATED
        if row["predicted"] < 1.0:
            # Fewer than one expected exceedance cannot confirm anything; the
            # ratio is then a statement about a single observation.
            break
        if not (low <= row["ratio"] <= high):
            break
        best = row["k"]
    return best


def _print_table(title: str, z: np.ndarray) -> None:
    print(title)
    print("  {} observations, excess kurtosis {:+.2f}".format(
        len(z), float((z ** 4).mean() - 3.0) if len(z) else float("nan")))
    print("  {:>5}  {:>12}  {:>9}  {:>12}".format(
        "sd", "Gaussian n", "actual n", "actual/pred"))
    rows = exceedance_table(z)
    for row in rows:
        flag = "  <- calibrated" if CALIBRATED[0] <= row["ratio"] <= CALIBRATED[1] else ""
        print("  {:>5.1f}  {:>12.3f}  {:>9.0f}  {:>11.1f}x{}".format(
            row["k"], row["predicted"], row["actual"], row["ratio"], flag))
    print("  usable out to {:.1f} sd".format(calibrated_to(rows)))
    print()


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel", type=Path,
                        default=Path("data") / "panel" / "daily.csv")
    parser.add_argument("--hold-days", type=int, default=7)
    parser.add_argument("--min-volume", type=float, default=5e6)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    from analysis.factor_panel import (
        build_features, load_panel, run_factor, tradeable,
    )

    panel = load_panel(args.panel)
    hold = args.hold_days

    # --- single assets ----------------------------------------------------
    eligible = tradeable(panel, min_history=90, min_volume=args.min_volume)
    pooled: List[np.ndarray] = []
    for column in range(panel.close.shape[1]):
        prices = panel.close[:, column]
        index = np.flatnonzero(eligible[:, column] & np.isfinite(prices))
        if len(index) < 200:
            continue
        series = prices[index[0]:index[-1] + 1]
        if not np.all(np.isfinite(series)):
            continue
        entries = np.arange(0, len(series) - hold, hold)
        z = standardise(np.log(series[entries + hold] / series[entries]))
        if len(z) >= 50:
            pooled.append(z)
    _print_table("SINGLE ASSETS ({} symbols)".format(len(pooled)),
                 np.concatenate(pooled) if pooled else np.array([]))

    # --- the book, over the grid so it is not one cell's luck --------------
    features = build_features(panel)
    book: List[np.ndarray] = []
    for top_frac in (0.2, 0.3):
        for min_volume in (2e6, 5e6, 2e7):
            usable = tradeable(panel, min_history=90, min_volume=min_volume)
            result = run_factor(panel, features["carry_7"], usable,
                                hold_days=hold, top_frac=top_frac,
                                cost_bps=10.0, start=120)
            z = standardise(result.net)
            if len(z):
                book.append(z)
    _print_table("THE CROSS-SECTIONAL BOOK ({} grid cells)".format(len(book)),
                 np.concatenate(book) if book else np.array([]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
