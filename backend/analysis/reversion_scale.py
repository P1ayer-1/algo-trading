"""Is the mean reversion real, or is it the spread? And how strong would it need to be?

    python backend\\analysis\\reversion_scale.py
    python backend\\analysis\\reversion_scale.py --symbols BTCUSDT --sweep

Two questions a reversion strategy has to answer before it is written, and
neither needs the strategy to exist yet.

**Is the reversion tradeable?** The variance ratio over 24h - Var(24h returns)
over 24h x Var(short returns) - is below 1 for every crypto major measured
(BTC 0.90, DOGE 0.61 on a 1m base, 2026-09-11), which reads as mean reversion
and is the statistic a range strategy is implicitly betting on. Measure it
against a LONGER base interval and it climbs back to 1:

    symbol    1m base    5m     15m     60m
    BTCUSDT     0.904  0.953   0.985   1.044
    DOGEUSDT    0.613  0.371   0.949   1.015
    synth OU    0.796  0.799   0.793   0.794   <- genuine reversion, flat
    synth RW    1.074  1.076   1.066   1.056

Real reversion is SCALE-INVARIANT: an Ornstein-Uhlenbeck path pulls back the
same whether it is sampled every minute or every hour, and its ratio is flat.
What decays as the base lengthens is bid-ask bounce - the price alternating
between touching bid and ask inflates the shortest interval's variance and
nothing else. That is the Roll (1984) effect, it looks exactly like mean
reversion in the statistic, and it cannot be traded: the bounce IS the spread
you pay to touch it.

**How strong would it have to be?** The sweep runs the identical fade over
Ornstein-Uhlenbeck paths at the same volatility as the majors and a range of
reversion speeds. Measured 2026-09-11 (entry 0.25 / stop 0.25 / target 0.5 of
width, 24h lookback, VIP 1, pessimistic fills):

    half-life   variance ratio   net bps per trade
    1h                   0.055              +56.0
    6h                   0.322              +41.1
    24h                  0.709              +16.6
    48h                  0.824              +10.8
    random walk          0.986               -0.1

So a genuine 48h half-life would pay +10.8 bps a trade. The majors show ratios
in that band on a 1m base and the fade still loses 7.7 bps (README step 9m),
which is the whole point of the first table: their ratio is not reversion.

Synthetic data cannot supply an edge - a generator only returns the assumptions
put into it, and a mean-reverting generator makes any fade look brilliant. What
it CAN do is calibrate a yardstick, and that is all it is used for here.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.range_backtest import Costs, Ohlc, RollingRange, simulate  # noqa: E402
from analysis.range_information import load_cached  # noqa: E402
from trading.strategies.range_trade.levels import RangeParams  # noqa: E402

MINUTE_MS = 60_000
T0 = 20_000 * 86_400_000 + 59_999
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,ADAUSDT"
DEFAULT_BASES = (1, 5, 15, 60)
# How far below 1.0 a ratio must sit before it is called reversion at all.
TOLERANCE = 0.05


def variance_ratio(close: np.ndarray, base_minutes: int,
                   window_minutes: int = 1440) -> float:
    """Var(window returns) / (k x Var(base returns)). 1.0 is a random walk.

    Below 1 means the long horizon moved less than the short one implies, which
    is what mean reversion looks like - and also what a bid-ask bounce looks
    like, which is why this is measured at several base intervals.
    """
    if base_minutes < 1 or window_minutes <= base_minutes:
        raise ValueError("window_minutes must exceed base_minutes >= 1")
    base = (np.log(close[base_minutes::base_minutes])
            - np.log(close[:-base_minutes:base_minutes]))
    window = (np.log(close[window_minutes::window_minutes])
              - np.log(close[:-window_minutes:window_minutes]))
    if len(base) < 3 or len(window) < 3:
        return float("nan")
    spread = np.var(base, ddof=1)
    if spread <= 0:
        return float("nan")
    return float(np.var(window, ddof=1) / ((window_minutes / base_minutes) * spread))


def variance_ratio_curve(close: np.ndarray, bases: Sequence[int] = DEFAULT_BASES,
                         window_minutes: int = 1440) -> Dict[int, float]:
    return {base: variance_ratio(close, base, window_minutes) for base in bases}


def classify(curve: Dict[int, float], tolerance: float = TOLERANCE) -> str:
    """`artifact`, `scale-invariant` or `none`.

    The distinction that matters: reversion a strategy can trade shows up at
    every sampling interval, while a spread artifact lives only at the
    shortest one and washes out as the base lengthens.
    """
    usable = {base: value for base, value in curve.items() if np.isfinite(value)}
    if len(usable) < 2:
        return "none"
    bases = sorted(usable)
    shortest, longest = usable[bases[0]], usable[bases[-1]]
    if shortest >= 1 - tolerance:
        return "none"
    if longest >= 1 - tolerance:
        return "artifact"
    return "scale-invariant"


def ou_bars(days: int, half_life_hours: Optional[float], *, daily_vol: float = 0.03,
            seed: int = 1, sub_steps: int = 60) -> Ohlc:
    """1m OHLC from an Ornstein-Uhlenbeck log-price path.

    Built from `sub_steps` ticks inside each minute so the bar's high and low
    are real extremes of a path rather than two draws - a fade lives on exactly
    those extremes, and a bar built without them understates how often a level
    is touched. `half_life_hours=None` gives a driftless random walk.
    """
    minutes = days * 1440
    steps = minutes * sub_steps
    rng = np.random.default_rng(seed)
    dt = 1.0 / (sub_steps * 1440)
    shocks = rng.normal(0.0, daily_vol * np.sqrt(dt), steps)

    if half_life_hours is None:
        path = np.cumsum(shocks)
    else:
        from scipy.signal import lfilter
        kappa = np.log(2) / (half_life_hours / 24)
        # x_t = decay * x_{t-1} + shock_t, run exactly as a one-pole filter
        # rather than in a multi-million step Python loop.
        path = lfilter([1.0], [1.0, -(1.0 - kappa * dt)], shocks)

    grid = (100.0 * np.exp(path)).reshape(minutes, sub_steps)
    return Ohlc(ts=T0 + MINUTE_MS * np.arange(minutes, dtype=np.int64),
                open=grid[:, 0].copy(), high=grid.max(axis=1),
                low=grid.min(axis=1), close=grid[:, -1].copy())


def fade_net_bps(ohlc: Ohlc, params: RangeParams, costs: Costs,
                 leverage: float = 5.0) -> Tuple[int, float]:
    trades = simulate(ohlc, RollingRange.build(ohlc, params.lookback_minutes),
                      params, costs, optimistic=False, leverage=leverage)
    if not len(trades):
        return 0, float("nan")
    return len(trades), float(trades.net_bps.mean())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cache", type=Path, default=repo_root / "data" / "cache")
    parser.add_argument("--bases", default="1,5,15,60",
                        help="Base intervals in minutes.")
    parser.add_argument("--window-hours", type=float, default=24.0)
    parser.add_argument("--sweep", action="store_true",
                        help="Also run the fade over OU paths to price the reversion.")
    parser.add_argument("--sweep-days", type=int, default=180)
    parser.add_argument("--daily-vol", type=float, default=0.03)
    parser.add_argument("--seeds", default="1,2,3")
    args = parser.parse_args(argv)

    bases = [int(part) for part in args.bases.split(",") if part.strip()]
    window = int(round(args.window_hours * 60))
    if any(base < 1 for base in bases) or window <= max(bases):
        raise SystemExit("Refusing to run:\n  - every base must be >= 1 minute and "
                         "shorter than --window-hours")
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    start = end - timedelta(days=args.days - 1)

    print(f"\nVARIANCE RATIO OVER {args.window_hours:g}h  {start} .. {end}")
    print("  Below 1 reads as mean reversion. Real reversion is the same at every "
          "base interval;\n  a bid-ask bounce lives only in the shortest one.\n")
    header = "".join(f"{f'{base}m':>9}" for base in bases)
    print(f"  {'symbol':<12}{header}   verdict")
    print("  " + "-" * (14 + 9 * len(bases) + 20))

    verdicts = []
    for symbol in [part.strip().upper() for part in args.symbols.split(",") if part.strip()]:
        ohlc = load_cached(symbol, start, end, args.cache)
        if len(ohlc) == 0:
            print(f"  {symbol:<12} nothing cached - skipped")
            continue
        curve = variance_ratio_curve(ohlc.close, bases, window)
        verdict = classify(curve)
        verdicts.append(verdict)
        row = "".join(f"{curve[base]:>9.3f}" for base in bases)
        print(f"  {symbol:<12}{row}   {verdict}")

    print("\n  yardsticks, same measurement on generated paths:")
    for label, half_life in (("random walk", None), ("OU, HL 24h", 24.0)):
        ohlc = ou_bars(args.sweep_days, half_life, daily_vol=args.daily_vol, seed=seeds[0])
        curve = variance_ratio_curve(ohlc.close, bases, window)
        row = "".join(f"{curve[base]:>9.3f}" for base in bases)
        print(f"  {label:<12}{row}   {classify(curve)}")

    if verdicts:
        artifacts = verdicts.count("artifact")
        print(f"\n  {artifacts}/{len(verdicts)} symbols show reversion at the shortest "
              "base that is GONE by the longest.")
        if artifacts:
            print("  That is the spread, not the market returning to a level, and a "
                  "strategy that\n  fades it pays that spread on the way in.")

    if not args.sweep:
        print("\n  Pass --sweep to price how strong genuine reversion would have to be.")
        return 0

    costs_source = None
    try:
        from config import MAKER_FEE_BPS, TAKER_FEE_BPS, VIP_TIER
        costs = Costs(maker_bps=float(MAKER_FEE_BPS), taker_bps=float(TAKER_FEE_BPS),
                      slippage_bps=3.0)
        costs_source = f"VIP {VIP_TIER}"
    except SystemExit:                                   # no fee schedule configured
        costs = Costs(maker_bps=0.6, taker_bps=5.0, slippage_bps=3.0)
        costs_source = "assumed 0.6/5.0"

    params = RangeParams(lookback_minutes=window, entry_frac=0.25, stop_frac=0.25,
                         target_frac=0.5, hold_minutes=window, min_width_bps=50.0)
    print(f"\nWHAT THE FADE EARNS AGAINST REVERSION THAT IS REAL  "
          f"({args.sweep_days}d paths, {args.daily_vol:.0%}/day, {costs_source})")
    print(f"\n  {'half-life':<20}{'variance ratio':>16}{'trades':>9}{'net bps':>10}")
    print("  " + "-" * 55)
    for half_life in (1.0, 3.0, 6.0, 12.0, 24.0, 48.0, None):
        nets, ratios, trades = [], [], 0
        for seed in seeds:
            ohlc = ou_bars(args.sweep_days, half_life, daily_vol=args.daily_vol, seed=seed)
            ratios.append(variance_ratio(ohlc.close, bases[0], window))
            count, mean = fade_net_bps(ohlc, params, costs)
            trades += count
            if count:
                nets.append(mean * count)
        label = "none (random walk)" if half_life is None else f"{half_life:g}h"
        net = sum(nets) / trades if trades else float("nan")
        print(f"  {label:<20}{np.mean(ratios):>16.3f}{trades:>9,}{net:>+10.1f}")

    print("\n  A generated path returns the assumptions put into it: this prices a "
          "yardstick,\n  it does not evidence an edge. What the market actually did "
          "is README step 9m.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
