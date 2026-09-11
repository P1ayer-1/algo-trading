"""What is a forecast range worth, and which half of it carries the value?

    python backend\\analysis\\range_information.py
    python backend\\analysis\\range_information.py --symbols BTCUSDT --centre-alphas 0,0.1,0.5,1

Step 9k killed the range fade as written. The obvious next move is to forecast
the range with a model, so this prices the forecast BEFORE anyone builds one:
it feeds the fade a range interpolated in log space between the TRAILING range
(no information) and the TRUE FUTURE high/low (perfect information), dialling
the centre and the width in separately.

Measured 2026-09-11 - 24h horizon, five majors, a year of 1m bars, entry 0.25 /
stop 0.25 / target 0.5 of width, pessimistic fills, VIP 1 fees, 5x:

    centre  width   mean bps per trade
    0.00    0.00    -7.1     <- the strategy as it stands
    0.00    1.00    -15.9    <- PERFECT width knowledge, and it is WORSE
    0.25    0.00    +11.2
    0.50    0.00    +68.7
    1.00    0.00    +102.1

The half that is predictable - next-day width, IC ~0.4 off the trailing width -
is worth nothing here, and the half that pays is WHERE the range sits, which is
the direction problem steps 6, 7 and 8 already failed to solve. A model that
forecasts ranges symmetrically around the current price is forecasting the
worthless half.

Reading alpha as model skill
----------------------------
`alpha` is the weight on the perfect forecast in log space. Under a
linear-Gaussian model the optimal shrinkage of a forecast correlated rho with
the truth is rho itself, so alpha is read as "a model whose centre IC is
alpha". That is an approximation - it assumes the error is unbiased and
independent of the level - which is why what comes out of it is a THRESHOLD to
clear rather than a P&L forecast.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bars_import import daterange, kline_urls  # noqa: E402
from analysis.range_backtest import (  # noqa: E402
    Costs,
    Ohlc,
    RollingRange,
    contiguous,
    load_ohlc,
    simulate,
    trailing_max,
    trailing_min,
)
from trading.strategies.range_trade.levels import RangeParams  # noqa: E402

MINUTE_MS = 60_000
BPS = 10_000.0
MEASURED_ON = "2026-09-11"
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,ADAUSDT"

# (alpha, mean bps per trade), measured on MEASURED_ON by this module's main().
# Re-run it after any change to the fade geometry or to costs: these numbers are
# the exchange rate between model skill and money, and everything downstream
# quotes them.
CENTRE_VALUE_BPS: Tuple[Tuple[float, float], ...] = (
    (0.00, -7.1), (0.25, 11.2), (0.50, 68.7), (0.75, 94.7), (1.00, 102.1),
)
WIDTH_VALUE_BPS: Tuple[Tuple[float, float], ...] = (
    (0.00, -7.1), (0.25, -11.6), (0.50, -11.8), (0.75, -14.6), (1.00, -15.9),
)


def future_max(values: np.ndarray, window: int) -> np.ndarray:
    """out[i] = max(values[i+1 : i+1+window]). The mirror of trailing_max."""
    return trailing_max(values[::-1], window)[::-1]


def future_min(values: np.ndarray, window: int) -> np.ndarray:
    return trailing_min(values[::-1], window)[::-1]


def future_valid(ts: np.ndarray, window: int) -> np.ndarray:
    """True at i when bars i .. i+window are consecutive minutes."""
    n = len(ts)
    out = np.zeros(n, dtype=bool)
    if n <= window:
        return out
    breaks = np.concatenate(([0], np.cumsum(np.diff(ts) != MINUTE_MS)))
    out[:n - window] = breaks[window:] == breaks[:n - window]
    return out


def blend(trailing_high: np.ndarray, trailing_low: np.ndarray,
          future_high: np.ndarray, future_low: np.ndarray, *,
          centre_alpha: float, width_alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate in LOG space between the trailing range and the true one.

    Log space rather than price space so centre and width stay independent:
    moving the centre cannot change the width, and a range is a multiplicative
    object anyway.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        centre_t = np.log(np.sqrt(trailing_high * trailing_low))
        half_t = np.log(trailing_high / trailing_low) / 2
        centre_o = np.log(np.sqrt(future_high * future_low))
        half_o = np.log(future_high / future_low) / 2
        centre = (1 - centre_alpha) * centre_t + centre_alpha * centre_o
        half = (1 - width_alpha) * half_t + width_alpha * half_o
        return np.exp(centre + half), np.exp(centre - half)


def value_of_centre_skill(alpha: float,
                          table: Sequence[Tuple[float, float]] = CENTRE_VALUE_BPS) -> float:
    """Mean bps per trade for a model with centre IC `alpha`, interpolated."""
    xs = np.array([point[0] for point in table], dtype=float)
    ys = np.array([point[1] for point in table], dtype=float)
    return float(np.interp(alpha, xs, ys))


def breakeven_centre_skill(
        table: Sequence[Tuple[float, float]] = CENTRE_VALUE_BPS) -> Optional[float]:
    """The centre IC where the fade stops losing money. None if it never does."""
    for (low_a, low_v), (high_a, high_v) in zip(table, table[1:]):
        if low_v <= 0 <= high_v and high_v != low_v:
            return low_a + (high_a - low_a) * (-low_v) / (high_v - low_v)
    return None


def load_cached(symbol: str, start: date, end: date, cache: Path) -> Ohlc:
    """1m bars already in the cache. Downloads nothing - see fetch_klines.py."""
    directory = cache / symbol
    paths = [directory / url.rsplit("/", 1)[-1]
             for url in kline_urls(symbol, daterange(start, end), "klines", "1m")]
    return load_ohlc([path for path in paths if path.exists()])


def prepare(ohlc: Ohlc, horizon: int):
    """(trailing high, trailing low, future high, future low, usable)."""
    future_high = future_max(ohlc.high, horizon)
    future_low = future_min(ohlc.low, horizon)
    past_high = trailing_max(ohlc.high, horizon)
    past_low = trailing_min(ohlc.low, horizon)
    usable = (contiguous(ohlc.ts, horizon) & future_valid(ohlc.ts, horizon)
              & np.isfinite(past_high) & np.isfinite(past_low)
              & np.isfinite(future_high) & np.isfinite(future_low))
    return past_high, past_low, future_high, future_low, usable


def run_alpha(ohlc: Ohlc, prepared, params: RangeParams, costs: Costs, *,
              centre_alpha: float, width_alpha: float, leverage: float,
              optimistic: bool = False):
    past_high, past_low, future_high, future_low, usable = prepared
    high, low = blend(past_high, past_low, future_high, future_low,
                      centre_alpha=centre_alpha, width_alpha=width_alpha)
    with np.errstate(invalid="ignore"):
        width_bps = (high - low) / ((high + low) / 2) * BPS
    rolling = RollingRange(params.lookback_minutes, high, low, width_bps,
                           np.zeros(len(high)), usable)
    return simulate(ohlc, rolling, params, costs,
                    optimistic=optimistic, leverage=leverage)


def parse_floats(text: str) -> List[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cache", type=Path, default=repo_root / "data" / "cache")
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--entry-frac", type=float, default=0.25)
    parser.add_argument("--stop-frac", type=float, default=0.25)
    parser.add_argument("--target-frac", type=float, default=0.5)
    parser.add_argument("--min-width-bps", type=float, default=50.0)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--centre-alphas", default="0,0.05,0.1,0.15,0.2,0.25,0.5,0.75,1")
    parser.add_argument("--width-alphas", default="0.25,0.5,0.75,1")
    args = parser.parse_args(argv)

    from config import MAKER_FEE_BPS, TAKER_FEE_BPS, VIP_TIER

    costs = Costs(maker_bps=float(MAKER_FEE_BPS), taker_bps=float(TAKER_FEE_BPS),
                  slippage_bps=args.slippage_bps)
    end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1))
    start = end - timedelta(days=args.days - 1)
    horizon = int(round(args.horizon_hours * 60))
    params = RangeParams(lookback_minutes=horizon, entry_frac=args.entry_frac,
                         stop_frac=args.stop_frac, target_frac=args.target_frac,
                         hold_minutes=horizon, min_width_bps=args.min_width_bps)
    problems = params.problems()
    if problems:
        raise SystemExit("Refusing to run:\n  - " + "\n  - ".join(problems))

    symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    series, prepared = {}, {}
    for symbol in symbols:
        ohlc = load_cached(symbol, start, end, args.cache)
        if len(ohlc) == 0:
            print(f"  {symbol}: nothing cached for {start}..{end} - skipped")
            continue
        series[symbol] = ohlc
        prepared[symbol] = prepare(ohlc, horizon)
    if not series:
        raise SystemExit(
            "No cached bars for any symbol.\n"
            "Fetch them first: python backend\\analysis\\fetch_klines.py "
            "--symbols <SYMBOL>")
    live = list(series)

    print(f"\nVALUE OF RANGE INFORMATION  {len(live)} symbols, {start} .. {end}, "
          f"{args.horizon_hours:g}h horizon, VIP {VIP_TIER}")
    print(f"  entry {args.entry_frac:g} / stop {args.stop_frac:g} / target "
          f"{args.target_frac:g} of width, {args.leverage:g}x, pessimistic fills")
    print("  alpha = weight on the TRUE future range, read as a model's IC\n")
    print(f"  {'centre':>7}{'width':>7}  |"
          + "".join(f"{s.replace('USDT', ''):>9}" for s in live)
          + f"{'pooled':>9}{'trades':>8}")
    print("  " + "-" * (16 + 9 * len(live) + 17))

    def sweep(centre_alpha: float, width_alpha: float) -> float:
        means, pooled, trades = [], [], 0
        for symbol in live:
            result = run_alpha(series[symbol], prepared[symbol], params, costs,
                               centre_alpha=centre_alpha, width_alpha=width_alpha,
                               leverage=args.leverage)
            means.append(result.net_bps.mean() if len(result) else np.nan)
            pooled.append(result.net_bps)
            trades += len(result)
        everything = np.concatenate(pooled) if pooled else np.empty(0)
        mean = float(everything.mean()) if len(everything) else float("nan")
        print(f"  {centre_alpha:>7.2f}{width_alpha:>7.2f}  |"
              + "".join(f"{value:>+9.1f}" for value in means)
              + f"{mean:>+9.1f}{trades:>8,}")
        return mean

    measured_centre = [(alpha, sweep(alpha, 0.0))
                       for alpha in parse_floats(args.centre_alphas)]
    print()
    measured_width = [(0.0, measured_centre[0][1] if measured_centre else float("nan"))]
    measured_width += [(alpha, sweep(0.0, alpha))
                       for alpha in parse_floats(args.width_alphas)]

    crossing = breakeven_centre_skill(tuple(measured_centre))
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    width_best = max(value for _, value in measured_width if np.isfinite(value))
    print(f"  Perfect WIDTH knowledge is worth at best {width_best:+.1f} bps per trade.")
    print(f"  Perfect CENTRE knowledge is worth {measured_centre[-1][1]:+.1f}.")
    if crossing is None:
        print("\n  No amount of centre skill in this sweep reaches break-even.")
    else:
        print(f"\n  BREAK-EVEN CENTRE SKILL: IC {crossing:.3f}. A model below that "
              f"loses money\n  however well calibrated it is - and a range forecast "
              "that is symmetric around\n  the current price has a centre IC of zero "
              "by construction.")
    print("\n  Paste the measured centre column into CENTRE_VALUE_BPS when the "
          "geometry or\n  costs above differ from the table this module ships with.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
