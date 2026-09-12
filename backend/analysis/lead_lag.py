"""Do the alts follow BTC with a lag that a hedged basket can trade?

    python backend\\analysis\\lead_lag.py
    python backend\\analysis\\lead_lag.py --window-min 60 --hold-min 60 --z 2

The hypothesis
--------------
BTC is the most liquid contract and the first to move; the alts re-price to
it a few minutes later. If that lag is longer than the time it takes to send
an order, then after a large BTC move the equal-weight basket of alts should
keep moving in BTC's direction, and a basket hedged with BTC (so the trade
earns only the CATCH-UP, not BTC's next move) should pay.

Nothing in steps 6-9ab tested this. Step 8's cross-section demeaned every
label, which removes exactly the market-wide term a lead-lag lives in; the
extreme-move tools conditioned on a coin's own move and on the index's, not
on BTC's move relative to the rest.

What is measured
----------------
1. The cross-autocorrelation between BTC's return over the last `--window-min`
   and each alt's HEDGED return over the next `--hold-min` (alt return minus
   its trailing 30-day beta times BTC's return over the same forward window),
   pooled across eligible alts, by year. Zero means no lag at that scale.
2. A trade: when |BTC's window return| exceeds `--z` times its trailing 30-day
   standard deviation at that window, take the equal-weight alt basket in
   BTC's direction and short beta units of BTC against it; hold `--hold-min`;
   never overlap positions. Cost is a taker round trip on the basket plus a
   taker round trip on the hedge, scaled by beta.
3. The same event split by which alts LAGGED (own residual over the window
   against BTC's move) and which LED, because "the laggards catch up" and "the
   basket keeps going" are different claims and only the first is a lead-lag.

A placebo repeats the trade at the same clock times one day later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.intraday import build_grid, log_returns  # noqa: E402

STEP = 15
BTC = "BTCUSDT"


def daily_eligibility(grid, min_volume: float, min_complete: float = 0.9):
    """Trailing-30-day median daily volume floor, held for the day, per bar."""
    n_t, n_s = grid.close.shape
    day_bars = 1440 // STEP
    n_days = n_t // day_bars + 1
    daily_volume = np.full((n_days, n_s), np.nan)
    daily_complete = np.zeros((n_days, n_s))
    for k in range(n_days):
        block = grid.volume[k * day_bars:(k + 1) * day_bars]
        if len(block):
            daily_volume[k] = np.nansum(block, axis=0)
            daily_complete[k] = grid.complete[k * day_bars:(k + 1) * day_bars].mean(axis=0)
    eligible_day = np.zeros_like(daily_volume, dtype=bool)
    for k in range(30, n_days):
        with np.errstate(invalid="ignore"):
            eligible_day[k] = ((np.nanmedian(daily_volume[k - 30:k], axis=0) >= min_volume)
                               & (daily_complete[k - 30:k].mean(axis=0) >= min_complete))
    return eligible_day[np.arange(n_t) // day_bars]


def trailing_beta(ret: np.ndarray, market: np.ndarray, days: int = 30):
    """Beta of each column on `market`, from the prior `days` days, held a day."""
    n_t, n_s = ret.shape
    day_bars = 1440 // STEP
    beta = np.full(ret.shape, np.nan)
    for k in range(days, n_t // day_bars + 1):
        lo, hi = (k - days) * day_bars, k * day_bars
        m = market[lo:hi]
        r = ret[lo:hi]
        ok = np.isfinite(r) & np.isfinite(m)[:, None]
        mm = np.where(ok, m[:, None], np.nan)
        rr = np.where(ok, r, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            m_mean = np.nanmean(mm, axis=0)
            r_mean = np.nanmean(rr, axis=0)
            cov = np.nanmean((mm - m_mean) * (rr - r_mean), axis=0)
            var = np.nanmean((mm - m_mean) ** 2, axis=0)
            b = cov / var
        b[ok.sum(axis=0) < day_bars * days // 2] = np.nan
        beta[hi:hi + day_bars] = b
    return beta


def trailing_sd(values: np.ndarray, stride: int, days: int = 30):
    n_t = len(values)
    day_bars = 1440 // STEP
    sd = np.full(n_t, np.nan)
    for k in range(days, n_t // day_bars + 1):
        lo, hi = (k - days) * day_bars, k * day_bars
        sample = values[lo:hi:stride]
        sample = sample[np.isfinite(sample)]
        if len(sample) > 20:
            sd[hi:hi + day_bars] = sample.std()
    return sd


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--window-min", type=int, default=15)
    parser.add_argument("--hold-min", type=int, default=15)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--cost-bps", type=float, default=None, help="per leg")
    parser.add_argument("--delay-min", type=int, default=0,
                        help="enter this many minutes after the signal bar closes")
    args = parser.parse_args(argv)
    cost_leg = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps
    w, h, d = args.window_min // STEP, args.hold_min // STEP, args.delay_min // STEP

    grid = build_grid(STEP, fields=("close", "volume"))
    close = grid.close
    n_t, n_s = close.shape
    b = grid.symbols.index(BTC)
    eligible = daily_eligibility(grid, args.min_volume) & grid.complete
    eligible[:, b] = False                       # BTC is the hedge, never in the basket

    r1 = log_returns(close, 1)
    beta = trailing_beta(r1, r1[:, b])
    back = log_returns(close, w)                 # window return ending at row t
    btc_back = back[:, b]
    fwd = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd[:-(h + d)] = np.log(close[h + d:] / close[d:n_t - h]) * 10_000.0
    hedged = fwd - beta * fwd[:, b][:, None]
    years = np.array([dd[:4] for dd in grid.dates])

    print("window {}m, hold {}m, delay {}m, |BTC move| > {:g} sd, cost {:.0f}/leg, {} alts max".format(
        args.window_min, args.hold_min, args.delay_min, args.z, cost_leg, n_s - 1))

    # 1. Cross-autocorrelation, pooled over eligible alt-bars.
    print("\ncorr(BTC window return, alt HEDGED forward return), pooled; and raw (unhedged)")
    print("  {:>6} {:>10} {:>10} {:>10} {:>12}".format("year", "n", "hedged", "raw", "alts lead BTC"))
    basket_back = np.nanmean(np.where(eligible, back, np.nan), axis=1)
    btc_fwd = fwd[:, b]
    for year in sorted(set(years)):
        rows = np.flatnonzero((years == year))
        x = np.repeat(btc_back[rows][:, None], n_s, axis=1)
        yh = hedged[rows]
        yr = fwd[rows]
        ok = eligible[rows] & np.isfinite(x) & np.isfinite(yh) & np.isfinite(yr)
        if ok.sum() < 1000:
            continue
        ch = np.corrcoef(x[ok], yh[ok])[0, 1]
        cr = np.corrcoef(x[ok], yr[ok])[0, 1]
        okb = np.isfinite(basket_back[rows]) & np.isfinite(btc_fwd[rows])
        cl = np.corrcoef(basket_back[rows][okb], btc_fwd[rows][okb])[0, 1]
        print("  {:>6} {:>10d} {:>+10.4f} {:>+10.4f} {:>+12.4f}".format(year, int(ok.sum()), ch, cr, cl))

    # 2. The trade.
    sd = trailing_sd(btc_back, w)
    signal = np.isfinite(btc_back) & np.isfinite(sd) & (np.abs(btc_back) > args.z * sd)

    def run(shift_rows: int = 0):
        out = []
        t = 0
        while t < n_t - h - d:
            if not signal[t]:
                t += 1
                continue
            row = t + shift_rows
            if row < 0 or row >= n_t - h - d:
                t += 1
                continue
            sign = 1.0 if btc_back[t] > 0 else -1.0
            names = eligible[row] & np.isfinite(hedged[row]) & np.isfinite(beta[row])
            if names.sum() < 10:
                t += 1
                continue
            basket = float(np.mean(hedged[row][names]))
            mean_beta = float(np.mean(beta[row][names]))
            cost = 2 * cost_leg * (1.0 + abs(mean_beta))
            # Laggards: alts whose own residual over the window opposed BTC's move.
            resid = back[row] - beta[row] * btc_back[t]
            lag = names & (sign * resid < 0)
            led = names & (sign * resid > 0)
            lag_pnl = sign * float(np.mean(hedged[row][lag])) - cost if lag.sum() >= 5 else np.nan
            led_pnl = sign * float(np.mean(hedged[row][led])) - cost if led.sum() >= 5 else np.nan
            raw_basket = sign * float(np.mean(fwd[row][names]))
            out.append((t, sign * basket - cost, sign * basket, raw_basket, lag_pnl, led_pnl,
                        sign * btc_back[t], int(names.sum())))
            t += h + d                              # never overlap
        return out

    real = run()
    placebo = run(shift_rows=96)
    if not real:
        print("no events")
        return 0
    arr = np.array([r[1:] for r in real], dtype=float)
    rows = np.array([r[0] for r in real])
    parr = np.array([r[1:] for r in placebo], dtype=float) if placebo else np.zeros((0, 7))

    def se(v):
        v = v[np.isfinite(v)]
        return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")

    print("\nhedged basket in BTC's direction after a {:g}-sd BTC move, non-overlapping".format(args.z))
    print("  events {}   mean |BTC move| {:.0f} bps   mean basket size {:.0f}".format(
        len(real), np.mean(np.abs(arr[:, 5])), arr[:, 6].mean()))
    print("  net {:+.2f} ±{:.2f}   gross hedged {:+.2f}   raw unhedged {:+.2f}   placebo net {:+.2f} ±{:.2f}".format(
        arr[:, 0].mean(), se(arr[:, 0]), arr[:, 1].mean(), arr[:, 2].mean(),
        parr[:, 0].mean() if len(parr) else float("nan"), se(parr[:, 0]) if len(parr) else float("nan")))
    print("  laggards only net {:+.2f} ±{:.2f}   leaders only net {:+.2f} ±{:.2f}".format(
        np.nanmean(arr[:, 3]), se(arr[:, 3]), np.nanmean(arr[:, 4]), se(arr[:, 4])))
    print("  {:>6} {:>7} {:>9} {:>7} {:>9} {:>9} {:>9}".format(
        "year", "events", "net", "se", "gross", "laggards", "leaders"))
    for year in sorted(set(years[rows])):
        pick = years[rows] == year
        print("  {:>6} {:>7d} {:>+9.2f} {:>7.2f} {:>+9.2f} {:>+9.2f} {:>+9.2f}".format(
            year, int(pick.sum()), arr[pick, 0].mean(), se(arr[pick, 0]), arr[pick, 1].mean(),
            np.nanmean(arr[pick, 3]), np.nanmean(arr[pick, 4])))
    # Does it scale with the size of BTC's move?
    z = np.abs(arr[:, 5]) / np.where(np.isfinite(sd[rows]), sd[rows], np.nan)
    print("  by z: ", end="")
    for lo, hi in ((args.z, args.z + 1), (args.z + 1, args.z + 2), (args.z + 2, args.z + 4), (args.z + 4, 1e9)):
        pick = (z >= lo) & (z < hi)
        if pick.sum() >= 30:
            print("[{:g},{:g}) n={} gross {:+.1f}  ".format(lo, hi, int(pick.sum()), arr[pick, 1].mean()), end="")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
