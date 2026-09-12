"""The live perp premium as an intraday cross-sectional signal.

    python backend\\analysis\\premium_signal.py --hold-min 60
    python backend\\analysis\\premium_signal.py --hold-min 240 --cost-bps 10 --tail 10

Why the premium and not the funding rate
----------------------------------------
Funding is the premium averaged over eight hours and paid at the end. Every
funding-based signal in this repo (`carry_last`, the settlement studies)
therefore sees the perp-index basis with a lag of hours, three times a day.
`fetch_premium_index.py` holds the premium itself at 15 minutes, so the
question "what does a perp trading far above its index do next" can be
asked at every bar close with nothing from the future in it. That is the
honest version of the test step 9ab ran with placebo instants that used
the settled rate, and it is why the tail of that study (+49 bps in the
hour after a >10 bps settlement, gone with a 15-minute delay) needed this.

Signs, stated before the run (HIGH score = expected to outperform)
-------------------------------------------------------------------
  prem_now   negated premium at the bar close: a perp above its index
             reverts toward it, so the perp underperforms   -> factor is -premium
  prem_1h    negated mean premium over the last hour
  prem_8h    negated mean over the last 8h (the rate that WILL print)
  prem_chg   negated (1h mean - 8h mean): a premium that has just widened

The tail is priced separately: long every coin whose 1h mean premium is
above `--tail` bps at the bar close, held `--hold-min`, versus short. The
two are opposite bets and the year table decides which, if either, is real.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.factor_panel import block_bootstrap, run_factor  # noqa: E402
from analysis.fetch_premium_index import OUT as PREMIUM_DIR, load_premium  # noqa: E402
from analysis.intraday import build_grid  # noqa: E402
from analysis.lead_lag import daily_eligibility, STEP  # noqa: E402

MINUTE_MS = 60_000


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    filled = np.nan_to_num(values, nan=0.0)
    count = np.isfinite(values).astype(float)
    cs = np.cumsum(filled, axis=0)
    cc = np.cumsum(count, axis=0)
    out = np.full(values.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = cs[window:] - cs[:-window]
        den = cc[window:] - cc[:-window]
        out[window:] = np.where(den >= window * 0.8, num / den, np.nan)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hold-min", type=int, default=60)
    parser.add_argument("--top-frac", type=float, default=0.3)
    parser.add_argument("--tail", type=float, default=10.0, help="bps of 1h mean premium")
    parser.add_argument("--min-volume", type=float, default=5e6)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--start", type=int, default=3000)
    args = parser.parse_args(argv)
    cost = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps
    h = args.hold_min // STEP

    grid = build_grid(STEP, fields=("close", "volume", "funding"))
    n_t, n_s = grid.close.shape
    premium = np.full(grid.close.shape, np.nan)
    have = 0
    for s, symbol in enumerate(grid.symbols):
        path = PREMIUM_DIR / (symbol + ".15m.npz")
        if not path.exists():
            continue
        p = load_premium(symbol)
        # A premium bar opening at T closes at T + 15m; the grid row is the close.
        idx = (p["ts_open"] + STEP * MINUTE_MS - grid.ts_end[0]) // (STEP * MINUTE_MS)
        ok = (idx >= 0) & (idx < n_t)
        premium[idx[ok], s] = p["close"][ok]
        have += 1
    print("premium loaded for {} of {} symbols".format(have, n_s))
    eligible = daily_eligibility(grid, args.min_volume) & grid.complete & np.isfinite(premium)
    prem_1h = trailing_mean(premium, 4)
    prem_8h = trailing_mean(premium, 32)
    feats = {"prem_now": -premium, "prem_1h": -prem_1h, "prem_8h": -prem_8h,
             "prem_chg": -(prem_1h - prem_8h)}
    panel = grid.to_panel()
    periods_per_year = 365.0 * 1440 / args.hold_min
    years = np.array([d[:4] for d in grid.dates])

    print("hold {}m, top/bottom {:.0%}, cost {:.0f} bps, eligible median {:.0f}".format(
        args.hold_min, args.top_frac, cost, np.median(eligible[args.start:].sum(axis=1))))
    print("{:<10} {:>7} {:>16} {:>7} {:>6} {:>6} {:>7} {:>7} {:>7} {:>5}  {}".format(
        "factor", "net", "95% block", "Sharpe", "turn", "IC", "price", "fund", "ctrl", "pct", "by year"))
    for name, score in feats.items():
        real = run_factor(panel, score, eligible, hold_days=h, top_frac=args.top_frac,
                          cost_bps=cost, start=args.start)
        if len(real.periods) < 10:
            continue
        net = real.net
        controls = np.array([run_factor(panel, score, eligible, hold_days=h, top_frac=args.top_frac,
                                        cost_bps=cost, start=args.start, shuffle_seed=seed).net.mean()
                             for seed in range(args.shuffles)])
        lo, hi = block_bootstrap(net, block=max(4, 1440 // args.hold_min))
        sharpe = net.mean() / net.std(ddof=1) * math.sqrt(periods_per_year)
        entry_years = years[[p.entry for p in real.periods]]
        by_year = " ".join("{}:{:+.1f}".format(y[2:], net[entry_years == y].mean()) for y in sorted(set(entry_years)))
        print("{:<10} {:>+7.2f} [{:>+6.1f},{:>+6.1f}] {:>7.2f} {:>6.2f} {:>+6.3f} {:>+7.2f} {:>+7.2f} {:>+7.2f} {:>4.0%}  {}".format(
            name, net.mean(), lo, hi, sharpe, np.mean([p.turnover for p in real.periods]), real.mean_ic,
            np.mean([p.price_bps for p in real.periods]), np.mean([p.funding_bps for p in real.periods]),
            controls.mean(), (net.mean() > controls).mean(), by_year))

    # The tail: 1h mean premium beyond the threshold at the bar close.
    close = grid.close
    fwd = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd[:-h] = np.log(close[h:] / close[:-h]) * 1e4
    market = np.nanmean(np.where(eligible & np.isfinite(fwd), fwd, np.nan), axis=1)
    ex = fwd - market[:, None]
    for label, mask in (("premium > +{:g}".format(args.tail), eligible & (prem_1h > args.tail)),
                        ("premium < -{:g}".format(args.tail), eligible & (prem_1h < -args.tail))):
        rows_all, cols_all = np.nonzero(mask & np.isfinite(ex))
        keep, last = np.ones(len(rows_all), dtype=bool), {}
        for i, (r, c) in enumerate(zip(rows_all, cols_all)):
            if last.get(c, -1) >= r:
                keep[i] = False
            else:
                last[c] = r + h
        rows, cols = rows_all[keep], cols_all[keep]
        if len(rows) < 30:
            print("\n{}: {} events, too few".format(label, len(rows)))
            continue
        # Funding paid inside the hold, for a long.
        paid = np.zeros(len(rows))
        for i, (r, c) in enumerate(zip(rows, cols)):
            paid[i] = np.nansum(np.nan_to_num(panel.funding[r + 1:r + h + 1, c], nan=0.0))
        long_net = ex[rows, cols] - paid - 2 * cost
        groups = {}
        for v, r in zip(long_net, rows):
            groups.setdefault(r // h, []).append(v)
        means = np.array([np.mean(g) for g in groups.values()])
        se = means.std(ddof=1) / np.sqrt(len(means))
        yrs = years[rows]
        print("\n{}: {} events on {} coins; LONG net excess {:+.1f} ±{:.1f} (short is the negative), hit {:.0%}, "
              "funding paid {:+.1f}".format(label, len(rows), len(np.unique(cols)), long_net.mean(), se,
                                            (long_net > 0).mean(), paid.mean()))
        print("  " + "  ".join("{}:{:+.0f}(n{})".format(y[2:], long_net[yrs == y].mean(), int((yrs == y).sum()))
                              for y in sorted(set(yrs))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
