"""Open-interest factors on an hourly cross-section, scored as money.

    python backend\\analysis\\oi_factors.py --hold 4
    python backend\\analysis\\oi_factors.py --hold 8 --cost-bps 10 --shuffles 50

Why this is the first positioning test in the repo
--------------------------------------------------
Every cross-sectional factor scored so far came off a kline - price, volume,
taker share - or off funding. Open interest is the one public series that
says how much leverage is actually in a name, and `fetch_bybit_oi.py` now
holds it hourly from 2022 for ~80 Bybit perps beside Bybit's own klines and
funding. Same harness as everything else: `factor_panel.run_factor` with
non-overlapping holds, cross-sectionally demeaned labels net of funding, a
point-in-time universe, cost on turnover, and the mean of shuffled controls.

Signs, stated before the run (HIGH score = expected to outperform)
-------------------------------------------------------------------
  doi_1 / doi_4 / doi_24   OI growth over 1, 4, 24 hours: POSITIVE, new
                           money continues (the "OI confirms the move" folklore)
  doi_24_z                 the 24h growth against its own trailing 30 days: POSITIVE
  newlongs_1 / newlongs_24 sign(return) x OI growth, i.e. leverage arriving on
                           the side that is winning: NEGATIVE, it is the crowd
                           and it gets squeezed or bled
  oi_lvl                   OI notional over 30-day mean daily volume, i.e. how
                           crowded a name is relative to how easily it exits:
                           NEGATIVE
  oi_vol                    OI growth over 24h against volume over 24h (turnover
                           of positions): NEGATIVE, churn without new money
  rev_1 / rev_24           negated 1h / 24h return, as the known reference
                           (step 9ab: real, worth under a basis point)

Anything whose sign comes out opposite is reported as such, not flipped.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.factor_panel import Panel, block_bootstrap, run_factor, spearman  # noqa: E402
from analysis.fetch_bybit_oi import OUT as BYBIT_DIR, HOUR_MS, load_hourly  # noqa: E402
from analysis.panel_venue import get_json  # noqa: E402

FUNDING_START_MS = 1_640_995_200_000     # 2022-01-01


def fetch_funding(symbol: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    cursor = FUNDING_START_MS
    now = int(time.time() * 1000)
    for _ in range(120):
        if cursor >= now:
            break
        url = ("https://api.bybit.com/v5/market/funding/history?category=linear"
               "&symbol=" + symbol + "&limit=200&startTime=" + str(cursor)
               + "&endTime=" + str(min(now, cursor + 200 * 8 * HOUR_MS)))
        rows = get_json(url)["result"]["list"]
        if not rows:
            cursor += 200 * 8 * HOUR_MS
            continue
        newest = cursor
        for row in rows:
            ts = int(row["fundingRateTimestamp"])
            out[ts] = float(row["fundingRate"]) * 10_000.0
            newest = max(newest, ts)
        cursor = newest + 1
        time.sleep(0.1)
    return out


def ensure_funding(symbol: str, out_dir: Path = BYBIT_DIR) -> Dict[int, float]:
    path = out_dir / (symbol + ".funding.json")
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        return {int(k): float(v) for k, v in raw.items()}
    data = fetch_funding(symbol)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in data.items()}, handle)
    return data


def build(symbols: Optional[List[str]] = None, out_dir: Path = BYBIT_DIR):
    if symbols is None:
        symbols = sorted(p.stem for p in out_dir.glob("*.npz"))
    per = {s: load_hourly(s, out_dir) for s in symbols}
    lo = min(int(d["ts_open"][0]) for d in per.values())
    hi = max(int(d["ts_open"][-1]) for d in per.values())
    ts = np.arange(lo, hi + HOUR_MS, HOUR_MS, dtype=np.int64)
    n_t, n_s = len(ts), len(symbols)
    close = np.full((n_t, n_s), np.nan)
    volume = np.full((n_t, n_s), np.nan)
    oi = np.full((n_t, n_s), np.nan)
    funding = np.full((n_t, n_s), np.nan)
    with ThreadPoolExecutor(max_workers=6) as pool:
        fundings = list(pool.map(ensure_funding, symbols))
    for s, symbol in enumerate(symbols):
        d = per[symbol]
        idx = (d["ts_open"] - lo) // HOUR_MS
        close[idx, s] = d["close"]
        volume[idx, s] = d["quote_volume"]
        oi[idx, s] = d["oi"] * d["close"]
        # Funding stamped T paid for the interval ending at T: it belongs to
        # the hourly row that CLOSES at T, i.e. the row opening at T - 1h.
        for t, bps in fundings[s].items():
            i = (t - HOUR_MS - lo) // HOUR_MS
            if 0 <= i < n_t:
                funding[i, s] = bps
    dates = [time.strftime("%Y-%m-%dT%H", time.gmtime(t / 1000 + 3600)) for t in ts]
    return ts, dates, symbols, close, volume, oi, funding


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


def trailing_std(values: np.ndarray, window: int) -> np.ndarray:
    mean = trailing_mean(values, window)
    sq = trailing_mean(values * values, window)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.maximum(sq - mean * mean, 0.0))


def log_change(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = values[lag:] / values[:-lag]
        out[lag:] = np.log(np.where(ratio > 0, ratio, np.nan))
    return out


def features(close, volume, oi):
    ret1 = log_change(close, 1) * 1e4
    ret24 = log_change(close, 24) * 1e4
    doi1, doi4, doi24 = log_change(oi, 1), log_change(oi, 4), log_change(oi, 24)
    mean30, sd30 = trailing_mean(doi24, 720), trailing_std(doi24, 720)
    with np.errstate(invalid="ignore", divide="ignore"):
        doi24_z = (doi24 - mean30) / sd30
        daily_vol = trailing_mean(volume, 24) * 24.0
        vol30 = trailing_mean(daily_vol, 720)
        oi_lvl = np.log(oi / vol30)
        vol24 = trailing_mean(volume, 24) * 24.0
        oi_vol = np.abs(oi - np.roll(oi, 24, axis=0)) / vol24
    oi_vol[:24] = np.nan
    return {
        "doi_1": doi1, "doi_4": doi4, "doi_24": doi24, "doi_24_z": doi24_z,
        "newlongs_1": -np.sign(ret1) * doi1, "newlongs_24": -np.sign(ret24) * doi24,
        "oi_lvl": -oi_lvl, "oi_vol": -oi_vol,
        "rev_1": -ret1, "rev_24": -ret24,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hold", type=int, default=4, help="hours")
    parser.add_argument("--top-frac", type=float, default=0.3)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--min-volume", type=float, default=5e6, help="median daily $ volume, 30d")
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--factors", default=None)
    parser.add_argument("--start", type=int, default=800, help="first row scored")
    args = parser.parse_args(argv)
    cost = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps

    ts, dates, symbols, close, volume, oi, funding = build()
    n_t, n_s = close.shape
    daily_vol = trailing_mean(volume, 24) * 24.0
    med30 = np.full(close.shape, np.nan)
    for t in range(720, n_t, 24):                    # once a day, held for the day
        with np.errstate(invalid="ignore"):
            med30[t:t + 24] = np.nanmedian(daily_vol[t - 720:t:24], axis=0)
    complete = np.isfinite(close)
    eligible = complete & np.isfinite(oi) & (med30 >= args.min_volume)
    rv = np.full(close.shape, np.nan)
    panel = Panel(dates, symbols, close, volume, funding, rv, close, close, rv, complete)
    feats = features(close, volume, oi)
    names = args.factors.split(",") if args.factors else list(feats)
    periods_per_year = 8760.0 / args.hold
    print("Bybit hourly, {} symbols, {} hours {}..{}, hold {}h, top/bottom {:.0%}, cost {:.0f} bps, "
          "eligible per hour median {:.0f}".format(
              n_s, n_t, dates[0], dates[-1], args.hold, args.top_frac, cost,
              np.median(eligible[args.start:].sum(axis=1))))
    print("{:<13} {:>7} {:>16} {:>7} {:>6} {:>6} {:>7} {:>7} {:>7} {:>6}  {}".format(
        "factor", "net", "95% block", "Sharpe", "turn", "IC", "price", "fund", "ctrl", "pct", "by year (net)"))
    years = np.array([d[:4] for d in dates])
    for name in names:
        score = feats[name]
        real = run_factor(panel, score, eligible, hold_days=args.hold, top_frac=args.top_frac,
                          cost_bps=cost, start=args.start)
        if len(real.periods) < 10:
            print(name + ": too few periods")
            continue
        net = real.net
        controls = []
        for seed in range(args.shuffles):
            ctrl = run_factor(panel, score, eligible, hold_days=args.hold, top_frac=args.top_frac,
                              cost_bps=cost, start=args.start, shuffle_seed=seed)
            controls.append(float(ctrl.net.mean()))
        controls = np.array(controls)
        lo, hi = block_bootstrap(net, block=max(4, 96 // args.hold))
        sharpe = net.mean() / net.std(ddof=1) * math.sqrt(periods_per_year) if net.std(ddof=1) > 0 else float("nan")
        turnover = np.mean([p.turnover for p in real.periods])
        price = np.mean([p.price_bps for p in real.periods])
        fund = np.mean([p.funding_bps for p in real.periods])
        entry_years = years[[p.entry for p in real.periods]]
        by_year = " ".join("{}:{:+.1f}".format(y[2:], net[entry_years == y].mean())
                           for y in sorted(set(entry_years)))
        print("{:<13} {:>+7.2f} [{:>+6.1f},{:>+6.1f}] {:>7.2f} {:>6.2f} {:>+6.3f} {:>+7.2f} {:>+7.2f} {:>+7.2f} {:>5.0%}  {}".format(
            name, net.mean(), lo, hi, sharpe, turnover, real.mean_ic, price, fund,
            controls.mean(), (net.mean() > controls).mean(), by_year))
    return 0


if __name__ == "__main__":
    sys.exit(main())
