"""Do Hyperliquid's liquidation levels say where price goes next?

    python backend\\analysis\\liquidation_signal.py
    python backend\\analysis\\liquidation_signal.py --coins BTC,ETH --horizon-min 15,60,240

Two hypotheses, stated before the first run
-------------------------------------------
`record_hyperliquid.py` writes a liquidation map every ~60s per coin: the
notional of tracked positions whose exchange-reported liquidation price sits
within 1%, 2%, 5% and 10% of mark, on each side, plus coverage of open
interest. Either of two stories would make that tradeable inside eight
hours:

* **Magnet.** Price is drawn toward the side with more liquidatable notional
  close to it - liquidity hunts, or simply the fact that a cascade there
  is a large forced flow waiting to happen. Prediction: the forward return
  correlates POSITIVELY with (short notional within 1% above) minus (long
  notional within 1% below), each as a share of tracked size.
* **Overshoot.** Once price crosses into a dense band the forced flow runs
  through the book and the move overshoots, so the return AFTER a band is
  swept reverses. Prediction: conditional on the last 15 minutes having
  crossed a band holding more than `--sweep-frac` of one side's tracked
  size, the next `--horizon-min` return has the opposite sign to the sweep.

Both are scored on the map's own mark series, in bps, per coin and pooled,
with Newman-style clustered errors by hour (adjacent snapshots share their
forward window). Coverage is printed beside every number: a map covering 3%
of open interest is a sample, and the study says so rather than pretending.

This tool was written on 2026-09-12 against ~14 hours of BTC and ETH, which
is a smoke test and not evidence. Its job is to be run again in a few weeks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HL_DIR = REPO_ROOT / "data" / "hyperliquid"


def load_maps(coin: str, root: Path = HL_DIR) -> List[dict]:
    out = []
    for path in sorted((root / coin).glob("liquidation-levels-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    out.sort(key=lambda m: m["as_of_ms"])
    return out


def series(maps: List[dict]) -> Dict[str, np.ndarray]:
    t = np.array([m["as_of_ms"] for m in maps], dtype=np.int64)
    mark = np.array([m["mark_px"] for m in maps], dtype=float)
    oi = np.array([m["open_interest"] for m in maps], dtype=float)
    cov_l = np.array([m["coverage"]["long"] for m in maps], dtype=float)
    cov_s = np.array([m["coverage"]["short"] for m in maps], dtype=float)

    def within(side: str, key: str) -> np.ndarray:
        return np.array([m[side]["within_notional"].get(key, 0.0) for m in maps], dtype=float)

    tracked_l = np.array([m["longs"]["size"] for m in maps]) * mark
    tracked_s = np.array([m["shorts"]["size"] for m in maps]) * mark
    return {
        "t": t, "mark": mark, "oi_notional": oi * mark,
        "cov_long": cov_l, "cov_short": cov_s,
        "long_1": within("longs", "1%"), "long_2": within("longs", "2%"),
        "short_1": within("shorts", "1%"), "short_2": within("shorts", "2%"),
        "tracked_long": tracked_l, "tracked_short": tracked_s,
    }


def forward_return(t: np.ndarray, mark: np.ndarray, horizon_ms: int) -> np.ndarray:
    """Return from each snapshot to the first snapshot at least `horizon_ms` later."""
    out = np.full(len(t), np.nan)
    j = np.searchsorted(t, t + horizon_ms, side="left")
    ok = j < len(t)
    out[ok] = np.log(mark[j[ok]] / mark[ok]) * 10_000.0
    return out


def clustered_corr(x: np.ndarray, y: np.ndarray, t: np.ndarray, cluster_ms: int):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, t = x[ok], y[ok], t[ok]
    if len(x) < 30:
        return float("nan"), float("nan"), 0
    corr = float(np.corrcoef(x, y)[0, 1])
    # Slope t-stat with cluster-robust standard error by time bucket.
    xc = x - x.mean()
    beta = float((xc * y).sum() / (xc ** 2).sum())
    resid = y - y.mean() - beta * xc
    groups = t // cluster_ms
    scores = {}
    for g, xi, ri in zip(groups, xc, resid):
        scores[g] = scores.get(g, 0.0) + xi * ri
    v = sum(s * s for s in scores.values())
    se = np.sqrt(v) / (xc ** 2).sum()
    return corr, beta / se if se > 0 else float("nan"), len(scores)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--coins", default=None, help="default: every coin directory with maps")
    parser.add_argument("--horizon-min", default="15,60,240")
    parser.add_argument("--sweep-frac", type=float, default=0.05,
                        help="share of a side's tracked notional inside the swept band")
    args = parser.parse_args(argv)
    coins = args.coins.split(",") if args.coins else sorted(
        p.name for p in HL_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and list(p.glob("liquidation-levels-*.jsonl")))
    horizons = [int(h) for h in args.horizon_min.split(",")]

    pooled = {h: ([], [], []) for h in horizons}
    for coin in coins:
        maps = load_maps(coin)
        if len(maps) < 60:
            print(coin + ": " + str(len(maps)) + " snapshots, skipped")
            continue
        s = series(maps)
        span_h = (s["t"][-1] - s["t"][0]) / 3_600_000
        with np.errstate(divide="ignore", invalid="ignore"):
            imbalance = (s["short_1"] / s["tracked_short"]) - (s["long_1"] / s["tracked_long"])
        print("\n{}: {} snapshots over {:.1f}h, coverage long {:.1%} / short {:.1%} (median), "
              "within 1%: long ${:,.0f} short ${:,.0f} (median)".format(
                  coin, len(maps), span_h, np.median(s["cov_long"]), np.median(s["cov_short"]),
                  np.median(s["long_1"]), np.median(s["short_1"])))
        print("  magnet: corr(imbalance within 1%, forward return)")
        for h in horizons:
            fwd = forward_return(s["t"], s["mark"], h * 60_000)
            corr, tstat, n = clustered_corr(imbalance, fwd, s["t"], max(h, 60) * 60_000)
            print("    {:>4}m  corr {:+.3f}  t {:+.2f}  clusters {}".format(h, corr, tstat, n))
            pooled[h][0].extend(imbalance.tolist())
            pooled[h][1].extend(fwd.tolist())
            pooled[h][2].extend((s["t"] // 1000 + hash(coin) % 7).tolist())
        # Overshoot: a 15-minute move that crossed a band holding >= sweep_frac of a side.
        back = np.full(len(s["t"]), np.nan)
        j = np.searchsorted(s["t"], s["t"] - 15 * 60_000, side="right") - 1
        ok = j >= 0
        back[ok] = np.log(s["mark"][ok] / s["mark"][j[ok]]) * 10_000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            long_share_1 = s["long_1"] / s["tracked_long"]
            short_share_1 = s["short_1"] / s["tracked_short"]
        # A down move of >= 1% through a band of longs; an up move through shorts.
        prev = np.clip(j, 0, None)
        swept_down = ok & (back <= -100) & (long_share_1[prev] >= args.sweep_frac)
        swept_up = ok & (back >= 100) & (short_share_1[prev] >= args.sweep_frac)
        for label, mask, sign in (("down through longs", swept_down, -1.0), ("up through shorts", swept_up, 1.0)):
            n = int(mask.sum())
            if n == 0:
                print("  overshoot, {}: no events".format(label))
                continue
            line = "  overshoot, {}: {} snapshots; reversal after ".format(label, n)
            for h in horizons:
                fwd = forward_return(s["t"], s["mark"], h * 60_000)
                v = -sign * fwd[mask]
                v = v[np.isfinite(v)]
                line += "{}m {:+.1f}  ".format(h, v.mean() if len(v) else float("nan"))
            print(line)

    print("\npooled magnet correlation across coins")
    for h in horizons:
        x, y, t = (np.array(v, dtype=float) for v in pooled[h])
        corr, tstat, n = clustered_corr(x, y, t.astype(np.int64), max(h, 60) * 60)
        print("  {:>4}m  corr {:+.3f}  t {:+.2f}  clusters {}".format(h, corr, tstat, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
