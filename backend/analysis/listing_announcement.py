"""The Binance listing announcement, read at 5 minutes on the venue where the coin already trades.

    python backend\\analysis\\listing_announcement.py
    python backend\\analysis\\listing_announcement.py --kind futures --since 2024

Why this is a different event from `listing_day.py`
---------------------------------------------------
`listing_day.py` measured the first hours of a Binance perpetual on Binance,
where the position could only be taken once the contract existed. The
announcement is earlier and is the event the folklore is actually about:
"Binance will list X" moves X on every venue that already trades it, within
seconds, by tens of percent for a small coin. What a strategy can capture is
not the first second - it is whatever is left from the close of the
five-minute bar the announcement lands in to some hours later, on a venue
with the coin already listed. Bybit is that venue here: its 5m klines are
fetched on demand for a 3h-before / 9h-after window around every event and
cached, with BTCUSDT over the same window as the market term.

Source: Binance's own announcement catalogue (id 48, "New Cryptocurrency
Listing"), fetched once and cached; each article carries a millisecond
release stamp. Tickers are read from the title. A spot listing ("Binance
Will List X (X)") and a futures launch ("Binance Futures Will Launch
USDS-M XUSDT Perpetual") are separate events and are reported separately.

Per event: the return of the announcement bar itself (not tradeable), the
return of the next bar (a fast entry at bar close), then from the
announcement bar's close to +15m, +1h, +4h and +8h, in excess of BTC and
net of a taker round trip; by year, with a median beside every mean because
these are lottery-shaped. A coin is only counted if Bybit already had two
hours of bars before the release.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from analysis.panel_venue import get_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "data" / "cache" / "binance_announcements.json"
KLINES = REPO_ROOT / "data" / "cache" / "announcement_klines"
URL = ("https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
       "?type=1&catalogId=48&pageSize=20&pageNo=")
BAR_MS = 5 * 60_000
BEFORE, AFTER = 36, 108                         # bars: 3h before, 9h after
BAR_MIN = 5

SPOT = re.compile(r"Binance Will List ")
FUTURES = re.compile(r"Binance Futures Will Launch ", re.I)
SPOT_TICKER = re.compile(r"\(([A-Z0-9]+)\)")
FUTURES_TICKER = re.compile(r"\b([A-Z0-9]+)USDT\b")


def classify(title: str):
    """`(kind, [tickers])` for a listing title, `(None, [])` otherwise.

    A title often names several coins ("Will Launch AUSDT and BUSDT Perpetual
    Contracts", "Will List A (A), B (B)"), and each is its own event at the
    same instant. Binance Alpha listings are a different, smaller event and
    are excluded by name.
    """
    if "Alpha" in title:
        return None, []
    if FUTURES.search(title):
        return "futures", [t.upper() for t in FUTURES_TICKER.findall(title) if t.upper() != "USD"]
    if SPOT.search(title):
        return "spot", [t.upper() for t in SPOT_TICKER.findall(title)]
    return None, []


def fetch_announcements() -> List[dict]:
    if CACHE.exists():
        with CACHE.open(encoding="utf-8") as handle:
            return json.load(handle)
    articles: List[dict] = []
    page = 1
    while True:
        data = get_json(URL + str(page))["data"]
        catalog = data["catalogs"][0]
        rows = catalog.get("articles") or []
        if not rows:
            break
        articles.extend({"title": r["title"], "release_ms": int(r["releaseDate"])} for r in rows)
        total = int(catalog.get("total") or 0)
        if len(articles) >= total:
            break
        page += 1
        time.sleep(0.3)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", encoding="utf-8") as handle:
        json.dump(articles, handle)
    return articles


def bybit_5m(symbol: str, start_ms: int, end_ms: int) -> Dict[int, float]:
    """Close per bar opening in [start, end]; empty if Bybit has no such symbol."""
    path = KLINES / (symbol + "-" + str(BAR_MIN) + "m-" + str(start_ms) + ".json")
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return {int(k): float(v) for k, v in json.load(handle).items()}
    url = ("https://api.bybit.com/v5/market/kline?category=linear&symbol=" + symbol
           + "&interval=" + str(BAR_MIN) + "&limit=200&start=" + str(start_ms) + "&end=" + str(end_ms))
    try:
        rows = get_json(url)["result"]["list"]
    except Exception:                                 # noqa: BLE001
        rows = []
    out = {int(r[0]): float(r[4]) for r in rows}
    KLINES.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in out.items()}, handle)
    return out


def event_path(kind: str, ticker: str, release_ms: int) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    bar0 = (release_ms // BAR_MS) * BAR_MS
    start, end = bar0 - BEFORE * BAR_MS, bar0 + AFTER * BAR_MS
    for symbol in (ticker + "USDT", "1000" + ticker + "USDT", "10000" + ticker + "USDT"):
        closes = bybit_5m(symbol, start, end)
        if len(closes) < 24 or (bar0 - 24 * BAR_MS) not in closes or bar0 not in closes:
            continue
        btc = bybit_5m("BTCUSDT", start, end)
        grid = np.arange(start, end + BAR_MS, BAR_MS)
        c = np.array([closes.get(int(t), np.nan) for t in grid])
        b = np.array([btc.get(int(t), np.nan) for t in grid])
        return symbol, c, b
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kind", choices=("spot", "futures", "both"), default="both")
    parser.add_argument("--since", type=int, default=2022)
    parser.add_argument("--cost-bps", type=float, default=None, help="per leg")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--events", action="store_true", help="print every event")
    parser.add_argument("--spot-hold", type=int, default=60, help="minutes, --minutes mode")
    parser.add_argument("--futures-hold", type=int, default=15, help="minutes, --minutes mode")
    parser.add_argument("--minutes", action="store_true",
                        help="1-minute bars, 30 before and 169 after, to see how fast the move is")
    args = parser.parse_args(argv)
    global BAR_MS, BAR_MIN, BEFORE, AFTER
    if args.minutes:
        BAR_MS, BAR_MIN, BEFORE, AFTER = 60_000, 1, 30, 169
    cost = float(config.TAKER_FEE_BPS) if args.cost_bps is None else args.cost_bps

    articles = fetch_announcements()
    wanted = []
    for art in articles:
        kind, tickers = classify(art["title"])
        if kind is None or (args.kind != "both" and kind != args.kind):
            continue
        if time.gmtime(art["release_ms"] / 1000).tm_year < args.since:
            continue
        for ticker in tickers:
            wanted.append((kind, ticker, art["release_ms"]))
    print("{} announcements matched since {}; fetching Bybit 5m windows".format(len(wanted), args.since), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(pool.map(lambda w: event_path(*w), wanted))
    events = [(w, p) for w, p in zip(wanted, paths) if p is not None]
    print("{} with a Bybit perp already trading two hours before the release".format(len(events)))

    i0 = BEFORE                                       # index of the announcement bar's OPEN
    horizons = ({"+1m": 1, "+2m": 2, "+5m": 5, "+15m": 15, "+60m": 60, "+120m": 120}
                if args.minutes else {"+15m": 3, "+1h": 12, "+4h": 48, "+8h": 96})

    def ret(c, a, b):
        return np.log(c[b] / c[a]) * 1e4

    for kind in ("spot", "futures"):
        ev = [(w, p) for w, p in events if w[0] == kind]
        if len(ev) < 5:
            continue
        rows = []
        for (k, ticker, release), (symbol, c, b) in ev:
            if not (np.isfinite(c[i0 - 1]) and np.isfinite(c[i0]) and np.isfinite(c[i0 + 1])):
                continue
            pre = ret(c, i0 - 25, i0 - 1) - ret(b, i0 - 25, i0 - 1) if np.isfinite(c[i0 - 25]) else np.nan
            ann = ret(c, i0 - 1, i0)                  # the bar containing the release
            nxt = ret(c, i0, i0 + 1)                  # the bar after
            out = {}
            for label, hbar in horizons.items():
                j = i0 + hbar
                if j < len(c) and np.isfinite(c[j]) and np.isfinite(b[j]) and np.isfinite(b[i0]):
                    out[label] = (ret(c, i0, j) - ret(b, i0, j)) - 2 * cost
                    out[label + "_d1"] = (ret(c, i0 + 1, j) - ret(b, i0 + 1, j)) - 2 * cost
                else:
                    out[label] = out[label + "_d1"] = np.nan
            rows.append((release, symbol, pre, ann, nxt, out))
        if not rows:
            continue
        ann = np.array([r[3] for r in rows])
        nxt = np.array([r[4] for r in rows])
        pre = np.array([r[2] for r in rows])
        print("\n{} listings, n {}: pre-25-bar excess {:+.0f}; announcement 5m bar {:+.0f} (median {:+.0f}); "
              "next 5m bar {:+.0f} (median {:+.0f})".format(kind, len(rows), np.nanmean(pre), ann.mean(),
                                                            np.median(ann), nxt.mean(), np.median(nxt)))
        print("  long from the announcement bar's close, excess of BTC, net of {:.0f}/leg:".format(cost))
        print("  {:>6} {:>9} {:>7} {:>9} {:>5} | {:>9} {:>9}   (entered one bar later)".format(
            "hold", "mean", "se", "median", "hit", "mean", "median"))
        for label in horizons:
            v = np.array([r[5][label] for r in rows])
            d = np.array([r[5][label + "_d1"] for r in rows])
            v, d = v[np.isfinite(v)], d[np.isfinite(d)]
            print("  {:>6} {:>+9.1f} {:>7.1f} {:>+9.1f} {:>4.0%} | {:>+9.1f} {:>+9.1f}".format(
                label, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), np.median(v), (v > 0).mean(),
                d.mean(), np.median(d)))
        years = np.array([time.gmtime(r[0] / 1000).tm_year for r in rows])
        for y in sorted(set(years)):
            pick = years == y
            k1, k4 = (("+5m", "+60m") if args.minutes else ("+1h", "+4h"))
            v1 = np.array([r[5][k1] for r in rows])[pick]
            v4 = np.array([r[5][k4] for r in rows])[pick]
            print("  {}  n {:>3}  ann bar {:+.0f}  {} {:+.1f} (median {:+.1f})  {} {:+.1f} (median {:+.1f})".format(
                y, int(pick.sum()), ann[pick].mean(), k1, np.nanmean(v1), np.nanmedian(v1),
                k4, np.nanmean(v4), np.nanmedian(v4)))
        if args.events:
            keys = list(horizons)
            print("  events (excess of BTC, net):  " + "  ".join(keys))
            for r in sorted(rows, key=lambda x: x[0]):
                print("    {}  {:<14} ann {:+6.0f}  ".format(
                    time.strftime("%Y-%m-%d %H:%M", time.gmtime(r[0] / 1000)), r[1], r[3])
                    + "  ".join("{:+7.0f}".format(r[5][k]) for k in keys))
    # The two legs as one strategy on CAPITAL: one unit per event, long a spot
    # listing for --spot-hold bars, short a futures launch for --futures-hold
    # bars, from the close of the announcement bar. Daily P&L with zeros on
    # quiet days is what a Sharpe can honestly be computed on.
    if args.minutes:
        spot_h, fut_h = args.spot_hold, args.futures_hold
        pnl_by_day = {}
        per_event = []
        for (kind, ticker, release), (symbol, c, b) in events:
            hbar = spot_h if kind == "spot" else fut_h
            j = i0 + hbar
            if j >= len(c) or not (np.isfinite(c[i0]) and np.isfinite(c[j]) and np.isfinite(b[i0]) and np.isfinite(b[j])):
                continue
            ex = (ret(c, i0, j) - ret(b, i0, j))
            v = (ex if kind == "spot" else -ex) - 2 * cost
            day = time.strftime("%Y-%m-%d", time.gmtime(release / 1000))
            pnl_by_day[day] = pnl_by_day.get(day, 0.0) + v
            per_event.append((release, kind, symbol, v))
        if per_event:
            first = min(r for r, _, _, _ in per_event) // 86_400_000
            last = max(r for r, _, _, _ in per_event) // 86_400_000
            days = [time.strftime("%Y-%m-%d", time.gmtime(d * 86_400)) for d in range(first, last + 1)]
            daily = np.array([pnl_by_day.get(d, 0.0) for d in days])
            years = np.array([d[:4] for d in days])
            sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(365) if daily.std() > 0 else float("nan")
            vals = np.array([v for _, _, _, v in per_event])
            print("\nSTRATEGY on capital: long spot listings {} min, short futures launches {} min, one unit per event, "
                  "cost {:.0f}/leg".format(spot_h, fut_h, cost))
            print("  {} events over {} days: mean {:+.0f} bps per event, worst {:+.0f}, best {:+.0f}, hit {:.0%}".format(
                len(vals), len(days), vals.mean(), vals.min(), vals.max(), (vals > 0).mean()))
            print("  daily series: {:+.1f} bps/day = {:+.0f} bps/yr on capital, Sharpe {:.2f}, worst day {:+.0f}".format(
                daily.mean(), daily.mean() * 365, sharpe, daily.min()))
            for y in sorted(set(years)):
                pick = years == y
                n_ev = sum(1 for r, _, _, _ in per_event if time.gmtime(r / 1000).tm_year == int(y))
                d = daily[pick]
                print("  {}  events {:>3}  total {:+7.0f} bps  Sharpe {:5.2f}".format(
                    y, n_ev, d.sum(), d.mean() / d.std(ddof=1) * np.sqrt(365) if d.std() > 0 else float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
