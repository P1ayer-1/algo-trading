"""Hourly open interest and klines from Bybit, the one venue that serves OI history.

    python backend\\analysis\\fetch_bybit_oi.py --top 80
    python backend\\analysis\\fetch_bybit_oi.py --symbols DOGEUSDT,SUIUSDT

Why Bybit, and why this is a new data class for the repo
--------------------------------------------------------
Every intraday result so far (steps 6-9ab) was scored on price, volume,
taker share and funding, because those are what a kline carries. Positioning
- how much open interest exists and how it changes - was never in any panel,
because Binance keeps only 30 days of `openInterestHist` and BloFin publishes
none. Bybit's `/v5/market/open-interest` answers with hourly readings back to
2022 (checked 2026-09-12: DOGEUSDT returns rows at 1660000000000), and its
kline endpoint covers the same span, so an OI-conditioned cross-section can
be measured on ~4 years x ~80 names rather than on a month.

The OI is Bybit's own and so are the klines here, deliberately: an OI change
on one venue against a price on another would put every timestamp join in
question. Everything is UTC-hour stamped by Bybit; the OI row at `T` is the
reading taken AT `T` (end of the hour), and the kline row at `T` is the hour
that OPENS at `T`. `load_hourly` aligns them so that a row's OI is known at
the row's close and nothing later.

Output: `data/panel/bybit_1h/<SYMBOL>.npz` with `ts_open`, `open`, `high`,
`low`, `close`, `quote_volume`, `oi` (contracts, base units) - one row per
hour, NaN where a series has no reading. Re-running skips symbols already
on disk unless `--force`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.panel_venue import _bybit_universe, get_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = REPO_ROOT / "data" / "panel" / "bybit_1h"
HOUR_MS = 3_600_000
START_MS = 1_640_995_200_000            # 2022-01-01


def fetch_klines(symbol: str, start_ms: int = START_MS) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    end = int(time.time() * 1000)
    for _ in range(60):
        url = ("https://api.bybit.com/v5/market/kline?category=linear&symbol="
               + symbol + "&interval=60&limit=1000&end=" + str(end))
        rows = get_json(url)["result"]["list"]
        if not rows:
            break
        for row in rows:
            ts = int(row[0])
            out[ts] = [float(row[1]), float(row[2]), float(row[3]),
                       float(row[4]), float(row[6])]
        oldest = min(int(row[0]) for row in rows)
        if oldest <= start_ms or len(rows) < 1000:
            break
        end = oldest - 1
        time.sleep(0.12)
    return out


def fetch_oi(symbol: str, start_ms: int = START_MS) -> Dict[int, float]:
    """Paged with the cursor Bybit hands back; 200 rows a page, newest first."""
    out: Dict[int, float] = {}
    cursor: Optional[str] = None
    now = int(time.time() * 1000)
    for _ in range(400):
        url = ("https://api.bybit.com/v5/market/open-interest?category=linear&symbol="
               + symbol + "&intervalTime=1h&limit=200&startTime=" + str(start_ms)
               + "&endTime=" + str(now))
        if cursor:
            url += "&cursor=" + cursor
        result = get_json(url)["result"]
        rows = result.get("list") or []
        if not rows:
            break
        for row in rows:
            out[int(row["timestamp"])] = float(row["openInterest"])
        cursor = result.get("nextPageCursor") or None
        if not cursor or len(rows) < 200:
            break
        time.sleep(0.12)
    return out


def fetch_symbol(symbol: str, out_dir: Path = OUT) -> Path:
    klines = fetch_klines(symbol)
    oi = fetch_oi(symbol)
    if not klines:
        raise RuntimeError(symbol + ": no klines")
    ts = np.array(sorted(klines), dtype=np.int64)
    arr = np.array([klines[t] for t in ts])
    # OI stamped T is the reading at the END of the hour opening at T - 1h;
    # store it on the row whose close is T, i.e. the kline opening at T - 1h.
    oi_col = np.full(len(ts), np.nan)
    index = {int(t): i for i, t in enumerate(ts)}
    for t, value in oi.items():
        i = index.get(t - HOUR_MS)
        if i is not None:
            oi_col[i] = value
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (symbol + ".npz")
    np.savez_compressed(path, ts_open=ts, open=arr[:, 0], high=arr[:, 1],
                        low=arr[:, 2], close=arr[:, 3], quote_volume=arr[:, 4],
                        oi=oi_col)
    return path


def load_hourly(symbol: str, out_dir: Path = OUT) -> Dict[str, np.ndarray]:
    with np.load(out_dir / (symbol + ".npz")) as z:
        return {k: z[k] for k in z.files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--top", type=int, default=80, help="by 24h turnover")
    parser.add_argument("--symbols", help="comma-separated Bybit symbols")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        universe = sorted(_bybit_universe(), key=lambda l: -l.volume_usd_24h)
        symbols = [l.symbol for l in universe[:args.top]]
    print(str(len(symbols)) + " symbols")
    todo = [s for s in symbols if args.force or not (OUT / (s + ".npz")).exists()]

    def one(symbol: str) -> str:
        started = time.time()
        try:
            fetch_symbol(symbol)
        except Exception as exc:                        # noqa: BLE001
            return "{:<14} FAILED {}".format(symbol, exc)
        data = load_hourly(symbol)
        return "{:<14} {:>6} hours, {:>6} with OI, {:.0f}s".format(
            symbol, len(data["ts_open"]), int(np.isfinite(data["oi"]).sum()),
            time.time() - started)

    # ~250 requests a symbol at ~0.5s each is two hours serially; Bybit's
    # public rate limit is far above six concurrent readers (2026-09-12).
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, line in enumerate(pool.map(one, todo), 1):
            print("{:>3}/{} {}".format(n, len(todo), line), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
