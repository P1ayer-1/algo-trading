"""15-minute bars for every panel symbol, folded once from the 1m archive.

    python backend\analysis\panel_intraday.py                 # every symbol in the daily panel
    python backend\analysis\panel_intraday.py --symbols BTCUSDT,ETHUSDT

Why this exists
---------------
`panel_daily.py` folds the 5.7 GB 1-minute archive into one row per day, and
every result on it (steps 9q-9z) lives at holds of 3 to 30 days. Nothing in
the repo can ask a question BETWEEN a minute and a day across the whole
cross-section: what happens around a funding settlement, whether the carry
signal pays at an 8-hour hold, whether a 4-hour move across a hundred coins
reverts or continues. Those questions need a bar that is coarse enough to
hold five years of a hundred symbols in memory and fine enough to resolve a
settlement hour. Fifteen minutes is that bar: ~175k rows a symbol, 8 MB as
float64, and every coarser bar (1h, 4h, 8h, 1d) derives from it exactly
because 15 divides them all.

What a row is
-------------
A bar ENDS at `ts_end` (a multiple of 15 minutes, UTC) and contains the 1m
bars whose `close_time` lies in `(ts_end - 15m, ts_end]` - the same anchor
`panel_daily.py` uses, so a 15m bar ending 00:00 belongs to the day that just
closed, and an 8h bar ending 08:00 contains exactly the interval Binance's
08:00 settlement pays for. `minutes` counts the 1m bars actually present;
`ss_logret` is the sum of squared 1m log returns taken within the bar only.

Monthly and daily archives overlap at month edges by design, so minutes are
de-duplicated on `close_time` before folding.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "data" / "cache"
OUT = REPO_ROOT / "data" / "panel" / "intraday"
DAILY_PANEL = REPO_ROOT / "data" / "panel" / "daily.csv"

BAR_MS = 15 * 60_000
# open, high, low, close, close_time, quote_volume, count, taker_buy_quote_volume
USECOLS = (1, 2, 3, 4, 6, 7, 8, 10)
FIELDS = ("open", "high", "low", "close", "quote_volume", "trades",
          "taker_buy_quote", "minutes", "ss_logret")


def kline_zips(symbol: str, cache: Path = CACHE) -> List[Path]:
    directory = cache / symbol
    if not directory.is_dir():
        return []
    return sorted(directory.glob(symbol + "-1m-*.zip"))


def _read_zip(path: Path) -> Optional[np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            return None
        data = archive.read(names[0])
    if not data:
        return None
    skip = 1 if data[:1].isalpha() else 0
    try:
        arr = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=skip,
                         usecols=USECOLS, ndmin=2)
    except ValueError:
        return None
    return arr if arr.size else None


def fold_symbol(symbol: str, cache: Path = CACHE, out: Path = OUT) -> Dict[str, object]:
    """Fold one symbol's 1m archive to 15m bars and write `<symbol>.npz`."""
    started = time.time()
    parts = [a for a in (_read_zip(p) for p in kline_zips(symbol, cache)) if a is not None]
    if not parts:
        return {"symbol": symbol, "bars": 0, "seconds": 0.0}
    raw = np.concatenate(parts, axis=0)
    ts = raw[:, 4].astype(np.int64)
    ts = np.where(ts > 10_000_000_000_000, ts // 1000, ts)     # microsecond archives
    good = (raw[:, 3] > 0) & (raw[:, 1] > 0) & (raw[:, 2] > 0)
    raw, ts = raw[good], ts[good]
    ts, first = np.unique(ts, return_index=True)               # sorted + de-duplicated
    raw = raw[first]

    key = (ts // BAR_MS + 1) * BAR_MS
    bar_end, start = np.unique(key, return_index=True)
    close = raw[:, 3]
    step = np.zeros(len(close))
    step[1:] = np.log(close[1:] / close[:-1])
    step[start] = 0.0                                          # no step across a bar edge
    n = len(bar_end)
    end = np.append(start[1:], len(ts))

    arrays = {
        "ts_end": bar_end.astype(np.int64),
        "open": raw[start, 0],
        "high": np.maximum.reduceat(raw[:, 1], start),
        "low": np.minimum.reduceat(raw[:, 2], start),
        "close": close[end - 1],
        "quote_volume": np.add.reduceat(raw[:, 5], start),
        "trades": np.add.reduceat(raw[:, 6], start),
        "taker_buy_quote": np.add.reduceat(raw[:, 7], start),
        "minutes": (end - start).astype(np.int16),
        "ss_logret": np.add.reduceat(step * step, start),
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / (symbol + ".npz"), **arrays)
    return {"symbol": symbol, "bars": int(n), "minutes": int(len(ts)),
            "first": int(bar_end[0]), "last": int(bar_end[-1]),
            "seconds": round(time.time() - started, 1)}


def panel_symbols(path: Path = DAILY_PANEL) -> List[str]:
    symbols: Dict[str, None] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbols[row["symbol"]] = None
    return sorted(symbols)


def load_bars(symbol: str, out: Path = OUT) -> Dict[str, np.ndarray]:
    with np.load(out / (symbol + ".npz")) as z:
        return {k: z[k] for k in z.files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbols", help="comma-separated; default: every symbol in the daily panel")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="rebuild even if the npz exists")
    args = parser.parse_args(argv)

    symbols = args.symbols.split(",") if args.symbols else panel_symbols()
    todo = [s for s in symbols if args.force or not (OUT / (s + ".npz")).exists()]
    print(f"{len(symbols)} symbols, {len(todo)} to fold, {args.workers} workers", flush=True)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fold_symbol, s): s for s in todo}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(f"  {report['symbol']:16s} {report.get('bars', 0):7d} bars  "
                  f"{report.get('seconds', 0):6.1f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(sorted(reports, key=lambda r: r["symbol"]), handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
