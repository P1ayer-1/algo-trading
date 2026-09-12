"""Binance's premium index at 15 minutes: the funding signal before it settles.

    python backend\\analysis\\fetch_premium_index.py
    python backend\\analysis\\fetch_premium_index.py --symbols DOGEUSDT,SUIUSDT --interval 15m

Why
---
A settlement's funding rate is the time-average of the perp's premium over
the index across the eight hours before it, so at the instant it prints it
is stale by four hours on average, and at any other instant the rate that
WILL print is partly in the future. `settlement_event.py` (step 9ab) used
the settled rate at placebo instants two and four hours before the
settlement, which reads future premium, and drew a conclusion from it. The
premium index kline is the live series: Binance Vision publishes it per
symbol and month (`futures/um/monthly/premiumIndexKlines/<SYM>/<interval>/`),
open/high/low/close of the premium as a FRACTION of the index per bar. With
it, "the premium over the last N minutes" is known at the bar close and
nothing later.

Output: `data/cache/premium/<SYMBOL>.<interval>.npz` with `ts_open` and
`open/high/low/close` of the premium in bps. Months already on disk are not
re-downloaded; a month that returns 404 (before listing) is skipped.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.panel_intraday import panel_symbols  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = REPO_ROOT / "data" / "cache" / "premium"
RAW = OUT / "monthly"
BASE = "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/"


def months(start: str = "2021-09", end: Optional[str] = None) -> List[str]:
    end = end or time.strftime("%Y-%m", time.gmtime())
    y, m = (int(v) for v in start.split("-"))
    ey, em = (int(v) for v in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append("{:04d}-{:02d}".format(y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def fetch_month(symbol: str, interval: str, month: str) -> Optional[Path]:
    path = RAW / symbol / (symbol + "-" + interval + "-" + month + ".zip")
    if path.exists():
        return path
    url = BASE + symbol + "/" + interval + "/" + path.name
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return path
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            time.sleep(2.0 * (attempt + 1))
        except Exception:                             # noqa: BLE001
            time.sleep(2.0 * (attempt + 1))
    return None


def parse(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0]).decode("utf-8")
    ts, rows = [], []
    for row in csv.reader(io.StringIO(raw)):
        if not row or not row[0].isdigit():
            continue
        ts.append(int(row[0]))
        rows.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
    return np.array(ts, dtype=np.int64), np.array(rows) * 10_000.0


def fetch_symbol(symbol: str, interval: str, month_list: Sequence[str]) -> str:
    started = time.time()
    all_ts, all_rows = [], []
    for month in month_list:
        path = fetch_month(symbol, interval, month)
        if path is None:
            continue
        ts, rows = parse(path)
        if len(ts):
            all_ts.append(ts)
            all_rows.append(rows)
    if not all_ts:
        return symbol + ": nothing"
    ts = np.concatenate(all_ts)
    rows = np.concatenate(all_rows)
    order = np.argsort(ts)
    ts, rows = ts[order], rows[order]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / (symbol + "." + interval + ".npz"), ts_open=ts,
                        open=rows[:, 0], high=rows[:, 1], low=rows[:, 2], close=rows[:, 3])
    return "{:<14} {:>7} bars from {}  {:.0f}s".format(
        symbol, len(ts), time.strftime("%Y-%m-%d", time.gmtime(ts[0] / 1000)), time.time() - started)


def load_premium(symbol: str, interval: str = "15m") -> Dict[str, np.ndarray]:
    with np.load(OUT / (symbol + "." + interval + ".npz")) as z:
        return {k: z[k] for k in z.files}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbols")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", default="2021-09")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    symbols = args.symbols.split(",") if args.symbols else panel_symbols()
    month_list = months(args.start)
    todo = [s for s in symbols if args.force or not (OUT / (s + "." + args.interval + ".npz")).exists()]
    print("{} symbols, {} months each".format(len(todo), len(month_list)), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, line in enumerate(pool.map(lambda s: fetch_symbol(s, args.interval, month_list), todo), 1):
            print("{:>3}/{} {}".format(n, len(todo), line), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
