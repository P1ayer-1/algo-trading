"""Watch Binance's listing announcements and record what BloFin's book does next.

    python backend\\announcement_watch.py --check
    python backend\\announcement_watch.py
    python backend\\announcement_watch.py --poll-seconds 5 --track-minutes 180

Why this exists, and why it sends nothing
-----------------------------------------
`analysis/listing_announcement.py` (2026-09-12) measured, on Bybit's 1-minute
klines, that a coin Binance announces it will list keeps rising for one to
two hours after the announcement minute (+594 bps at 60 minutes, n=36) and
that a coin whose Binance FUTURES launch is announced gives back a third of
its jump within 15 minutes (-238 bps, n=73). Both clear a 30 bps taker leg
by a wide margin on paper. Every number in that study is a Bybit close: the
venue this account trades is BloFin, whose book on a thin coin in the minute
after such an announcement has never been observed here, and whose spread in
that minute is the whole question for a strategy that pays it twice.

So this records. It polls Binance's announcement catalogue, and when a new
listing or futures-launch article appears it resolves the coin to a BloFin
instrument and samples that instrument's top of book (and BTC-USDT's, as the
market term) every few seconds for the next `--track-minutes`, to
`data/announcements/<stamp>-<instId>.csv` beside the article itself. There
is no order path in this file. When the CSVs say what the Bybit klines said,
an executor is the next thing to ask for; until then this is step one, as it
was for everything else in this repo.

Latency is part of the measurement: the first row's timestamp against the
article's `releaseDate` is how late a poller at this cadence sees the news.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data" / "announcements"
CMS = ("https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
       "?type=1&catalogId=48&pageSize=20&pageNo=1")
BLOFIN = "https://openapi.blofin.com/api/v1/market/"

from analysis.listing_announcement import classify  # noqa: E402


def get_json(url: str, timeout: float = 15.0):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                   "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def latest_articles() -> List[dict]:
    data = get_json(CMS)["data"]["catalogs"][0]
    return [{"id": str(r["code"]), "title": r["title"], "release_ms": int(r["releaseDate"])}
            for r in (data.get("articles") or [])]


def blofin_instruments() -> Set[str]:
    return {row["instId"] for row in get_json(BLOFIN + "instruments")["data"]}


def resolve(ticker: str, instruments: Set[str]) -> Optional[str]:
    for inst in (ticker + "-USDT", "1000" + ticker + "-USDT", "10000" + ticker + "-USDT"):
        if inst in instruments:
            return inst
    return None


def ticker(inst_id: str) -> Optional[dict]:
    rows = get_json(BLOFIN + "tickers?instId=" + inst_id, timeout=8.0)["data"]
    return rows[0] if rows else None


def track(article: dict, kind: str, inst_id: str, minutes: int, sample_seconds: float) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(article["release_ms"] / 1000))
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / (stamp + "-" + inst_id)
    with (base.with_suffix(".json")).open("w", encoding="utf-8") as handle:
        json.dump({**article, "kind": kind, "instId": inst_id}, handle)
    path = base.with_suffix(".csv")
    started = time.time()
    first_mid: Optional[float] = None
    marks: Dict[int, float] = {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_ms", "instId", "bid", "ask", "bid_size", "ask_size", "last",
                         "btc_bid", "btc_ask"])
        while time.time() - started < minutes * 60:
            now = int(time.time() * 1000)
            try:
                row = ticker(inst_id)
                btc = ticker("BTC-USDT")
            except Exception as exc:                  # noqa: BLE001
                print("  {} sample failed: {}".format(inst_id, exc), flush=True)
                time.sleep(sample_seconds)
                continue
            if row and btc:
                bid, ask = float(row["bidPrice"]), float(row["askPrice"])
                writer.writerow([now, inst_id, bid, ask, row["bidSize"], row["askSize"], row["last"],
                                 btc["bidPrice"], btc["askPrice"]])
                handle.flush()
                mid = (bid + ask) / 2.0
                if first_mid is None and mid > 0:
                    first_mid = mid
                    print("  {} first sample {:.1f}s after release: bid {} ask {} spread {:.1f} bps".format(
                        inst_id, (now - article["release_ms"]) / 1000.0, bid, ask,
                        (ask - bid) / mid * 1e4), flush=True)
                elapsed_min = int((time.time() - started) // 60)
                if first_mid and elapsed_min in (1, 2, 5, 15, 60, 120) and elapsed_min not in marks:
                    marks[elapsed_min] = (mid / first_mid - 1.0) * 1e4
                    print("  {} +{}m: {:+.0f} bps from first sample".format(
                        inst_id, elapsed_min, marks[elapsed_min]), flush=True)
            time.sleep(sample_seconds)
    print("{} {} tracked for {} min: ".format(kind, inst_id, minutes)
          + "  ".join("+{}m {:+.0f}".format(k, v) for k, v in sorted(marks.items())), flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--track-minutes", type=int, default=150)
    parser.add_argument("--check", action="store_true",
                        help="reach both endpoints, show the latest articles, and exit")
    args = parser.parse_args(argv)

    instruments = blofin_instruments()
    articles = latest_articles()
    print("BloFin lists {} instruments; Binance catalogue reachable, {} latest articles".format(
        len(instruments), len(articles)), flush=True)
    for art in articles[:5]:
        kind, ticks = classify(art["title"])
        print("  {}  {}  {}".format(time.strftime("%Y-%m-%d %H:%M", time.gmtime(art["release_ms"] / 1000)),
                                    (kind or "-").ljust(7), art["title"][:90].encode("ascii", "replace").decode()))
    if args.check:
        return 0

    seen = {a["id"] for a in articles}
    refreshed = time.time()
    print("watching every {:.0f}s; tracking {} min per event; writing to {}".format(
        args.poll_seconds, args.track_minutes, OUT), flush=True)
    while True:
        time.sleep(args.poll_seconds)
        try:
            if time.time() - refreshed > 3600:
                instruments = blofin_instruments()
                refreshed = time.time()
            articles = latest_articles()
        except Exception as exc:                      # noqa: BLE001
            print("poll failed: {}".format(exc), flush=True)
            continue
        for art in articles:
            if art["id"] in seen:
                continue
            seen.add(art["id"])
            kind, ticks = classify(art["title"])
            title = art["title"].encode("ascii", "replace").decode()
            if kind is None:
                print("new article, not a listing: {}".format(title[:90]), flush=True)
                continue
            for tick in ticks:
                inst_id = resolve(tick, instruments)
                if inst_id is None:
                    print("{} listing of {}: not on BloFin ({})".format(kind, tick, title[:60]), flush=True)
                    continue
                print("{} listing of {} -> {}: {}".format(kind, tick, inst_id, title[:80]), flush=True)
                threading.Thread(target=track, args=(art, kind, inst_id, args.track_minutes,
                                                     args.sample_seconds), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
