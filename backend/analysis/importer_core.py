"""Shared machinery for the historical importers.

`binance_import.py` and `tardis_import.py` both do the same three things:
fetch a file, turn it into BloFin-shaped websocket messages, and push those
through the *same* OrderBook / TradeTape / FeatureEngine / FeatureRecorder the
live bot uses. Only the middle step — the venue-specific parsing — differs, so
the outer two live here and neither importer gets to drift from the other.

Both importers emit events as `(timestamp_ms, per_stream_index, message)`
tuples. `build_features` merges the two streams on that key, so an importer
only has to produce correctly-ordered events per stream and never has to think
about interleaving.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trading.features import FeatureEngine
from trading.orderbook import OrderBook
from trading.recorder import FeatureRecorder, LabelConfig
from trading.tape import TradeTape

# (timestamp_ms, index within its own stream, BloFin-shaped message)
Event = Tuple[int, int, dict]


class SchemaError(RuntimeError):
    """Raised when a downloaded file doesn't look like what we expect.

    Failing loudly matters more than usual here: a silently mis-parsed column
    produces a feature file that looks perfectly normal and is entirely wrong.
    """


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download(url: str, destination: Path, *, force: bool = False,
             not_found_hint: Optional[str] = None,
             unauthorised_hint: Optional[str] = None,
             headers: Optional[Dict[str, str]] = None) -> Path:
    """Fetch a file unless it's already on disk. Streams to a .part file so an
    interrupted download can never be mistaken for a complete one."""
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"  cached  {destination.name} "
              f"({destination.stat().st_size / 1e6:.1f} MB)")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"  fetching {url}")

    started = time.time()
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = downloaded / total * 100
                        print(f"\r    {downloaded/1e6:7.1f} / {total/1e6:.1f} MB "
                              f"({percent:5.1f}%)", end="", flush=True)
            print()
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        if exc.code == 404 and not_found_hint:
            raise SystemExit(f"\nNot found: {url}\n{not_found_hint}") from exc
        if exc.code in (401, 403) and unauthorised_hint:
            raise SystemExit(
                f"\nHTTP {exc.code} (not authorised): {url}\n{unauthorised_hint}"
            ) from exc
        if exc.code == 404:
            raise SystemExit(
                f"\nNot found: {url}\n"
                "The file does not exist for that date and symbol."
            ) from exc
        raise SystemExit(f"\nHTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"\nCould not reach {url}: {exc.reason}\n"
            "This needs plain internet access. If you are behind a proxy or VPN "
            "that blocks it, download the file manually in a browser and place "
            f"it at {destination}."
        ) from exc

    partial.replace(destination)
    print(f"  saved   {destination.name} "
          f"({destination.stat().st_size / 1e6:.1f} MB in {time.time() - started:.0f}s)")
    return destination


# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------


def normalise_timestamp(value: int) -> int:
    """Return milliseconds.

    Venues and archives use both milliseconds and microseconds. A millisecond
    timestamp for any plausible date is ~1.7e12; anything at ~1.7e15 is
    microseconds. Guessing wrong here would silently scale every horizon by
    1000, so it is detected rather than assumed.
    """
    if value > 1e14:
        return value // 1000
    return value


def check_epoch_ms(ts: int, where: str) -> None:
    """Reject a timestamp that isn't a plausible millisecond epoch."""
    if not 1_000_000_000_000 < ts < 4_000_000_000_000:
        raise SchemaError(
            f"{where}: timestamp {ts} is not a plausible millisecond epoch. "
            "Check the timestamp column."
        )


def book_message(bids: List[List[float]], asks: List[List[float]],
                 ts: int, seq: int) -> dict:
    """A BloFin-shaped `books` snapshot.

    Every row is emitted as a full snapshot rather than an incremental update.
    That is honest about what the source contains and sidesteps sequence
    handling entirely, since each message is self-contained.
    """
    return {
        "arg": {"channel": "books", "instId": "IMPORT"},
        "action": "snapshot",
        "data": {
            "bids": bids,
            "asks": asks,
            "ts": str(ts),
            "seqId": str(seq),
            "prevSeqId": "0",
        },
    }


def trade_message(price: float, size: float, side: str, ts: int) -> dict:
    """A BloFin-shaped `trades` message. `side` is the AGGRESSOR side."""
    return {
        "arg": {"channel": "trades", "instId": "IMPORT"},
        "data": [{
            "price": str(price),
            "size": str(size),
            "side": side,
            "ts": str(ts),
        }],
    }


# ---------------------------------------------------------------------------
# The pipeline itself
# ---------------------------------------------------------------------------


def build_features(
    books: List[Event],
    trades: List[Event],
    out_dir: Path,
    *,
    sample_ms: int,
    horizons: Tuple[float, ...],
    threshold_bps: float,
) -> Dict[str, object]:
    """Replay merged events through the live feature stack, writing a CSV.

    Features are computed on *every* event, exactly as the live bot does,
    rather than only at the recorder's sample interval. Computing less often
    would be faster but would give imported data subtly different label
    anchors than live data, and quietly divergent training inputs are a far
    more expensive problem than a slow import.
    """
    if not books:
        raise SystemExit("No usable book updates were parsed.")

    # Both streams are already sorted, so a merge on (timestamp, per-stream
    # index) restores event order across them. At an identical timestamp book
    # events sort first, which is the conservative choice: the trade is then
    # priced against a book that already reflects everything at that instant.
    print("Merging and computing features...")
    merged = sorted(books + trades, key=lambda item: (item[0], item[1]))

    book, tape, engine = OrderBook(), TradeTape(), FeatureEngine()
    recorder = FeatureRecorder(
        out_dir,
        label_config=LabelConfig(horizons_seconds=horizons, threshold_bps=threshold_bps),
        sample_interval_ms=sample_ms,
    )

    processed = 0
    try:
        for _, _, message in merged:
            channel = message["arg"]["channel"]
            if channel == "books":
                book.apply(message)
                if book.is_ready and not book.is_crossed():
                    engine.on_book_event(book)
                    recorder.observe(engine.compute(book, tape))
            else:
                if tape.add_message(message["data"]):
                    recorder.observe(engine.compute(book, tape))
            processed += 1
            if processed % 500_000 == 0:
                print(f"  {processed:,} / {len(merged):,} events "
                      f"({recorder.rows_written:,} rows written)")
    finally:
        recorder.close()

    stats = dict(recorder.stats())
    stats["events"] = processed
    stats["books"] = len(books)
    stats["trades"] = len(trades)
    return stats


def print_summary(stats: Dict[str, object], out_dir: Path) -> None:
    print("\n" + "=" * 66)
    print(f"  events processed  {stats['events']:,}")
    print(f"  rows written      {stats['rowsWritten']:,}")
    print(f"  rows dropped      {stats['rowsDropped']:,} (no observable future)")
    print(f"  rows pending      {stats['pending']:,} (inside the horizon at EOF)")
    print(f"  output            {out_dir}")


def ensure_clean_output(out_dir: Path) -> None:
    """Refuse to import into a directory that already holds a feature CSV.

    `FeatureRecorder` names its file `features-<today>.csv` by **wall clock**
    date and opens it in append mode. That is right for live recording, where
    wall clock and market time are the same thing, and wrong for importing,
    where they are not: two imports run on the same afternoon land in one
    file, ordered by when you ran them rather than by market time.

    `check_features.py` then does a time-ordered, purged train/test split on
    rows that are not in time order. It will not complain — it will just quietly
    report a number that means nothing. So this stops instead.
    """
    existing = sorted(out_dir.glob("features-*.csv")) if out_dir.exists() else []
    if not existing:
        return
    names = "\n".join(f"    {path}" for path in existing)
    raise SystemExit(
        f"\n{out_dir} already contains a feature CSV:\n{names}\n\n"
        "The recorder appends, and it names files by today's date rather than\n"
        "the market date, so importing here again would interleave two runs in\n"
        "one file and silently break the time-ordered split in check_features.\n\n"
        "Delete that file to redo this import, or pass --out to write elsewhere."
    )
