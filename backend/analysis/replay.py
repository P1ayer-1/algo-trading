"""Rebuild a feature file from archived raw events.

    python backend/analysis/replay.py --date 2026-09-06
    python backend/analysis/replay.py --date 2026-09-06 --sample-ms 100 \
        --horizons 0.5,2,10 --out data/replayed

This is what makes the raw archive worth keeping. Change a feature, add a new
one, or relabel at a different horizon, then replay every hour you have ever
recorded and get a fresh dataset — without waiting days to collect it again.

It reuses the exact same `OrderBook`, `TradeTape`, `FeatureEngine` and
`FeatureRecorder` as the live path. That is deliberate: if replay used a
separate implementation, a model trained on replayed data would be trained on
subtly different features than the live bot computes, and the discrepancy
would be nearly impossible to find later.

Because the code is shared, replay also reproduces the desyncs: if the book hit
a sequence gap live, it hits the same gap here and the same rows go missing.
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.features import FeatureEngine  # noqa: E402
from trading.orderbook import OrderBook  # noqa: E402
from trading.rawlog import iter_events  # noqa: E402
from trading.recorder import FeatureRecorder, LabelConfig  # noqa: E402
from trading.tape import TradeTape  # noqa: E402

CHANNELS = ("books", "books5", "trades", "funding-rate")


def merged_events(raw_dir: Path, date: Optional[str]) -> Iterator[Tuple[int, int, dict]]:
    """All archived events for a date (or everything), in exact arrival order.

    Files are per-channel, so books and trades must be interleaved back into a
    single stream — otherwise every trade would be applied against a book from
    the wrong moment. `heapq.merge` does this lazily, so memory stays flat
    regardless of archive size.

    The sort key is `(receive_ms, sequence)`, not the timestamp alone. Book
    updates and trades regularly share a millisecond, and ordering them wrongly
    changes the computed features — a trade applied before rather than after a
    book update lands in a different feature snapshot.
    """
    pattern = f"{date}/*.jsonl.gz" if date else "*/*.jsonl.gz"
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No raw logs matching {pattern} under {raw_dir}")

    print(f"Replaying {len(files)} file(s):")
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        print(f"  {path.relative_to(raw_dir)}  ({size / 1e6:.1f} MB compressed)")
    print(f"  total {total / 1e6:.1f} MB\n")

    streams = [iter_events(path) for path in files]
    return heapq.merge(*streams, key=lambda item: (item[0], item[1]))


def replay(
    raw_dir: Path,
    out_dir: Path,
    *,
    date: Optional[str],
    sample_ms: int,
    horizons: Tuple[float, ...],
    threshold_bps: float,
) -> dict:
    book = OrderBook()
    tape = TradeTape()
    engine = FeatureEngine()
    recorder = FeatureRecorder(
        out_dir,
        label_config=LabelConfig(horizons_seconds=horizons, threshold_bps=threshold_bps),
        sample_interval_ms=sample_ms,
    )

    counts = {"books": 0, "trades": 0, "funding": 0, "other": 0, "desyncs": 0}

    try:
        for _, _, message in merged_events(raw_dir, date):
            if not isinstance(message, dict):
                continue
            channel = message.get("arg", {}).get("channel", "")

            if channel in ("books", "books5"):
                book.apply(message)
                counts["books"] += 1
                if not book.ready:
                    # Same desync behaviour as live. Live would reconnect and
                    # get a snapshot; here we simply wait for the next
                    # archived snapshot to arrive.
                    counts["desyncs"] += 1
                    continue
                if book.is_crossed():
                    continue
                engine.on_book_event(book)
                recorder.observe(engine.compute(book, tape))
            elif channel == "trades":
                counts["trades"] += 1
                if tape.add_message(message.get("data")):
                    recorder.observe(engine.compute(book, tape))
            elif channel == "funding-rate":
                counts["funding"] += 1
                engine.on_funding(message.get("data"))
            else:
                counts["other"] += 1
    finally:
        recorder.close()

    counts.update(recorder.stats())
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument("--out", type=Path, default=repo_root / "data" / "replayed")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD; omit for all.")
    parser.add_argument("--sample-ms", type=int, default=250)
    parser.add_argument("--horizons", default="1,5,30",
                        help="Comma-separated forward horizons in seconds.")
    parser.add_argument("--threshold-bps", type=float, default=3.0)
    args = parser.parse_args(argv)

    horizons = tuple(float(part) for part in args.horizons.split(",") if part.strip())

    result = replay(
        args.raw_dir, args.out,
        date=args.date, sample_ms=args.sample_ms,
        horizons=horizons, threshold_bps=args.threshold_bps,
    )

    print("Replay complete:")
    print(f"  book messages   {result['books']:,}")
    print(f"  trade messages  {result['trades']:,}")
    print(f"  funding         {result['funding']:,}")
    print(f"  desync events   {result['desyncs']:,}")
    print(f"  rows written    {result['rowsWritten']:,}")
    print(f"  rows dropped    {result['rowsDropped']:,} (no observable future)")
    print(f"\nOutput: {args.out}")
    print("Run the predictiveness check against it with --data-dir", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
