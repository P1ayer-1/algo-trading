"""Record several instruments at once, headless. No chart, no HTTP, no ports.

    python backend\\record.py --instruments BTC-USDT,ADA-USDT,PUMP-USDT
    python backend\\record.py --instruments ADA-USDT --data-dir data
    python backend\\record.py --list-cost --instruments A,B,C   # disk only, no run

`live-chart.py` records one instrument and serves a chart for it. That is the
right shape for watching a market and the wrong shape for collecting a
dataset: the passive branch needs book and trade data on the WIDE-spread
instruments (see `analysis/blofin_spread_survey.py`), while the prediction
branch needs continuity on BTC-USDT, and those are different symbols.

Running N copies of the chart would mean N sets of HTTP and websocket ports to
keep straight, N browser pages nobody looks at, and N chances to typo a port
into a collision. Instead this runs N `MicrostructureFeed`s as N supervised
asyncio tasks in one process, with no servers at all.

Each feed opens its own websocket connection and writes to its own
`data/<INST-ID>/` directory. Nothing is shared between them except the
process, so one instrument desyncing, stalling or reconnecting cannot touch
another's data.

Every feed runs under `server.supervise`, so no exception in one loop can end
the run - the same guarantee `live-chart.py` gets, and for the same reason: a
collection run that dies at 03:00 costs the hours nobody was awake for, and
the raw archive is the half that cannot be re-collected.

Cost
----
Roughly **140 MB/day per instrument** (~100 MB raw + ~40 MB features at the 1s
sample interval). That is measured, not estimated - see `analysis/README.md`.
Six instruments is therefore ~840 MB/day and ~25 GB/month. `--list-cost`
prints the arithmetic for a chosen set and exits without connecting, which is
worth doing before leaving anything running for a week.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    BOOK_DEPTH,
    DATA_DIR,
    FEATURE_SAMPLE_INTERVAL_MS,
    FEED_STALL_TIMEOUT_S,
    INST_ID,
    LABEL_HORIZONS,
    LABEL_THRESHOLD_BPS,
    RECORD_FEATURES,
    RECORD_RAW,
    TAPE_WINDOW_SECONDS,
    USE_DEMO,
)
from server import log, supervise  # noqa: E402
from trading.ingest import MicrostructureFeed  # noqa: E402
from trading.recorder import LabelConfig  # noqa: E402

# Measured, in analysis/README.md. Raw dominates and scales with message rate,
# so a busier instrument costs more than a quiet one; this is the BTC-USDT
# figure and therefore an upper-ish bound for the thinner symbols.
MB_PER_DAY_PER_INSTRUMENT = 140


def build_feed(inst_id: str, data_dir: Path) -> MicrostructureFeed:
    """One feed, configured exactly as `live-chart.py` configures its own."""
    return MicrostructureFeed(
        inst_id,
        use_demo=USE_DEMO,
        book_depth=BOOK_DEPTH,
        data_dir=data_dir,
        record=RECORD_FEATURES,
        record_raw=RECORD_RAW,
        sample_interval_ms=FEATURE_SAMPLE_INTERVAL_MS,
        tape_window_seconds=TAPE_WINDOW_SECONDS,
        stall_timeout_s=FEED_STALL_TIMEOUT_S,
        label_config=LabelConfig(
            horizons_seconds=LABEL_HORIZONS,
            threshold_bps=LABEL_THRESHOLD_BPS,
        ),
    )


def report_cost(instruments: List[str], data_dir: Path) -> None:
    daily = len(instruments) * MB_PER_DAY_PER_INSTRUMENT
    print(f"\n{len(instruments)} instrument(s) -> {data_dir}")
    for inst_id in instruments:
        print(f"  {inst_id:<18} {data_dir / inst_id}")
    print(f"\nMeasured cost, at ~{MB_PER_DAY_PER_INSTRUMENT} MB/day each:")
    print(f"  per day      {daily / 1000:.2f} GB")
    print(f"  per week     {daily * 7 / 1000:.2f} GB")
    print(f"  per 30 days  {daily * 30 / 1000:.2f} GB")
    print("\nThe raw archive is most of that, and it is the half that cannot "
          "be re-collected.\nFeature CSVs are regenerable from it with "
          "analysis/replay.py, so if disk gets\ntight, delete those first and "
          "never the archive.")


async def run(instruments: List[str], data_dir: Path) -> None:
    feeds = [(inst_id, build_feed(inst_id, data_dir)) for inst_id in instruments]

    log(f"Recording {len(feeds)} instrument(s), headless. No chart, no ports.")
    for inst_id, feed in feeds:
        writers = []
        if feed.recorder.enabled:
            writers.append("features")
        if feed.raw_log.enabled:
            writers.append("raw")
        log(f"  {inst_id:<18} -> {feed.instrument_dir}  "
            f"[{', '.join(writers) if writers else 'NOTHING ENABLED'}]")

    if not any(feed.recorder.enabled or feed.raw_log.enabled
               for _, feed in feeds):
        raise SystemExit(
            "Both writers are disabled (BLOFIN_RECORD_FEATURES and "
            "BLOFIN_RECORD_RAW), so this\nwould connect and discard "
            "everything. Nothing to do."
        )

    log(f"Nothing reaches the feature CSVs for the first "
        f"{max(LABEL_HORIZONS) / 60:.0f} minutes - a row cannot be written "
        f"until its\n  forward window has closed. The raw archive starts "
        f"immediately.")

    try:
        # Supervised, so a failure in one instrument's loop restarts that loop
        # instead of ending the process and every other instrument with it.
        await asyncio.gather(*(
            supervise(f"feed:{inst_id}", feed.run) for inst_id, feed in feeds
        ))
    finally:
        # Flush every writer, even on Ctrl-C. The pending-row buffer is lost
        # either way - those rows have no closed forward window yet - but what
        # is already labelled belongs on disk.
        for inst_id, feed in feeds:
            try:
                feed.close()
            except Exception as exc:  # pragma: no cover - shutdown path
                log(f"  {inst_id}: close failed: {exc}")
        log("Recorders flushed and closed.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--instruments", default=INST_ID,
        help="Comma-separated instrument ids (default: BLOFIN_INST_ID).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--list-cost", action="store_true",
        help="Print the disk arithmetic for this set and exit without "
             "connecting.")
    args = parser.parse_args(argv)

    instruments = [part.strip() for part in args.instruments.split(",")
                   if part.strip()]
    if not instruments:
        raise SystemExit("No instruments given.")

    duplicates = {name for name in instruments if instruments.count(name) > 1}
    if duplicates:
        # Two feeds on one symbol would interleave rows into one file from two
        # independent books, which is corruption rather than more data.
        raise SystemExit(
            f"Instrument(s) listed more than once: {', '.join(sorted(duplicates))}"
        )

    if args.list_cost:
        report_cost(instruments, args.data_dir)
        return 0

    report_cost(instruments, args.data_dir)
    try:
        asyncio.run(run(instruments, args.data_dir))
    except KeyboardInterrupt:
        log("Interrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
