"""Record open interest alongside an already-running recorder.

    python backend\\record_oi.py --instruments BTC-USDT,ADA-USDT,PUMP-USDT
    python backend\\record_oi.py --instruments BTC-USDT --once
    python backend\\record_oi.py --match-running

**This does not require restarting `record.py`.** It is a separate process
writing a separate channel: `raw/<day>/open-interest-<HH>.jsonl.gz`, never the
`books-` or `trades-` files a live recorder holds open. Start it against a run
that is already three days in and nothing about that run changes.

That property is the reason it is a standalone entrypoint rather than another
task inside `record.py`. Adding it there would mean stopping a recorder to
start collecting a series - and the argument for collecting OI at all is that
the hours you do not have are gone, which a restart makes worse before it
makes better.

**`record.py` now starts a poller of its own, so do not run both.** They would
write the same `open-interest-<HH>.jsonl.gz` from two processes, and two gzip
writers appending to one file produce a stream that does not decode - measured
at 0 of 40 records recovered, not merely duplicated. Each poller therefore
takes an exclusive per-instrument lock and the second is refused with the
owning pid. This entrypoint is for a recorder started BEFORE that change, or
for instruments the running recorder is not covering.

What it collects and why
------------------------
Open interest is the one publicly available input to estimating where OTHER
traders are liquidated. `trading/risk.py` models our own liquidation price
exactly; nothing models anyone else's, because none of `books`, `trades` or
`funding-rate` carries a fact about someone else's position. OI does, and
BloFin serves only a snapshot - no history endpoint - so the series has to be
accumulated live or it does not exist. See `trading/openinterest.py` for the
measured behaviour of the endpoint and for what is deliberately NOT built yet
(the cluster model itself).

Cost: one HTTP request per poll for ALL instruments, and roughly 300 KB/day
per instrument before compression. Against ~140 MB/day/instrument for the
book and trade feed, this is free.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, INST_ID, RECORD_RAW  # noqa: E402
from server import log, supervise  # noqa: E402
from trading.openinterest import (  # noqa: E402
    DEFAULT_POLL_SECONDS,
    OpenInterestPoller,
    PRODUCTION_BASE_URL,
)


def running_recorder_instruments() -> Optional[List[str]]:
    """The instrument list of a `record.py` already running on this machine.

    Typing the list twice is how the two processes end up recording different
    symbol sets, which produces an OI series with holes exactly where the book
    data is densest. Reading it off the running process removes that failure
    mode entirely. Windows-only (WMIC/CIM); returns None anywhere else, or if
    no recorder is running, and the caller falls back to --instruments.
    """
    if sys.platform != "win32":
        return None
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
             "| Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    for line in (completed.stdout or "").splitlines():
        if "record.py" in line and "record_oi.py" not in line:
            match = re.search(r"--instruments\s+(\S+)", line)
            if match:
                return [part.strip() for part in match.group(1).split(",")
                        if part.strip()]
    return None


async def run(poller: OpenInterestPoller) -> None:
    log("Polling open interest every "
        f"{poller.poll_seconds:.0f}s for {len(poller.instruments)} "
        "instrument(s).")
    log("  One request covers every instrument; rows are written only when "
        "the exchange's\n  minute-stamped value actually changes, so this is "
        "one row per instrument per minute.")
    for inst_id in poller.instruments:
        log(f"  {inst_id:<18} -> {poller.data_root / inst_id / 'raw'}")
    log("Different channel from books/trades, so a live record.py's feed "
        "files are untouched.")
    log("  A second OI poller on the same instrument is refused: two of them "
        "destroy the\n  hour's file rather than duplicating it.")

    try:
        # Supervised for the same reason every other loop here is: a dropped
        # HTTP connection must not end an overnight collection run.
        await supervise("open-interest", lambda: poller.run(on_log=log))
    finally:
        poller.close()
        log(f"Open interest poller closed. {poller.rows_written} row(s) "
            f"written, {poller.failures} failed poll(s).")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--instruments", default=None,
        help="Comma-separated instrument ids (default: BLOFIN_INST_ID, or the "
             "running recorder's list with --match-running).")
    parser.add_argument(
        "--match-running", action="store_true",
        help="Take the instrument list from a record.py already running on "
             "this machine, so the two cannot drift apart.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--poll-seconds", type=float,
                        default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--base-url", default=PRODUCTION_BASE_URL)
    parser.add_argument(
        "--once", action="store_true",
        help="One poll, then exit. Use this to prove the endpoint and the "
             "write path work before leaving it running.")
    args = parser.parse_args(argv)

    instruments: List[str] = []
    if args.match_running:
        found = running_recorder_instruments()
        if found:
            log(f"Matched running recorder: {len(found)} instrument(s).")
            instruments = found
        else:
            log("No running record.py found - falling back to --instruments.")
    if not instruments:
        instruments = [part.strip()
                       for part in (args.instruments or INST_ID).split(",")
                       if part.strip()]
    if not instruments:
        raise SystemExit("No instruments given.")

    duplicates = {name for name in instruments if instruments.count(name) > 1}
    if duplicates:
        raise SystemExit(
            f"Instrument(s) listed more than once: {', '.join(sorted(duplicates))}"
        )

    if not RECORD_RAW:
        raise SystemExit(
            "BLOFIN_RECORD_RAW is false, so this would poll and discard every "
            "response.\nNothing to do."
        )

    poller = OpenInterestPoller(
        instruments,
        data_dir=args.data_dir,
        base_url=args.base_url,
        poll_seconds=args.poll_seconds,
    )

    if args.once:
        written = poller.poll_once(on_log=log)
        poller.close()
        log(f"One poll: {written} row(s) written to {args.data_dir}.")
        if written == 0 and poller.failures:
            return 1
        return 0

    try:
        asyncio.run(run(poller))
    except KeyboardInterrupt:
        log("Interrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
