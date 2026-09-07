"""Archives raw websocket messages, exactly as received.

Why this matters more than the feature CSV
------------------------------------------
The feature recorder saves *derived* values. If in three weeks you want a
feature that doesn't exist yet — book slope, depth at 50 levels, queue
position, OFI over a different window, imbalance weighted by distance from mid
— you cannot compute it from `features-*.csv`. The information was thrown away
at write time.

The raw log keeps the source events, so any future feature can be recomputed
over all the history you've collected, and `analysis/replay.py` can regenerate
a feature file from scratch. That is the difference between a dataset that
appreciates and one that is frozen the day you designed the schema.

It is also what makes an honest backtest possible: replaying the actual book
updates reproduces the exact sequence the bot saw, including the desyncs.

Format
------
Gzipped JSON Lines, one file per channel per hour:

    data/raw/2026-09-06/books-14.jsonl.gz

Each line is a compact object (short keys, because they repeat millions of
times):

    {"t": 1757183000123, "n": 48213, "m": {...the exact exchange message...}}

  t = local receive time in ms. Kept alongside the exchange's own timestamp
      so you can measure feed latency after the fact — which you cannot
      reconstruct later if you don't store it now.
  n = a monotonic counter across ALL channels in this process.
  m = the message verbatim, unparsed and unmodified.

`n` exists because millisecond timestamps are not a fine enough clock to
recover arrival order. Book updates arrive ~10/s and trades arrive in bursts,
so a book update and a trade routinely share a millisecond. Merging the
per-channel files on `t` alone would let replay apply them in a different
order than they actually arrived, and a trade applied before rather than after
a book update produces different features. Replay must sort on `(t, n)`.

JSONL + gzip rather than a database because the write pattern is append-only,
the read pattern is a full sequential scan, and a truncated file (power cut
mid-write) costs you the last partial line instead of a corrupt table.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TextIO


class RawEventLog:
    """Append-only, hourly-rotated, gzipped archive of raw feed messages.

    Writes are buffered and flushed on a line/time threshold, so the asyncio
    event loop is not doing a gzip syscall on every book update.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        enabled: bool = True,
        channels: Optional[set] = None,
        flush_lines: int = 200,
        flush_seconds: float = 5.0,
        compress_level: int = 6,
    ):
        self.root = Path(data_dir) / "raw"
        self.enabled = enabled
        # `tickers` is deliberately excluded by default: it is redundant with
        # the book's top level and would roughly double the file count for no
        # extra information.
        self.channels = channels or {"books", "books5", "trades", "funding-rate"}
        self.flush_lines = flush_lines
        self.flush_seconds = flush_seconds
        self.compress_level = compress_level

        self._handles: Dict[str, TextIO] = {}
        self._slots: Dict[str, str] = {}
        self._pending: Dict[str, int] = {}
        self._last_flush = time.time()
        # Monotonic across every channel, so replay can recover exact arrival
        # order even when many messages share a millisecond.
        self._sequence = 0

        self.lines_written = 0
        self.bytes_estimate = 0

    # ---- writing ---------------------------------------------------------

    @staticmethod
    def _slot(now: float) -> tuple:
        stamp = datetime.fromtimestamp(now, tz=timezone.utc)
        return stamp.strftime("%Y-%m-%d"), stamp.strftime("%H")

    def _handle_for(self, channel: str, now: float) -> Optional[TextIO]:
        day, hour = self._slot(now)
        slot = f"{day}-{hour}"

        if self._slots.get(channel) == slot:
            return self._handles.get(channel)

        # Hour rolled over (or first write): close the old file cleanly first.
        old = self._handles.pop(channel, None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass

        directory = self.root / day
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{channel}-{hour}.jsonl.gz"
        # "at" appends, so a restart within the same hour adds to the existing
        # file rather than truncating an hour of history.
        handle = gzip.open(path, "at", compresslevel=self.compress_level,
                           encoding="utf-8")
        self._handles[channel] = handle
        self._slots[channel] = slot
        self._pending[channel] = 0
        return handle

    def write(self, channel: str, message: Dict[str, Any]) -> None:
        if not self.enabled or channel not in self.channels:
            return

        now = time.time()
        handle = self._handle_for(channel, now)
        if handle is None:
            return

        self._sequence += 1
        line = json.dumps(
            {"t": int(now * 1000), "n": self._sequence, "m": message},
            separators=(",", ":"),   # no spaces; this is written millions of times
            default=str,
        )
        handle.write(line)
        handle.write("\n")

        self.lines_written += 1
        self.bytes_estimate += len(line) + 1
        self._pending[channel] = self._pending.get(channel, 0) + 1

        if (
            self._pending[channel] >= self.flush_lines
            or now - self._last_flush >= self.flush_seconds
        ):
            self.flush()

    def flush(self) -> None:
        for channel, handle in self._handles.items():
            try:
                handle.flush()
            except Exception:
                pass
            self._pending[channel] = 0
        self._last_flush = time.time()

    def close(self) -> None:
        self.flush()
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        self._slots.clear()

    # ---- reporting -------------------------------------------------------

    def disk_bytes(self) -> int:
        """Actual compressed bytes on disk so far."""
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*.jsonl.gz"))

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "linesWritten": self.lines_written,
            "uncompressedBytes": self.bytes_estimate,
            "diskBytes": self.disk_bytes(),
        }


def iter_events(path: Path):
    """Read back one raw log file, yielding (receive_ms, sequence, message).

    Sort merged streams on the (receive_ms, sequence) pair — see the module
    docstring for why the timestamp alone is not sufficient.

    Tolerates a truncated final line, which is the normal result of the
    process being killed mid-write — one lost message, not a lost file.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail
            yield record.get("t", 0), record.get("n", 0), record.get("m", {})


def iter_directory(root: Path, channel: str):
    """Yield (receive_ms, sequence, message) for one channel across all
    archived hours, in chronological order."""
    for path in sorted(Path(root).rglob(f"{channel}-*.jsonl.gz")):
        yield from iter_events(path)
