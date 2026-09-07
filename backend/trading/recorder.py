"""Writes feature rows to disk with forward-looking labels attached.

Why this module exists
----------------------
None of the ML in the plan — LightGBM, the regime model, anything — can be
built without a dataset of (features at time t, what happened after t). This
records exactly that, so that after a few days of running you have something
real to train on.

The one thing this file must not get wrong: **lookahead bias**. A row is only
written once enough real time has passed to observe its label, and the label
is computed strictly from mid-prices timestamped *after* the feature snapshot.
Rows still inside their horizon sit in a pending buffer and are never written.
Get this wrong and you train a model with a 90% hit rate that loses money on
every live trade.

Output: one CSV per UTC day in the data directory. CSV rather than parquet by
default because it has zero dependencies, appends safely, and survives the
process being killed mid-write — all of which matter more than file size for a
recorder you leave running. `to_parquet.py`-style conversion is trivial later.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from .features import FeatureSnapshot, feature_columns


@dataclass
class LabelConfig:
    """Forward horizons (seconds) and the move size that counts as a signal.

    `threshold_bps` should be set relative to your actual round-trip cost. If
    taker fees are 6bps round trip, labelling a 1bps move as "up" trains the
    model to find moves it cannot profitably capture. The default of 3bps is a
    starting point, not a recommendation — measure your real fills.
    """

    horizons_seconds: Tuple[float, ...] = (1.0, 5.0, 30.0)
    threshold_bps: float = 3.0


class FeatureRecorder:
    """Buffers feature snapshots, labels them once their future is known,
    and appends completed rows to a daily CSV."""

    def __init__(
        self,
        data_dir: Path,
        *,
        label_config: Optional[LabelConfig] = None,
        sample_interval_ms: int = 250,
        flush_every: int = 50,
        enabled: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.labels = label_config or LabelConfig()
        self.sample_interval_ms = sample_interval_ms
        self.flush_every = flush_every
        self.enabled = enabled

        self.max_horizon_ms = int(max(self.labels.horizons_seconds) * 1000)

        # Snapshots waiting for their forward window to close.
        self._pending: Deque[FeatureSnapshot] = deque()
        # (ts_ms, mid) used to resolve labels. Must outlive the longest horizon.
        self._mid_history: Deque[Tuple[int, float]] = deque()

        self._last_sample_ms = 0
        self._latest_ts = 0
        self._rows_since_flush = 0
        self._handle = None
        self._writer: Optional[csv.DictWriter] = None
        self._open_date: Optional[str] = None

        self.rows_written = 0
        self.rows_dropped = 0

    # ---- column layout ---------------------------------------------------

    def _columns(self) -> List[str]:
        columns = feature_columns()
        for horizon in self.labels.horizons_seconds:
            tag = self._tag(horizon)
            columns.append(f"fwd_ret_bps_{tag}")
            columns.append(f"label_{tag}")
        return columns

    @staticmethod
    def _tag(horizon: float) -> str:
        return f"{horizon:g}s".replace(".", "p")

    # ---- ingestion -------------------------------------------------------

    def observe(self, snapshot: FeatureSnapshot) -> None:
        """Feed every computed snapshot in. Downsampling happens here, not at
        the call site, so the feature engine keeps full resolution for its own
        rolling calculations while the file stays a manageable size."""
        if not self.enabled or not snapshot.is_valid or snapshot.mid is None:
            return

        self._latest_ts = max(self._latest_ts, snapshot.ts)
        self._mid_history.append((snapshot.ts, snapshot.mid))
        self._trim_mid_history()

        if snapshot.ts - self._last_sample_ms >= self.sample_interval_ms:
            self._last_sample_ms = snapshot.ts
            self._pending.append(snapshot)

        self._drain()

    def _trim_mid_history(self) -> None:
        # Keep a generous margin beyond the longest horizon so label lookups
        # never fall off the front of the buffer.
        cutoff = self._latest_ts - (self.max_horizon_ms * 3)
        while self._mid_history and self._mid_history[0][0] < cutoff:
            self._mid_history.popleft()

    def _drain(self) -> None:
        """Write every pending row whose full forward window has elapsed."""
        while self._pending:
            snapshot = self._pending[0]
            if self._latest_ts - snapshot.ts < self.max_horizon_ms:
                break  # future not observed yet — the whole point
            self._pending.popleft()
            row = self._label(snapshot)
            if row is not None:
                self._write(row)
            else:
                self.rows_dropped += 1

    def _mid_at_or_after(self, target_ts: int) -> Optional[float]:
        """First observed mid at or after `target_ts`.

        Strictly forward-looking. If there is no sample at or after the target
        (a feed gap), we return None and the row is dropped rather than
        labelled from a price that predates the horizon.
        """
        for sample_ts, mid in self._mid_history:
            if sample_ts >= target_ts:
                return mid
        return None

    def _label(self, snapshot: FeatureSnapshot) -> Optional[Dict[str, object]]:
        if snapshot.mid is None or snapshot.mid <= 0:
            return None

        row: Dict[str, object] = dict(snapshot.to_dict())
        for horizon in self.labels.horizons_seconds:
            future_mid = self._mid_at_or_after(snapshot.ts + int(horizon * 1000))
            if future_mid is None or future_mid <= 0:
                return None
            forward_bps = (future_mid / snapshot.mid - 1.0) * 10_000.0
            tag = self._tag(horizon)
            row[f"fwd_ret_bps_{tag}"] = round(forward_bps, 4)
            if forward_bps >= self.labels.threshold_bps:
                row[f"label_{tag}"] = 1
            elif forward_bps <= -self.labels.threshold_bps:
                row[f"label_{tag}"] = -1
            else:
                row[f"label_{tag}"] = 0
        return row

    # ---- output ----------------------------------------------------------

    def _ensure_file(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._handle is not None and self._open_date == today:
            return

        self.close()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / f"features-{today}.csv"
        is_new = not path.exists() or path.stat().st_size == 0

        self._handle = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self._columns(), extrasaction="ignore"
        )
        if is_new:
            self._writer.writeheader()
        self._open_date = today

    def _write(self, row: Dict[str, object]) -> None:
        self._ensure_file()
        assert self._writer is not None and self._handle is not None
        self._writer.writerow(row)
        self.rows_written += 1
        self._rows_since_flush += 1
        # Flush periodically so an unclean shutdown loses seconds of data,
        # not hours of it.
        if self._rows_since_flush >= self.flush_every:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._rows_since_flush = 0

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except Exception:
                pass
            self._handle.close()
        self._handle = None
        self._writer = None
        self._open_date = None

    def stats(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "rowsWritten": self.rows_written,
            "rowsDropped": self.rows_dropped,
            "pending": len(self._pending),
            "file": f"features-{self._open_date}.csv" if self._open_date else None,
        }
