"""Where recorded data lives, in one place.

`recorder.py` writes to `data/<INST-ID>/`, and three separate tools need to
find their way back in: `check_features.py` wants the feature CSVs,
`replay.py` and `passive_sim.py` want the raw archive. Each of them having its
own idea of the layout is how `data/raw` ended up hardcoded in two defaults
that silently stopped existing the moment recording became per-instrument.

So the layout is knowledge this module owns and the tools import.

The rule the whole thing exists to enforce: **never combine two instruments
without being told to.** Rows for two symbols are structurally identical -
same columns, same order, same dtypes - so a matrix built from both is one
that no schema check can object to, and a verdict drawn from it describes no
instrument in particular. Directory layout is the only thing that separates
them, which makes crossing a directory boundary the thing to refuse.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

# What an instrument directory looks like: a BloFin instId, `BASE-QUOTE`,
# uppercase, digits allowed (`1000BONK-USDT`). Matching the shape rather than
# keeping a deny-list of tool directories means `data/replayed`, `data/bars`,
# `data/cache`, `data/tardis` and anything added later are excluded for a
# reason that stays true, instead of until someone forgets to update the list.
#
# That includes `data/hyperliquid`, a different VENUE whose coin directories
# (`data/hyperliquid/BTC`) describe a different market from `data/BTC-USDT`
# in near-identical events. It sits one level down and does not match, so no
# BloFin tool can pick it up - see `trading/hyperliquid.py`.
INSTRUMENT_DIR = re.compile(r"[A-Z0-9]+-[A-Z0-9]+")


def instrument_dirs(data_dir: Path) -> Dict[str, Path]:
    """Every per-instrument directory under `data_dir`, by instrument id."""
    if not data_dir.is_dir():
        return {}
    return {
        child.name: child
        for child in sorted(data_dir.iterdir())
        if child.is_dir() and INSTRUMENT_DIR.fullmatch(child.name)
    }


def _ambiguous(data_dir: Path, found: Dict[str, Path], what: str,
               flag: str) -> "SystemExit":
    listing = "\n".join(f"  {name}" for name in found)
    example = next(iter(found))
    return SystemExit(
        f"{data_dir} holds {what} for more than one instrument:\n{listing}\n\n"
        "Combining them would produce a result about no instrument in "
        f"particular.\nName one: {flag} {example}"
    )


def resolve_raw_dir(data_dir: Path, instrument: Optional[str] = None) -> Path:
    """The raw archive directory to read.

    With `instrument`, that instrument's archive. Without it, the only one
    present - and a refusal if there is more than one, because replaying two
    instruments' events into a single feature file would interleave two
    unrelated order books into one time series.

    A bare `data/raw` is the pre-2026-09-09 layout, from before recording was
    per-instrument. It is accepted on its own, since everything written that
    way came from a single instrument.
    """
    if instrument:
        candidate = data_dir / instrument / "raw"
        if not candidate.is_dir():
            available = ", ".join(instrument_dirs(data_dir)) or "none"
            raise SystemExit(
                f"No raw archive at {candidate}.\n"
                f"Instruments with data under {data_dir}: {available}"
            )
        return candidate

    scoped = {name: path / "raw" for name, path in instrument_dirs(data_dir).items()}
    scoped = {name: path for name, path in scoped.items() if path.is_dir()}

    legacy = data_dir / "raw"
    if scoped and legacy.is_dir():
        raise SystemExit(
            f"{data_dir} holds both an unscoped raw/ and per-instrument "
            f"archives ({', '.join(scoped)}).\n"
            "The unscoped one records its symbol nowhere, so it cannot be "
            "placed automatically.\nMove it under the instrument that "
            "produced it, then re-run."
        )

    if len(scoped) > 1:
        raise _ambiguous(data_dir, scoped, "raw archives", "--instrument")

    if scoped:
        return next(iter(scoped.values()))

    if legacy.is_dir():
        return legacy

    raise SystemExit(
        f"No raw archive under {data_dir}. Expected {data_dir}/<INST-ID>/raw/.\n"
        "Run the recorder first: python backend/record.py"
    )


def carry_dir(data_dir: Path, instrument: str, *, create: bool = False) -> Path:
    """Where a live carry's baseline and snapshot history live.

    Under the instrument, for the reason every other path here is: a carry on
    SUI-USDT and a carry on ADA-USDT have identically shaped snapshots, and a
    realised-funding series assembled from both would describe neither.

    Separate from `raw/` because this is not recorded market data. It is the
    record of a position that exists, and it has to outlive the process that
    opened it - the whole point of a 30-day hold is that no single run of
    anything spans it.
    """
    path = data_dir / instrument / "carry"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
