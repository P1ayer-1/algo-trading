"""Convert recorded feature CSVs to Parquet, and report storage usage.

    python backend/analysis/compact.py --report          # just show sizes
    python backend/analysis/compact.py                   # convert + report
    python backend/analysis/compact.py --delete-csv      # convert, remove CSVs

You do not need this yet. CSV plus numpy handles a few million rows fine, and
adding a dependency before it earns its place is how a research project turns
into an infrastructure project. Run it when loading starts to annoy you —
roughly past 10M rows, or when `data/` gets uncomfortably large.

What Parquet buys, once you're there:

  * Smaller than CSV — measured 2.4x on synthetic random data, which is the
    worst case for compression. Real recordings do better, because several
    columns barely change (spread, funding_rate, depth) and columnar encoding
    exploits exactly that. Don't expect the 10x figure often quoted for
    Parquet; that assumes far more repetitive data than this.
  * Reads only the columns you ask for. The predictiveness check touches ~20
    of 35 columns; CSV must parse every byte of every row regardless.
  * Preserves dtypes, so no re-parsing floats from text on every load.

Requires pyarrow (`pip install pyarrow`). If it isn't installed, this script
says so and does nothing rather than failing halfway through.

Querying afterwards
-------------------
DuckDB reads Parquet directly with SQL and needs no server:

    import duckdb
    duckdb.sql("SELECT AVG(fwd_ret_bps_5s) FROM 'data/parquet/*.parquet' "
               "WHERE obi_1 > 0.5").show()

That is the honest answer to "should I use a database": you get SQL over the
files you already have, with nothing to install, run, back up, or keep alive.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

# Columns that must stay integers/strings; everything else is a float.
INT_COLUMNS = {"ts", "received_ts", "trade_count_1s", "book_age_ms"}
STR_COLUMNS = {"vol_regime"}
BOOL_COLUMNS = {"is_valid"}


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:,.1f} PB"


def directory_size(path: Path, pattern: str = "**/*") -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.glob(pattern) if p.is_file())


def report(data_dir: Path) -> None:
    csv_bytes = directory_size(data_dir, "features-*.csv")
    raw_bytes = directory_size(data_dir / "raw", "**/*.jsonl.gz")
    parquet_bytes = directory_size(data_dir / "parquet", "*.parquet")
    replayed = directory_size(data_dir / "replayed", "features-*.csv")

    print("Storage:")
    print(f"  raw archive (gz)     {human(raw_bytes):>12}   <- irreplaceable, keep this")
    print(f"  feature CSV          {human(csv_bytes):>12}   <- regenerable via replay.py")
    print(f"  replayed CSV         {human(replayed):>12}")
    print(f"  parquet              {human(parquet_bytes):>12}")
    print(f"  TOTAL                {human(raw_bytes + csv_bytes + parquet_bytes + replayed):>12}")

    if csv_bytes and parquet_bytes:
        print(f"\n  parquet is {csv_bytes / parquet_bytes:.1f}x smaller than the CSV it replaced")

    days = len(list((data_dir / "raw").glob("*"))) if (data_dir / "raw").exists() else 0
    if days:
        per_day = (raw_bytes + csv_bytes) / days
        print(f"\n  {days} day(s) recorded, ~{human(per_day)}/day")
        print(f"  projected: {human(per_day * 30)}/month, {human(per_day * 365)}/year")


def convert_file(csv_path: Path, out_dir: Path) -> Optional[Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"  {csv_path.name}: empty, skipped")
        return None

    columns = list(rows[0].keys())
    arrays, names = [], []
    for column in columns:
        values = [row.get(column, "") for row in rows]
        if column in STR_COLUMNS:
            arrays.append(pa.array(values, type=pa.string()))
        elif column in BOOL_COLUMNS:
            arrays.append(pa.array([v in ("True", "true", "1") for v in values],
                                   type=pa.bool_()))
        elif column in INT_COLUMNS:
            arrays.append(pa.array([int(float(v)) if v else 0 for v in values],
                                   type=pa.int64()))
        else:
            arrays.append(pa.array([float(v) if v not in ("", None) else None
                                    for v in values], type=pa.float64()))
        names.append(column)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (csv_path.stem + ".parquet")
    # ZSTD beats snappy noticeably on this data and decompresses fast enough
    # that reads stay IO-bound rather than CPU-bound.
    pq.write_table(pa.Table.from_arrays(arrays, names=names), out_path,
                   compression="zstd")

    before, after = csv_path.stat().st_size, out_path.stat().st_size
    print(f"  {csv_path.name}: {human(before)} -> {human(after)}  "
          f"({before / max(after, 1):.1f}x smaller, {len(rows):,} rows)")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--report", action="store_true",
                        help="Show storage usage only; convert nothing.")
    parser.add_argument("--delete-csv", action="store_true",
                        help="Remove each CSV after it converts successfully.")
    args = parser.parse_args(argv)

    if args.report:
        report(args.data_dir)
        return 0

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is not installed, so nothing was converted.\n"
              "  pip install pyarrow\n"
              "Your CSVs are untouched. Note you probably don't need this yet — "
              "see the module docstring.")
        report(args.data_dir)
        return 1

    files = sorted(args.data_dir.glob("features-*.csv"))
    if not files:
        print(f"No features-*.csv in {args.data_dir}.")
        return 1

    print(f"Converting {len(files)} file(s) to Parquet:")
    converted = 0
    for path in files:
        try:
            if convert_file(path, args.data_dir / "parquet") is not None:
                converted += 1
                if args.delete_csv:
                    path.unlink()
                    print(f"    removed {path.name}")
        except Exception as exc:
            # Never delete a CSV whose conversion failed.
            print(f"  {path.name}: FAILED ({exc}) — left in place")

    print(f"\nConverted {converted}/{len(files)} file(s).\n")
    report(args.data_dir)
    print("\nQuery them with DuckDB (no server needed):")
    print("  import duckdb")
    print(f"  duckdb.sql(\"SELECT * FROM '{args.data_dir / 'parquet'}/*.parquet' LIMIT 5\").show()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
