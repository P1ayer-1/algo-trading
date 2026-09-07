"""Parquet compaction: correctness of the round trip, and safety of --delete-csv.

The dangerous operation here is deleting a CSV after conversion. These tests
pin that a failed conversion never destroys the source.
"""

import csv
from pathlib import Path

import pytest

from analysis.compact import convert_file, human, main, report

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow not installed")
import pyarrow.parquet as pq  # noqa: E402


def write_sample(path: Path, rows=200):
    columns = ["ts", "received_ts", "obi_1", "spread_bps", "vol_regime",
               "is_valid", "trade_count_1s", "fwd_ret_bps_5s", "label_5s"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(rows):
            writer.writerow({
                "ts": 1_700_000_000_000 + index * 250,
                "received_ts": 1_700_000_000_005 + index * 250,
                "obi_1": round(index / rows - 0.5, 6),
                "spread_bps": 0.05,
                "vol_regime": "normal_vol",
                "is_valid": "True",
                "trade_count_1s": index % 20,
                "fwd_ret_bps_5s": round((index % 7) - 3, 4),
                "label_5s": (index % 3) - 1,
            })
    return path


def test_conversion_preserves_every_value(tmp_path):
    source = write_sample(tmp_path / "features-2026-01-01.csv")
    out = convert_file(source, tmp_path / "parquet")
    table = pq.read_table(out)

    original = list(csv.DictReader(source.open()))
    assert table.num_rows == len(original)

    parquet_rows = table.to_pylist()
    for before, after in zip(original, parquet_rows):
        assert after["ts"] == int(before["ts"])
        assert after["obi_1"] == pytest.approx(float(before["obi_1"]))
        assert after["vol_regime"] == before["vol_regime"]
        assert after["is_valid"] is True
        assert after["trade_count_1s"] == int(before["trade_count_1s"])


def test_dtypes_are_preserved_not_stringified(tmp_path):
    source = write_sample(tmp_path / "features-2026-01-01.csv")
    schema = pq.read_table(convert_file(source, tmp_path / "parquet")).schema
    assert schema.field("ts").type == pyarrow.int64()
    assert schema.field("obi_1").type == pyarrow.float64()
    assert schema.field("vol_regime").type == pyarrow.string()
    assert schema.field("is_valid").type == pyarrow.bool_()


def test_empty_csv_is_skipped_not_crashed(tmp_path):
    path = tmp_path / "features-empty.csv"
    path.write_text("ts,obi_1\n", encoding="utf-8")
    assert convert_file(path, tmp_path / "parquet") is None


def test_delete_csv_removes_source_only_after_success(tmp_path):
    write_sample(tmp_path / "features-2026-01-01.csv")
    main(["--data-dir", str(tmp_path), "--delete-csv"])
    assert not (tmp_path / "features-2026-01-01.csv").exists()
    assert (tmp_path / "parquet" / "features-2026-01-01.parquet").exists()


def test_failed_conversion_never_deletes_the_csv(tmp_path, monkeypatch, capsys):
    """The one operation that can lose data must fail safe."""
    source = write_sample(tmp_path / "features-2026-01-01.csv")

    import analysis.compact as compact

    def boom(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(compact, "convert_file", boom)
    compact.main(["--data-dir", str(tmp_path), "--delete-csv"])

    assert source.exists(), "a CSV whose conversion failed must be kept"
    assert "FAILED" in capsys.readouterr().out


def test_report_runs_on_an_empty_directory(tmp_path, capsys):
    report(tmp_path)
    assert "TOTAL" in capsys.readouterr().out


def test_human_readable_sizes():
    assert human(512) == "512.0 B"
    assert human(2048) == "2.0 KB"
    assert human(5 * 1024**3) == "5.0 GB"
