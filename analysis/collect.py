"""
Utilities to consolidate per-condition CSV outputs into a single table.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


def _expand_paths(csv_paths: Sequence[Path] | Path | str) -> list[Path]:
    if isinstance(csv_paths, (str, Path)):
        base = Path(csv_paths)
        if base.is_dir():
            return sorted(p for p in base.rglob("*.csv") if p.is_file())
        return [base]
    return [Path(p) for p in csv_paths]


def collect_results(
    csv_paths: Sequence[Path] | Path | str,
    output_path: Path | str,
) -> None:
    """Merge CSV files into a single table with unified headers."""

    paths = _expand_paths(csv_paths)
    if not paths:
        raise ValueError("No CSV files found to collect.")

    union_header: list[str] = []
    rows: list[dict[str, str]] = []

    for path in paths:
        if not path.exists() or path == Path(output_path):
            continue
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                continue
            for name in reader.fieldnames:
                if name not in union_header:
                    union_header.append(name)
            for row in reader:
                rows.append(row)

    if not union_header:
        raise ValueError("No data rows were found in the provided CSV files.")

    key_fields = (
        "theta",
        "seed",
        "condition",
        "calibration",
        "model",
        "metric",
        "cutoff",
    )
    latest_rows: dict[tuple[str, ...], tuple[str, dict[str, str]]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        timestamp = row.get("timestamp", "")
        if key not in latest_rows:
            order.append(key)
            latest_rows[key] = (timestamp, row)
            continue
        prev_timestamp, _ = latest_rows[key]
        if timestamp and prev_timestamp:
            if timestamp >= prev_timestamp:
                latest_rows[key] = (timestamp, row)
        elif timestamp and not prev_timestamp:
            latest_rows[key] = (timestamp, row)
        elif not timestamp and not prev_timestamp:
            latest_rows[key] = (timestamp, row)

    deduped_rows = [latest_rows[key][1] for key in order]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=union_header)
        writer.writeheader()
        for row in deduped_rows:
            writer.writerow({field: row.get(field, "") for field in union_header})
