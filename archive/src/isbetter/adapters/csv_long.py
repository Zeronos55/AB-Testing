"""Read a canonical long-format CSV into a validated CanonicalTable."""

from __future__ import annotations

import csv
from pathlib import Path

from isbetter.core.dependence import Dependence
from isbetter.core.schema import CanonicalTable, SchemaError, build_table


def read_csv(path: str | Path, dependence: Dependence | None = None) -> CanonicalTable:
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            names = reader.fieldnames or []
            cols: dict[str, list[str]] = {n: [] for n in names}
            for row in reader:
                for n in names:
                    cols[n].append(row[n])
    except OSError as exc:
        raise SchemaError(f"cannot read data file {path}: {exc}") from exc
    for name in ("time", "fold", "repeat"):
        if name in cols:
            try:
                cols[name] = [float(v) for v in cols[name]]  # type: ignore[misc]
            except ValueError:
                pass  # keep as strings (e.g. ISO dates sort correctly)
    return build_table(cols, dependence)
