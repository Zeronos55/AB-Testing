"""Canonical long-table contract and validation.

Every adapter produces a :class:`CanonicalTable`. Validation fails loudly with
:class:`SchemaError` so bad inputs never reach a statistical method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from isbetter.core.dependence import Clustered, CVOverlap, Dependence, Paired, TimeSeries

REQUIRED = ("unit_id", "variant", "metric", "value")


class SchemaError(ValueError):
    """Raised when a table violates the canonical data contract."""


@dataclass(frozen=True)
class CanonicalTable:
    """Column-oriented long table: one row per (unit, variant, metric) observation."""

    columns: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.columns["value"])

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def variants(self) -> list[str]:
        return sorted({str(v) for v in self.columns["variant"]})

    def metrics(self) -> list[str]:
        return sorted({str(v) for v in self.columns["metric"]})

    def select(self, variant: str, metric: str) -> dict[str, np.ndarray]:
        """Rows of one variant/metric, in original order."""
        mask = (self.columns["variant"].astype(str) == variant) & (
            self.columns["metric"].astype(str) == metric
        )
        return {k: v[mask] for k, v in self.columns.items()}


def build_table(
    columns: Mapping[str, Sequence[Any] | np.ndarray], dependence: Dependence | None = None
) -> CanonicalTable:
    """Convert raw columns to arrays, validate, and return a CanonicalTable."""
    cols = {k: np.asarray(v) for k, v in columns.items()}
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise SchemaError(f"missing required column(s): {', '.join(missing)}")
    lengths = {len(v) for v in cols.values()}
    if len(lengths) != 1:
        raise SchemaError("all columns must have the same length")
    try:
        cols["value"] = cols["value"].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise SchemaError("column 'value' must be numeric") from exc
    table = CanonicalTable(cols)
    validate(table, dependence)
    return table


def validate(table: CanonicalTable, dependence: Dependence | None = None) -> None:
    cols = table.columns
    if len(table) == 0:
        raise SchemaError("table is empty")
    if np.isnan(cols["value"]).any():
        raise SchemaError(f"NaN in 'value' ({int(np.isnan(cols['value']).sum())} rows)")
    if not np.isfinite(cols["value"]).all():
        raise SchemaError("infinite values in 'value'")

    variant = cols["variant"].astype(str)
    metric = cols["metric"].astype(str)

    if "pair_id" in cols:
        keys = list(zip(variant, metric, cols["pair_id"].astype(str), strict=True))
        if len(set(keys)) != len(keys):
            raise SchemaError("duplicate (variant, metric, pair_id) rows")

    if isinstance(dependence, Paired | CVOverlap):
        _check_pairing(cols, variant, metric)
    elif isinstance(dependence, Clustered):
        _check_clusters(cols, dependence, variant, metric)
        if dependence.paired:
            _check_pairing(cols, variant, metric)
            _check_pair_clusters(cols, dependence, variant, metric)
    elif isinstance(dependence, TimeSeries):
        _check_time(cols, variant, metric)


def _check_pairing(cols: dict[str, np.ndarray], variant: np.ndarray, metric: np.ndarray) -> None:
    if "pair_id" not in cols:
        raise SchemaError("dependence=paired requires a 'pair_id' column")
    pair = cols["pair_id"].astype(str)
    for m in np.unique(metric):
        in_m = metric == m
        variants = np.unique(variant[in_m])
        sets = {v: set(pair[in_m & (variant == v)]) for v in variants}
        union = set().union(*sets.values())
        for v, s in sets.items():
            gone = union - s
            if gone:
                raise SchemaError(
                    f"paired data: variant '{v}' (metric '{m}') is missing "
                    f"{len(gone)} partner(s), e.g. pair_id={sorted(gone)[0]!r}"
                )


def _check_clusters(
    cols: dict[str, np.ndarray], dep: Clustered, variant: np.ndarray, metric: np.ndarray
) -> None:
    name = dep.cluster_col
    if name not in cols:
        raise SchemaError(f"dependence=clustered requires a '{name}' column")
    cl = cols[name].astype(str)
    for v in np.unique(variant):
        n = len(set(cl[variant == v]))
        if n < 2:
            raise SchemaError(f"variant '{v}' has {n} cluster(s); need at least 2")


def _check_pair_clusters(
    cols: dict[str, np.ndarray], dep: Clustered, variant: np.ndarray, metric: np.ndarray
) -> None:
    """A paired item must sit in the same cluster for every variant."""
    pair, cl = cols["pair_id"].astype(str), cols[dep.cluster_col].astype(str)
    seen: dict[tuple[str, str], str] = {}
    for m, p, c in zip(metric, pair, cl, strict=True):
        if seen.setdefault((str(m), p), c) != c:
            raise SchemaError(
                f"paired item pair_id={p!r} (metric '{m}') is in more than one cluster "
                f"('{seen[(str(m), p)]}' and '{c}')"
            )


def _check_time(cols: dict[str, np.ndarray], variant: np.ndarray, metric: np.ndarray) -> None:
    if "time" not in cols:
        raise SchemaError("dependence=timeseries requires a 'time' column")
    t = cols["time"]
    for v in np.unique(variant):
        for m in np.unique(metric):
            tt = t[(variant == v) & (metric == m)]
            try:
                increasing = len(tt) < 2 or bool((np.diff(tt) > 0).all())
            except TypeError as exc:
                raise SchemaError(
                    "'time' must be numeric (convert dates to numbers first)"
                ) from exc
            if not increasing:
                raise SchemaError(
                    f"'time' must be strictly increasing within variant '{v}', metric '{m}'"
                )
