"""A/B-test adapter: one row per randomisation unit, one column per metric.

Two paths share one file format (CSV or Parquet):

* :func:`read_ab_table`: unit-level. Reshapes the wide table into the canonical long table
  (polars, selected columns only). Needed for bootstrap/permutation cross-checks and for
  clustered analyses.
* :func:`aggregate_ab_table`: streams the file through DuckDB and keeps only
  :class:`~isbetter.core.sufficient.SufficientStats` per arm. Memory does not grow with the
  number of rows, so a 14-million-row table (Criteo) fits in well under 1 GB.

Both fail loudly (``SchemaError``) on missing columns, nulls, non-finite values and arms
that the config does not name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isbetter.core.dependence import Dependence
from isbetter.core.schema import CanonicalTable, SchemaError, build_table
from isbetter.core.sufficient import AggregatedTable, SufficientStats


@dataclass(frozen=True)
class AggSpec:
    """One metric to aggregate: outcome column ``y_col`` and an optional paired ``x_col``.

    ``x_col`` is the covariate (CUPED / regression adjustment) or the denominator column of a
    ratio metric. ``name`` is the key the runner looks the statistics up under.
    """

    name: str
    y_col: str
    x_col: str | None = None


def _is_parquet(path: Path) -> bool:
    return path.suffix.lower() in (".parquet", ".pq")


def _quote(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def read_ab_table(
    path: str | Path,
    *,
    variant_col: str,
    value_cols: Sequence[str],
    covariate_cols: Sequence[str] = (),
    unit_col: str | None = None,
    cluster_col: str | None = None,
    cluster_name: str = "cluster_id",
    dependence: Dependence | None = None,
) -> CanonicalTable:
    """Load a wide A/B table as a validated canonical long table (unit-level path).

    Each name in ``value_cols`` becomes a metric; each name in ``covariate_cols`` becomes a
    ``cov_<name>`` column. ``unit_col`` defaults to the row number. ``cluster_col`` (e.g. the
    user id when several rows belong to one randomised user) is written under ``cluster_name``.
    """
    import polars as pl

    p = Path(path)
    wanted = [variant_col, *value_cols, *covariate_cols]
    if unit_col:
        wanted.append(unit_col)
    if cluster_col:
        wanted.append(cluster_col)
    wanted = list(dict.fromkeys(wanted))
    try:
        df = (
            pl.read_parquet(p, columns=wanted)
            if _is_parquet(p)
            else pl.read_csv(p, columns=wanted, infer_schema_length=10_000)
        )
    except pl.exceptions.ColumnNotFoundError as exc:
        raise SchemaError(f"column missing from {p.name}: {exc}") from exc
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise SchemaError(f"cannot read data file {p}: {exc}") from exc

    nulls = {c: n for c, n in zip(df.columns, df.null_count().row(0), strict=True) if n}
    if nulls:
        raise SchemaError(f"missing values in {p.name}: {nulls}")

    n = df.height
    if n == 0:
        raise SchemaError(f"{p.name} has no rows")
    variant = df[variant_col].cast(pl.Utf8).to_numpy()
    unit = df[unit_col].cast(pl.Utf8).to_numpy() if unit_col else np.arange(n).astype(str)

    def numeric(col: str) -> np.ndarray:
        try:
            return df[col].cast(pl.Float64).to_numpy()
        except pl.exceptions.PolarsError as exc:
            raise SchemaError(f"column '{col}' must be numeric") from exc

    k = len(value_cols)
    cols: dict[str, np.ndarray] = {
        "unit_id": np.tile(unit, k),
        "variant": np.tile(variant, k),
        "metric": np.repeat(np.array(list(value_cols)), n),
        "value": np.concatenate([numeric(c) for c in value_cols]),
    }
    for c in covariate_cols:
        cols[f"cov_{c}"] = np.tile(numeric(c), k)
    if cluster_col:
        cols[cluster_name] = np.tile(df[cluster_col].cast(pl.Utf8).to_numpy(), k)
    return build_table(cols, dependence)


def aggregate_ab_table(
    path: str | Path,
    *,
    variant_col: str,
    specs: Sequence[AggSpec],
    variants: Sequence[str],
    memory_limit: str = "512MB",
) -> AggregatedTable:
    """Stream the file through DuckDB and return sufficient statistics per (arm, metric).

    One pass computes ``n``, ``sum``, ``sum_sq`` (and the x moments where ``x_col`` is given)
    for every metric. Floating-point sums use Kahan summation. ``variants`` lists the arms the
    analysis names; any other arm in the file is an error rather than silently dropped.
    """
    import duckdb

    p = Path(path)
    if not p.exists():
        raise SchemaError(f"cannot read data file {p}: no such file")
    source = "read_parquet(?)" if _is_parquet(p) else "read_csv(?, header=true)"

    # Distinct physical columns -> DOUBLE aliases c0, c1, ... so each is cast exactly once.
    phys: dict[str, str] = {}
    for s in specs:
        for col in (s.y_col, s.x_col):
            if col is not None and col not in phys:
                phys[col] = f"c{len(phys)}"
    select = [f"CAST({_quote(variant_col)} AS VARCHAR) AS v"]
    select += [f"CAST({_quote(col)} AS DOUBLE) AS {alias}" for col, alias in phys.items()]
    agg_exprs: dict[str, str] = {"n": "count(*)"}
    for alias in phys.values():
        agg_exprs[f"nn_{alias}"] = f"count({alias})"
        agg_exprs[f"bad_{alias}"] = (
            f"count(*) FILTER (WHERE {alias} IS NOT NULL AND NOT isfinite({alias}))"
        )
        agg_exprs[f"s_{alias}"] = f"fsum({alias})"
        agg_exprs[f"ss_{alias}"] = f"fsum({alias} * {alias})"
    for s in specs:
        if s.x_col is not None:
            ya, xa = phys[s.y_col], phys[s.x_col]
            agg_exprs[f"xy_{ya}_{xa}"] = f"fsum({ya} * {xa})"
    sql = (
        f"SELECT v, {', '.join(f'{e} AS {k}' for k, e in agg_exprs.items())} "
        f"FROM (SELECT {', '.join(select)} FROM {source}) GROUP BY v"
    )

    con = duckdb.connect(":memory:")
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute("SET preserve_insertion_order=false")
        try:
            cur = con.execute(sql, [str(p)])
            names = [d[0] for d in cur.description]
            rows = [dict(zip(names, r, strict=True)) for r in cur.fetchall()]
        except duckdb.Error as exc:
            raise SchemaError(f"cannot aggregate {p.name}: {exc}") from exc
    finally:
        con.close()

    if not rows:
        raise SchemaError(f"{p.name} has no rows")
    by_arm = {str(r["v"]): r for r in rows}
    extra = sorted(set(by_arm) - set(variants))
    if extra:
        raise SchemaError(
            f"{p.name} contains arm(s) not named in the config: {extra}; "
            "list them under variants or filter the file"
        )
    missing = sorted(set(variants) - set(by_arm))
    if missing:
        raise SchemaError(f"arm(s) named in the config have no rows in {p.name}: {missing}")

    out: dict[tuple[str, str], SufficientStats] = {}
    units: dict[str, int] = {}
    for arm, r in by_arm.items():
        n = int(r["n"])
        units[arm] = n
        for col, alias in phys.items():
            if int(r[f"nn_{alias}"]) != n:
                gone = n - int(r[f"nn_{alias}"])
                raise SchemaError(f"column '{col}' has {gone} missing value(s) in arm '{arm}'")
            if int(r[f"bad_{alias}"]):
                raise SchemaError(f"column '{col}' has non-finite values in arm '{arm}'")
        for s in specs:
            ya = phys[s.y_col]
            try:
                if s.x_col is None:
                    out[(arm, s.name)] = SufficientStats(
                        n, float(r[f"s_{ya}"]), float(r[f"ss_{ya}"])
                    )
                else:
                    xa = phys[s.x_col]
                    out[(arm, s.name)] = SufficientStats(
                        n,
                        float(r[f"s_{ya}"]),
                        float(r[f"ss_{ya}"]),
                        float(r[f"s_{xa}"]),
                        float(r[f"ss_{xa}"]),
                        float(r[f"xy_{ya}_{xa}"]),
                    )
            except ValueError as exc:
                raise SchemaError(f"bad moments for metric '{s.name}', arm '{arm}': {exc}") from exc
    return AggregatedTable(stats=out, units=units)
