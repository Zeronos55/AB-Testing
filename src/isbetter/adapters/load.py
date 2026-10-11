"""Load the data an experiment config points at (the only place config meets the file system)."""

from __future__ import annotations

from pathlib import Path

from isbetter.adapters.ab_table import AggSpec, aggregate_ab_table, read_ab_table
from isbetter.adapters.csv_long import read_csv
from isbetter.core.config import Config
from isbetter.core.dependence import Clustered
from isbetter.core.plan import AnalysisPlan
from isbetter.core.schema import CanonicalTable
from isbetter.core.sufficient import AggregatedTable


def resolve_data_path(cfg: Config, base_dir: Path) -> Path:
    data = Path(cfg.data)
    return data if data.is_absolute() else base_dir / data


def load_data(
    cfg: Config, plans: list[AnalysisPlan], base_dir: Path
) -> CanonicalTable | AggregatedTable:
    """Read ``cfg.data`` through the configured adapter (relative paths resolve to base_dir)."""
    path = resolve_data_path(cfg, base_dir)
    dep = plans[0].dependence
    if cfg.adapter != "ab_table":
        return read_csv(path, dep)

    variant_col = cfg.table.variant_col
    assert variant_col is not None  # enforced by Config validation
    arms = [cfg.variants.baseline, *cfg.variants.candidates]
    if cfg.table.mode == "stats":
        specs = []
        for p in plans:
            x = p.denominator if p.metric_kind == "ratio" else p.covariate
            specs.append(AggSpec(name=p.metric, y_col=p.metric, x_col=x))
        return aggregate_ab_table(path, variant_col=variant_col, specs=specs, variants=arms)

    values = list(
        dict.fromkeys([p.metric for p in plans] + [p.denominator for p in plans if p.denominator])
    )
    covariates = list(dict.fromkeys(p.covariate for p in plans if p.covariate))
    return read_ab_table(
        path,
        variant_col=variant_col,
        value_cols=values,
        covariate_cols=covariates,
        unit_col=cfg.table.unit_col,
        cluster_col=cfg.table.cluster_col,
        cluster_name=dep.cluster_col if isinstance(dep, Clustered) else "cluster_id",
        dependence=dep,
    )
