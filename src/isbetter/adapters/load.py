"""Load the data an experiment config points at (the only place config meets the file system)."""

from __future__ import annotations

from pathlib import Path

from isbetter.adapters.ab_table import AggSpec, aggregate_ab_table, read_ab_table
from isbetter.adapters.csv_long import read_csv
from isbetter.adapters.llm_eval import read_item_scores
from isbetter.adapters.strategy_returns import (
    DeflationInputs,
    TrialRegistry,
    read_returns_csv,
    resolve_deflation,
    returns_table,
)
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
    if cfg.adapter == "llm_eval":
        return read_item_scores(path, dep, metric=plans[0].metric)
    if cfg.adapter == "strategy_returns":
        panel = read_returns_csv(path, time_col=cfg.table.time_col)
        series = {name: panel.series(name) for name in panel.names}
        return returns_table(series, panel.times, metric=plans[0].metric)
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


def load_deflation(cfg: Config, base_dir: Path) -> DeflationInputs | None:
    """Trial count and Sharpe variance for the Deflated Sharpe Ratio (``None`` without finance).

    Reads the trial registry named in ``finance.registry`` (relative to ``base_dir``). Raises
    ``MissingTrialsError`` (a ``ValueError``) if the registry is missing, empty or too short.
    """
    f = cfg.finance
    if f is None:
        return None
    registry = None
    if f.registry is not None:
        reg_path = Path(f.registry)
        reg_path = reg_path if reg_path.is_absolute() else base_dir / reg_path
        registry = TrialRegistry(reg_path)
        if not reg_path.exists():
            raise ValueError(f"trial registry {reg_path} does not exist")
    return resolve_deflation(registry, f.n_trials, f.sr_variance)
