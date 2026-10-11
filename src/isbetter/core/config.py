"""YAML config model (pydantic) and conversion to :class:`Dependence` objects."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from isbetter.core.dependence import Clustered, CVOverlap, Dependence, Iid, Paired, TimeSeries


class ConfigError(ValueError):
    """Raised for unreadable or invalid configs, with a human-readable message."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VariantsCfg(_Strict):
    baseline: str
    candidates: list[str] = Field(min_length=1)

    @field_validator("baseline", mode="before")
    @classmethod
    def _baseline_str(cls, v: object) -> object:
        return str(v) if isinstance(v, int) and not isinstance(v, bool) else v

    @field_validator("candidates", mode="before")
    @classmethod
    def _candidates_str(cls, v: object) -> object:
        # YAML reads `treatment: 1` as an int; arms are labels, so accept numbers as labels.
        if isinstance(v, list):
            return [str(x) if isinstance(x, int) and not isinstance(x, bool) else x for x in v]
        return v


class MetricCfg(_Strict):
    """One metric. For a guardrail, ``min_effect`` is the non-inferiority margin (> 0)."""

    name: str
    higher_is_better: bool = True
    kind: Literal["continuous", "binary", "ratio"] = "continuous"
    denominator: str | None = None
    role: Literal["success", "guardrail"] = "success"
    min_effect: float | None = Field(default=None, ge=0)
    variance_reduction: Literal["none", "cuped", "regression"] = "none"
    covariate: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> MetricCfg:
        if self.kind == "ratio" and not self.denominator:
            raise ValueError("a ratio metric needs `denominator` (the denominator column)")
        if self.kind != "ratio" and self.denominator:
            raise ValueError("`denominator` is only valid for kind: ratio")
        if self.variance_reduction != "none" and not self.covariate:
            raise ValueError(f"variance_reduction: {self.variance_reduction} needs `covariate`")
        if self.variance_reduction == "none" and self.covariate:
            raise ValueError("`covariate` is set but variance_reduction is none")
        if self.kind == "ratio" and self.variance_reduction != "none":
            raise ValueError("variance reduction is not supported for ratio metrics")
        if self.role == "guardrail" and not self.min_effect:
            raise ValueError(
                f"guardrail '{self.name}' needs min_effect > 0 (its non-inferiority margin)"
            )
        return self


class IidCfg(_Strict):
    type: Literal["iid"]


class PairedCfg(_Strict):
    type: Literal["paired"]


class ClusteredCfg(_Strict):
    type: Literal["clustered"]
    cluster_col: str = "cluster_id"


class CVOverlapCfg(_Strict):
    type: Literal["cv_overlap"]
    n_train: int = Field(gt=0)
    n_test: int = Field(gt=0)
    repeats: int = Field(gt=0)
    folds: int = Field(gt=1)


class TimeSeriesCfg(_Strict):
    type: Literal["timeseries"]
    block_len: int | None = Field(default=None, gt=0)
    hac_lags: int | None = Field(default=None, ge=0)
    horizon: int = Field(default=1, gt=0)


DependenceCfg = Annotated[
    IidCfg | PairedCfg | ClusteredCfg | CVOverlapCfg | TimeSeriesCfg,
    Field(discriminator="type"),
]


class DecisionCfg(_Strict):
    min_effect: float = Field(default=0.0, ge=0)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    correction: Literal["none", "holm", "bonferroni", "bh"] = "none"
    ship_requires_practical: bool = False


class TableCfg(_Strict):
    """File layout for adapter ``ab_table``: one row per unit, one column per metric."""

    variant_col: str | None = None
    unit_col: str | None = None
    cluster_col: str | None = None
    mode: Literal["unit", "stats"] = "unit"


class DesignCfg(_Strict):
    """What randomisation was supposed to deliver (for the sample-ratio check)."""

    expected_split: dict[str, float] | None = None
    srm_alpha: float = Field(default=0.001, gt=0, lt=1)

    @field_validator("expected_split", mode="before")
    @classmethod
    def _split_keys_str(cls, v: object) -> object:
        if isinstance(v, dict):
            return {str(k): x for k, x in v.items()}
        return v


class ReportCfg(_Strict):
    format: list[Literal["markdown", "html"]] = ["markdown"]
    include_simulation_check: bool = False


class Config(_Strict):
    experiment: str
    adapter: str = "csv_long"
    data: str
    variants: VariantsCfg
    metric: MetricCfg | None = None
    metrics: list[MetricCfg] | None = Field(default=None, min_length=1)
    dependence: DependenceCfg
    decision: DecisionCfg = DecisionCfg()
    table: TableCfg = TableCfg()
    design: DesignCfg = DesignCfg()
    seed: int = 0
    report: ReportCfg = ReportCfg()

    @model_validator(mode="after")
    def _consistent(self) -> Config:
        if self.variants.baseline in self.variants.candidates:
            raise ValueError("baseline must not also be a candidate")
        if (self.metric is None) == (self.metrics is None):
            raise ValueError("give exactly one of `metric` (single) or `metrics` (list)")
        names = [m.name for m in self.metric_list]
        if len(set(names)) != len(names):
            raise ValueError("metric names must be unique")
        if self.adapter == "ab_table":
            if not self.table.variant_col:
                raise ValueError("adapter ab_table needs table.variant_col (the arm column)")
            if self.table.mode == "stats" and not isinstance(self.dependence, IidCfg):
                raise ValueError("table.mode: stats (aggregated data) supports dependence iid only")
        split = self.design.expected_split
        if split is not None:
            arms = {self.variants.baseline, *self.variants.candidates}
            if set(split) != arms:
                raise ValueError(f"design.expected_split must list exactly the arms {sorted(arms)}")
        return self

    @property
    def metric_list(self) -> list[MetricCfg]:
        if self.metrics is not None:
            return self.metrics
        assert self.metric is not None
        return [self.metric]

    @property
    def is_experiment(self) -> bool:
        """A/B experiments (adapter ab_table or a `metrics` list) get SRM and decision rules."""
        return self.adapter == "ab_table" or self.metrics is not None

    def to_dependence(self) -> Dependence:
        d = self.dependence
        if isinstance(d, IidCfg):
            return Iid()
        if isinstance(d, PairedCfg):
            return Paired()
        if isinstance(d, ClusteredCfg):
            return Clustered(d.cluster_col)
        if isinstance(d, CVOverlapCfg):
            return CVOverlap(d.n_train, d.n_test, d.repeats, d.folds)
        return TimeSeries(d.block_len, d.hac_lags, d.horizon)


def parse_config(raw: object) -> Config:
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        lines = [f"  {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        raise ConfigError("invalid config:\n" + "\n".join(lines)) from exc


def load_config(path: str | Path) -> Config:
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    return parse_config(raw)
