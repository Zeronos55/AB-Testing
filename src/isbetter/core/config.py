"""YAML config model (pydantic) and conversion to :class:`Dependence` objects."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from isbetter.core.dependence import Clustered, CVOverlap, Dependence, Iid, Paired, TimeSeries


class ConfigError(ValueError):
    """Raised for unreadable or invalid configs, with a human-readable message."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VariantsCfg(_Strict):
    baseline: str
    candidates: list[str] = Field(min_length=1)


class MetricCfg(_Strict):
    name: str
    higher_is_better: bool = True
    kind: Literal["continuous"] = "continuous"


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


class ReportCfg(_Strict):
    format: list[Literal["markdown", "html"]] = ["markdown"]
    include_simulation_check: bool = False


class Config(_Strict):
    experiment: str
    adapter: str = "csv_long"
    data: str
    variants: VariantsCfg
    metric: MetricCfg
    dependence: DependenceCfg
    decision: DecisionCfg = DecisionCfg()
    seed: int = 0
    report: ReportCfg = ReportCfg()

    @model_validator(mode="after")
    def _distinct_variants(self) -> Config:
        if self.variants.baseline in self.variants.candidates:
            raise ValueError("baseline must not also be a candidate")
        return self

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
