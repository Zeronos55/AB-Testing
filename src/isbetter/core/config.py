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
    kind: Literal["continuous", "binary", "ratio", "returns"] = "continuous"
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
    paired: bool = False


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


class SequentialCfg(_Strict):
    """Sequential monitoring: an interval and p-value that stay valid at every look.

    ``n_target`` (the sample size per arm you plan to reach) or ``rho`` must be fixed before
    looking at the data. ``sigma`` is a known standard deviation, or a bound on it (0.5 for scores
    in [0, 1]); without it the running standard deviation is used, which is asymptotic and needs
    ``min_n`` observations before the first look. Rows must carry a ``time`` column: arrival order.
    """

    n_target: float | None = Field(default=None, gt=0)
    rho: float | None = Field(default=None, gt=0)
    sigma: float | None = Field(default=None, gt=0)
    min_n: int = Field(default=20, ge=2)

    @model_validator(mode="after")
    def _one_tuning(self) -> SequentialCfg:
        if (self.n_target is None) == (self.rho is None):
            raise ValueError(
                "sequential needs exactly one of `n_target` (planned sample size per arm) or "
                "`rho`, fixed before looking at the data; the length of the data you already "
                "have is not a valid choice"
            )
        return self


class FinanceCfg(_Strict):
    """Overfitting guards for a Sharpe-ratio comparison (``metric.kind: returns``).

    The Deflated Sharpe Ratio must know how many strategy variants were tried, so either a
    ``registry`` (a TrialRegistry JSON Lines file) or an explicit ``n_trials`` with its
    ``sr_variance`` (variance of the trials' per-period Sharpe ratios) is required. There is no
    default trial count: assuming one would hide the selection bias.
    """

    registry: str | None = None
    n_trials: int | None = Field(default=None, ge=1)
    sr_variance: float | None = Field(default=None, ge=0)
    sr_benchmark: float = 0.0
    pbo_blocks: int = Field(default=16, ge=4)
    pbo_metric: Literal["sharpe", "mean"] = "sharpe"

    @model_validator(mode="after")
    def _consistent(self) -> FinanceCfg:
        if self.registry is None and self.n_trials is None:
            raise ValueError(
                "finance needs `registry` (a trial log) or `n_trials`: the Deflated Sharpe Ratio "
                "cannot be computed without knowing how many variants were tried"
            )
        if self.registry is None and self.sr_variance is None:
            raise ValueError(
                "finance.n_trials without a registry also needs `sr_variance`, the variance of "
                "the trials' per-period Sharpe ratios"
            )
        if self.pbo_blocks % 2:
            raise ValueError("finance.pbo_blocks must be even (half the blocks are in-sample)")
        return self


class TableCfg(_Strict):
    """File layout for adapter ``ab_table`` (one row per unit, one column per metric) and for
    adapter ``strategy_returns`` (a wide CSV: a ``time_col`` plus one column per strategy)."""

    variant_col: str | None = None
    time_col: str = "date"
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
    finance: FinanceCfg | None = None
    sequential: SequentialCfg | None = None
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
        returns = [m.name for m in self.metric_list if m.kind == "returns"]
        if returns:
            if self.metrics is not None:
                raise ValueError("metric kind 'returns' needs the single `metric` form")
            if not isinstance(self.dependence, TimeSeriesCfg):
                raise ValueError("metric kind 'returns' needs `dependence: {type: timeseries}`")
            if self.finance is None:
                raise ValueError(
                    "metric kind 'returns' needs a `finance` block (trial registry or n_trials "
                    "for the Deflated Sharpe Ratio)"
                )
            if self.decision.correction != "none":
                raise ValueError(
                    "decision.correction must be 'none' for returns: selection across variants is "
                    "handled by the Deflated Sharpe Ratio and PBO (finance block), not by Holm/BH"
                )
        elif self.finance is not None:
            raise ValueError("`finance` is only valid with metric kind 'returns'")
        if self.sequential is not None:
            if self.is_experiment:
                raise ValueError(
                    "`sequential` is for single-metric comparisons (adapter csv_long or "
                    "llm_eval), not A/B experiments with SRM and guardrails yet"
                )
            if not isinstance(self.dependence, IidCfg | PairedCfg):
                raise ValueError("`sequential` supports dependence iid and paired only")
            m = self.metric_list[0]
            if m.kind not in ("continuous", "binary") or m.variance_reduction != "none":
                raise ValueError(
                    "`sequential` supports continuous and binary metrics without variance reduction"
                )
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
            return Clustered(d.cluster_col, d.paired)
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
