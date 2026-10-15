"""AnalysisPlan and the method router.

The router maps a declared :class:`Dependence` to the valid primary method (plus
cross-checks). It never silently falls back to a naive test: unsupported
combinations raise :class:`UnsupportedPlan`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isbetter.core.config import Config
from isbetter.core.dependence import (
    Clustered,
    Dependence,
    dependence_from_dict,
    dependence_to_dict,
)


class UnsupportedPlan(ValueError):
    """The requested dependence/metric combination has no valid method yet."""


# dependence kind -> (primary method, cross-checks), for continuous metrics
ROUTES: dict[str, tuple[str, tuple[str, ...]]] = {
    "iid": ("welch_t", ()),
    "paired": ("paired_t", ("paired_bootstrap", "sign_flip_permutation")),
    "clustered": ("cluster_robust_diff", ("cluster_bootstrap",)),
    "cv_overlap": ("nadeau_bengio", ()),
    "timeseries": ("diebold_mariano_hln", ("stationary_block_bootstrap",)),
}

# paired right/wrong scores (small-N safe), and paired items grouped in clusters
PAIRED_BINARY_ROUTE: tuple[str, tuple[str, ...]] = (
    "paired_binary_score",
    ("paired_beta_posterior",),
)
PAIRED_CLUSTERED_ROUTE: tuple[str, tuple[str, ...]] = (
    "paired_cluster_robust",
    ("paired_cluster_bootstrap",),
)

# per-period returns of strategies (dependence=timeseries): the Sharpe ratio difference versus the
# baseline, with the overfitting guards reported alongside
SHARPE_ROUTE: tuple[str, tuple[str, ...]] = (
    "sharpe_block_bootstrap",
    ("probabilistic_sharpe", "deflated_sharpe", "min_track_record_length"),
)

# dependence=iid routes by (metric kind, variance reduction)
IID_ROUTES: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {
    ("continuous", "none"): ("welch_t", ()),
    ("continuous", "cuped"): ("cuped", ("welch_t",)),
    ("continuous", "regression"): ("regression_adjustment", ("welch_t",)),
    ("binary", "none"): ("two_proportion_z", ("fisher_exact",)),
    ("binary", "cuped"): ("cuped", ("two_proportion_z",)),
    ("binary", "regression"): ("regression_adjustment", ("two_proportion_z",)),
    ("ratio", "none"): ("delta_method", ("ratio_bootstrap",)),
}


@dataclass
class AnalysisPlan:
    experiment: str
    dependence: Dependence
    baseline: str
    candidates: list[str]
    metric: str
    higher_is_better: bool
    alpha: float
    min_effect: float
    correction: str
    seed: int
    primary: str
    cross_checks: list[str] = field(default_factory=list)
    metric_kind: str = "continuous"
    denominator: str | None = None
    variance_reduction: str = "none"
    covariate: str | None = None
    role: str = "success"
    finance: dict[str, Any] | None = None  # overfitting-guard options (kind: returns)

    def pretty(self) -> str:
        dep = dependence_to_dict(self.dependence)
        lines = [
            f"Analysis plan: {self.experiment}",
            f"  dependence : {dep}",
            f"  compare    : {', '.join(self.candidates)} vs baseline '{self.baseline}'",
            f"  metric     : {self._metric_text()}",
            f"  primary    : {self.primary}",
            f"  cross-check: {', '.join(self.cross_checks) or 'none'}",
            f"  alpha={self.alpha}  min_effect={self.min_effect}  "
            f"correction={self.correction}  seed={self.seed}",
        ]
        if self.variance_reduction != "none":
            lines.insert(
                4, f"  adjustment : {self.variance_reduction} on covariate '{self.covariate}'"
            )
        if self.finance is not None:
            f = self.finance
            where = f"registry {f['registry']}" if f.get("registry") else "explicit n_trials"
            lines.append(
                f"  guards     : DSR from {where}; PBO over {f['pbo_blocks']} blocks "
                f"({f['pbo_metric']}); Sharpe ratios are per period, not annualised"
            )
        return "\n".join(lines)

    def _metric_text(self) -> str:
        direction = "higher" if self.higher_is_better else "lower"
        kind = ""
        if self.metric_kind == "ratio":
            kind = f", ratio over '{self.denominator}'"
        elif self.metric_kind != "continuous":
            kind = f", {self.metric_kind}"
        role = "" if self.role == "success" else f", {self.role}"
        return f"{self.metric} ({direction} is better{kind}{role})"

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["dependence"] = dependence_to_dict(self.dependence)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnalysisPlan:
        d = dict(d)
        d["dependence"] = dependence_from_dict(d["dependence"])
        return cls(**d)


def route(
    dep: Dependence, kind: str = "continuous", variance_reduction: str = "none"
) -> tuple[str, tuple[str, ...]]:
    """Primary method and cross-checks for a dependence / metric kind / adjustment."""
    if kind == "returns":
        if dep.kind != "timeseries" or variance_reduction != "none":
            raise UnsupportedPlan(
                "metric kind 'returns' compares strategies' Sharpe ratios and needs "
                f"dependence 'timeseries' without variance reduction (got '{dep.kind}'). "
                "Refusing to fall back to a different test."
            )
        return SHARPE_ROUTE
    if dep.kind == "iid":
        try:
            return IID_ROUTES[(kind, variance_reduction)]
        except KeyError:
            raise UnsupportedPlan(
                f"no method for metric kind '{kind}' with variance_reduction "
                f"'{variance_reduction}'. Refusing to fall back to a different test."
            ) from None
    if dep.kind in ROUTES:
        if kind == "ratio" or variance_reduction != "none":
            raise UnsupportedPlan(
                f"dependence '{dep.kind}' supports neither ratio metrics nor variance reduction "
                "yet. Refusing to drop that part of the analysis silently."
            )
        if dep.kind == "timeseries" and kind != "continuous":
            raise UnsupportedPlan(
                f"dependence 'timeseries' compares per-origin losses (continuous); metric kind "
                f"'{kind}' is not routable. Refusing to fall back to a naive test."
            )
        if kind == "binary" and dep.kind == "cv_overlap":
            raise UnsupportedPlan(
                "binary metrics under dependence 'cv_overlap' need a paired test on item "
                "correctness (isbetter.methods.classification.mcnemar); not routable from config."
            )
        if dep.kind == "paired" and kind == "binary":
            return PAIRED_BINARY_ROUTE
        if isinstance(dep, Clustered) and dep.paired:
            return PAIRED_CLUSTERED_ROUTE
        return ROUTES[dep.kind]
    raise UnsupportedPlan(
        f"dependence '{dep.kind}' is not implemented yet. "
        "Refusing to fall back to a naive test that would give invalid inference."
    )


def build_plans(cfg: Config) -> list[AnalysisPlan]:
    """One plan per configured metric (each routed on its own kind and adjustment)."""
    dep = cfg.to_dependence()
    plans = []
    for m in cfg.metric_list:
        primary, checks = route(dep, m.kind, m.variance_reduction)
        success = m.role == "success"
        plans.append(
            AnalysisPlan(
                experiment=cfg.experiment,
                dependence=dep,
                baseline=cfg.variants.baseline,
                candidates=list(cfg.variants.candidates),
                metric=m.name,
                higher_is_better=m.higher_is_better,
                alpha=cfg.decision.alpha,
                min_effect=cfg.decision.min_effect if m.min_effect is None else m.min_effect,
                correction=cfg.decision.correction if success else "none",
                seed=cfg.seed,
                primary=primary,
                cross_checks=list(checks),
                metric_kind=m.kind,
                denominator=m.denominator,
                variance_reduction=m.variance_reduction,
                covariate=m.covariate,
                role=m.role,
                finance=cfg.finance.model_dump() if cfg.finance is not None else None,
            )
        )
    return plans


def build_plan(cfg: Config) -> AnalysisPlan:
    """The plan of a single-metric config (use :func:`build_plans` for several metrics)."""
    plans = build_plans(cfg)
    if len(plans) != 1:
        raise ValueError(
            f"config has {len(plans)} metrics; use build_plans / run_experiment for several"
        )
    return plans[0]
