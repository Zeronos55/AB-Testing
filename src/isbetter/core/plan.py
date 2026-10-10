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
    Dependence,
    dependence_from_dict,
    dependence_to_dict,
)


class UnsupportedPlan(ValueError):
    """The requested dependence/metric combination has no valid method yet."""


# dependence kind -> (primary method, cross-checks)
ROUTES: dict[str, tuple[str, tuple[str, ...]]] = {
    "iid": ("welch_t", ()),
    "paired": ("paired_t", ("paired_bootstrap", "sign_flip_permutation")),
    "clustered": ("cluster_robust_diff", ("cluster_bootstrap",)),
    "cv_overlap": ("nadeau_bengio", ()),
}

_PHASE = {"timeseries": 3}


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

    def pretty(self) -> str:
        dep = dependence_to_dict(self.dependence)
        lines = [
            f"Analysis plan: {self.experiment}",
            f"  dependence : {dep}",
            f"  compare    : {', '.join(self.candidates)} vs baseline '{self.baseline}'",
            f"  metric     : {self.metric} ({'higher' if self.higher_is_better else 'lower'}"
            " is better)",
            f"  primary    : {self.primary}",
            f"  cross-check: {', '.join(self.cross_checks) or 'none'}",
            f"  alpha={self.alpha}  min_effect={self.min_effect}  "
            f"correction={self.correction}  seed={self.seed}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["dependence"] = dependence_to_dict(self.dependence)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnalysisPlan:
        d = dict(d)
        d["dependence"] = dependence_from_dict(d["dependence"])
        return cls(**d)


def route(dep: Dependence) -> tuple[str, tuple[str, ...]]:
    if dep.kind in ROUTES:
        return ROUTES[dep.kind]
    phase = _PHASE.get(dep.kind, "later")
    raise UnsupportedPlan(
        f"dependence '{dep.kind}' is not implemented yet (planned for phase {phase}). "
        "Refusing to fall back to a naive test that would give invalid inference."
    )


def build_plan(cfg: Config) -> AnalysisPlan:
    dep = cfg.to_dependence()
    primary, checks = route(dep)
    return AnalysisPlan(
        experiment=cfg.experiment,
        dependence=dep,
        baseline=cfg.variants.baseline,
        candidates=list(cfg.variants.candidates),
        metric=cfg.metric.name,
        higher_is_better=cfg.metric.higher_is_better,
        alpha=cfg.decision.alpha,
        min_effect=cfg.decision.min_effect,
        correction=cfg.decision.correction,
        seed=cfg.seed,
        primary=primary,
        cross_checks=list(checks),
    )
