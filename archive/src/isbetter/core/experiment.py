"""A/B experiment layer: several metrics, a sample-ratio check and a ship decision.

An experiment is a list of per-metric :class:`~isbetter.core.plan.AnalysisPlan` objects (each
routed on its own metric kind) plus experiment-level rules:

1. the **SRM check** always runs first (for dependence iid and clustered);
2. **success metrics** are corrected together: one multiplicity family over every
   (candidate x success metric) hypothesis, guardrails excluded;
3. **guardrails** use a one-sided non-inferiority test at the unadjusted alpha, implemented as
   the two-sided (1 - 2 alpha) interval;
4. :func:`isbetter.decision.guardrails.decide_ship` produces one decision per candidate.

Pure: no file I/O.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from isbetter import __version__
from isbetter.core.config import Config
from isbetter.core.dependence import Clustered, Iid, dependence_to_dict
from isbetter.core.plan import AnalysisPlan, build_plans
from isbetter.core.result import Result
from isbetter.core.runner import Data, FamilyAdjustment, family_ci_alpha, raw_pvalues, run
from isbetter.core.schema import SchemaError
from isbetter.core.sufficient import AggregatedTable
from isbetter.decision.guardrails import (
    Decision,
    MetricCheck,
    decide_ship,
    guardrail_check,
    success_check,
)
from isbetter.methods.multiplicity import adjust
from isbetter.methods.srm import SRMResult, srm_check


@dataclass
class ExperimentPlan:
    experiment: str
    plans: list[AnalysisPlan]
    alpha: float
    correction: str
    ship_requires_practical: bool = False
    srm_alpha: float = 0.001
    expected_split: dict[str, float] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.plans:
            raise ValueError("an experiment needs at least one metric")
        if not any(p.role == "success" for p in self.plans):
            raise ValueError("an experiment needs at least one success metric")
        first = self.plans[0]
        for p in self.plans[1:]:
            if (p.baseline, p.candidates, p.dependence) != (
                first.baseline,
                first.candidates,
                first.dependence,
            ):
                raise ValueError("all metrics of an experiment must share arms and dependence")

    @property
    def baseline(self) -> str:
        return self.plans[0].baseline

    @property
    def candidates(self) -> list[str]:
        return self.plans[0].candidates

    def pretty(self) -> str:
        p0 = self.plans[0]
        split = (
            ", ".join(f"{a} {s:.1%}" for a, s in self.expected_split.items())
            if self.expected_split
            else "equal split"
        )
        lines = [
            f"Experiment plan: {self.experiment}",
            f"  arms       : {', '.join(self.candidates)} vs baseline '{self.baseline}'",
            f"  dependence : {dependence_to_dict(p0.dependence)}",
            f"  SRM check  : always on; expected {split}; flag at p < {self.srm_alpha:g}",
            f"  decision   : alpha={self.alpha}  correction={self.correction} "
            "(across success hypotheses only)  guardrails: one-sided, unadjusted"
            + ("  ship_requires_practical" if self.ship_requires_practical else ""),
            f"  seed       : {self.seed}",
            "  metrics:",
        ]
        for p in self.plans:
            adj = f", {p.variance_reduction} on '{p.covariate}'" if p.covariate else ""
            den = f" over '{p.denominator}'" if p.denominator else ""
            checks = f" (cross-check: {', '.join(p.cross_checks)})" if p.cross_checks else ""
            margin = "non-inferiority margin" if p.role == "guardrail" else "min effect"
            lines.append(
                f"    [{p.role}] {p.metric} ({p.metric_kind}{den}{adj}): {p.primary}{checks}; "
                f"{margin} {p.min_effect:g}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "plans"}
        d["plans"] = [p.to_dict() for p in self.plans]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentPlan:
        d = dict(d)
        d["plans"] = [AnalysisPlan.from_dict(p) for p in d["plans"]]
        return cls(**d)


def build_experiment_plan(cfg: Config) -> ExperimentPlan:
    return ExperimentPlan(
        experiment=cfg.experiment,
        plans=build_plans(cfg),
        alpha=cfg.decision.alpha,
        correction=cfg.decision.correction,
        ship_requires_practical=cfg.decision.ship_requires_practical,
        srm_alpha=cfg.design.srm_alpha,
        expected_split=cfg.design.expected_split,
        seed=cfg.seed,
    )


@dataclass
class ExperimentResult:
    experiment: str
    plan: ExperimentPlan
    results: list[Result]
    srm: SRMResult | None
    decisions: list[Decision]
    seed: int
    package_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "plan": self.plan.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "srm": asdict(self.srm) if self.srm else None,
            "decisions": [asdict(d) for d in self.decisions],
            "seed": self.seed,
            "package_version": self.package_version,
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=True)

    @classmethod
    def from_json(cls, text: str) -> ExperimentResult:
        d = json.loads(text)
        decisions = [
            Decision(
                candidate=x["candidate"],
                ship=x["ship"],
                reason=x["reason"],
                srm_failed=x["srm_failed"],
                checks=[MetricCheck(**c) for c in x["checks"]],
            )
            for x in d["decisions"]
        ]
        return cls(
            experiment=d["experiment"],
            plan=ExperimentPlan.from_dict(d["plan"]),
            results=[Result.from_json(json.dumps(r)) for r in d["results"]],
            srm=SRMResult(**d["srm"]) if d["srm"] else None,
            decisions=decisions,
            seed=d["seed"],
            package_version=d["package_version"],
            created_at=d["created_at"],
        )


def _arm_counts(plan: AnalysisPlan, data: Data) -> dict[str, int] | None:
    """Randomisation units per arm, or None where an SRM check does not apply."""
    arms = [plan.baseline, *plan.candidates]
    dep = plan.dependence
    if isinstance(data, AggregatedTable):
        missing = [a for a in arms if a not in data.units]
        if missing:
            raise SchemaError(f"aggregated data has no unit counts for arm(s) {missing}")
        return {a: data.units[a] for a in arms}
    if isinstance(dep, Iid):
        return {a: len(data.select(a, plan.metric)["value"]) for a in arms}
    if isinstance(dep, Clustered):
        out = {}
        for a in arms:
            rows = data.select(a, plan.metric)
            out[a] = len(set(rows[dep.cluster_col].astype(str)))
        return out
    return None  # paired / CV designs have equal arm sizes by construction


def run_experiment(eplan: ExperimentPlan, data: Data) -> ExperimentResult:
    """Run SRM, every metric, the multiplicity rules and the per-candidate ship decisions."""
    counts = _arm_counts(eplan.plans[0], data)
    srm = srm_check(counts, eplan.expected_split, eplan.srm_alpha) if counts is not None else None

    results: list[Result | None] = [None] * len(eplan.plans)
    success = [i for i, p in enumerate(eplan.plans) if p.role == "success"]

    # One multiplicity family across every (success metric, candidate) hypothesis.
    raw = [raw_pvalues(eplan.plans[i], data) for i in success]
    allp = np.concatenate(raw)
    adj = adjust(allp, eplan.correction)
    alpha_ci = family_ci_alpha(eplan.correction, eplan.alpha, len(allp), adj)
    start = 0
    for i, r in zip(success, raw, strict=True):
        fam = FamilyAdjustment(
            alpha_ci=alpha_ci,
            family_size=len(allp),
            adjusted_p=adj[start : start + len(r)] if eplan.correction != "none" else None,
        )
        results[i] = run(eplan.plans[i], data, fam)
        start += len(r)

    # Guardrails: one-sided non-inferiority at the unadjusted alpha = two-sided (1 - 2 alpha) CI.
    for i, p in enumerate(eplan.plans):
        if p.role == "guardrail":
            results[i] = run(p, data, FamilyAdjustment(alpha_ci=2 * eplan.alpha, family_size=1))

    done = [r for r in results if r is not None]
    assert len(done) == len(eplan.plans)

    decisions = []
    for j, cand in enumerate(eplan.candidates):
        checks = []
        for p, res in zip(eplan.plans, done, strict=True):
            c = res.comparisons[j]
            if p.role == "success":
                checks.append(
                    success_check(
                        p.metric,
                        c.verdict.label,
                        c.oriented_estimate,
                        c.oriented_ci_low,
                        c.oriented_ci_high,
                        c.adjusted_p,
                    )
                )
            else:
                checks.append(
                    guardrail_check(
                        p.metric,
                        c.oriented_estimate,
                        c.oriented_ci_low,
                        c.oriented_ci_high,
                        margin=p.min_effect,
                    )
                )
        decisions.append(
            decide_ship(
                cand,
                checks,
                srm_failed=bool(srm and srm.flagged),
                require_practical=eplan.ship_requires_practical,
            )
        )
    return ExperimentResult(
        experiment=eplan.experiment,
        plan=eplan,
        results=done,
        srm=srm,
        decisions=decisions,
        seed=eplan.seed,
    )
