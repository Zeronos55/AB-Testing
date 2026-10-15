"""Result objects. JSON-serialisable dataclasses; no I/O."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from isbetter import __version__
from isbetter.core.plan import AnalysisPlan


@dataclass
class MethodResult:
    """Output of one statistical method for estimate = mean(A) - mean(B) (A minus B)."""

    method: str
    estimate: float
    ci_low: float | None
    ci_high: float | None
    ci_level: float | None
    p_value: float | None
    statistic: float | None
    df: float | None
    n_effective: float
    rationale: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


VERDICT_LABELS = (
    "A_better_practical",
    "A_better_maybe_trivial",
    "equivalent",
    "inconclusive",
    "B_better_maybe_trivial",
    "B_better_practical",
)


@dataclass
class Verdict:
    label: str
    explanation: str
    n_needed: float | None = None
    n_needed_note: str | None = None


@dataclass
class ComparisonResult:
    """One (baseline, candidate, metric) comparison.

    ``primary.estimate`` is candidate - baseline in the metric's raw units; ``oriented_*``
    flip sign for lower-is-better metrics so positive always means the candidate wins.
    """

    baseline: str
    candidate: str
    metric: str
    primary: MethodResult
    cross_checks: list[MethodResult]
    verdict: Verdict
    oriented_estimate: float
    oriented_ci_low: float
    oriented_ci_high: float
    n_pairs: int
    adjusted_p: float | None = None
    alpha_used: float = 0.05
    family_size: int = 0  # hypotheses corrected together; 0 = unknown (older results)
    warnings: list[str] = field(default_factory=list)  # e.g. DSR below 0.95 (overfitting guards)


@dataclass
class Result:
    experiment: str
    plan: AnalysisPlan
    comparisons: list[ComparisonResult]
    seed: int
    package_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    # statistics of the whole set of candidates rather than of one comparison (PBO over a grid)
    family_checks: list[MethodResult] = field(default_factory=list)
    family_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["plan"] = self.plan.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=True)

    @classmethod
    def from_json(cls, text: str) -> Result:
        d = json.loads(text)
        comps = []
        for c in d["comparisons"]:
            comps.append(
                ComparisonResult(
                    **{
                        **c,
                        "primary": MethodResult(**c["primary"]),
                        "cross_checks": [MethodResult(**m) for m in c["cross_checks"]],
                        "verdict": Verdict(**c["verdict"]),
                    }
                )
            )
        return cls(
            experiment=d["experiment"],
            plan=AnalysisPlan.from_dict(d["plan"]),
            comparisons=comps,
            seed=d["seed"],
            package_version=d["package_version"],
            created_at=d["created_at"],
            family_checks=[MethodResult(**m) for m in d.get("family_checks", [])],
            family_warnings=list(d.get("family_warnings", [])),
        )
