"""Result objects. JSON-serialisable dataclasses; no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
