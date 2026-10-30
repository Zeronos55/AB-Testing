"""Three-way verdict with a practical-equivalence margin, plus ``n_needed``.

Convention: the CI is for ``delta = A - B`` oriented so that positive means A is better.
``min_effect`` (the margin) is the smallest difference that matters in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

from isbetter.core.result import Verdict
from isbetter.methods.power import cv_n_needed, n_needed

MIRROR = {
    "A_better_practical": "B_better_practical",
    "A_better_maybe_trivial": "B_better_maybe_trivial",
    "B_better_practical": "A_better_practical",
    "B_better_maybe_trivial": "A_better_maybe_trivial",
    "equivalent": "equivalent",
    "inconclusive": "inconclusive",
}


def mirror_label(label: str) -> str:
    """Label obtained when A and B are swapped."""
    return MIRROR[label]


@dataclass(frozen=True)
class CVSpec:
    """CV design facts needed for the repeats-vs-data advice."""

    n_scores: int
    n_train: int
    n_test: int


def decide(
    ci_low: float,
    ci_high: float,
    margin: float,
    n: float | None = None,
    cv: CVSpec | None = None,
) -> Verdict:
    """Classify the CI [ci_low, ci_high] for A - B against margin ``margin`` >= 0."""
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if ci_low > ci_high:
        raise ValueError("ci_low must be <= ci_high")
    L, U = ci_low, ci_high
    needed: float | None = None
    note: str | None = None
    if L > margin:
        label, why = "A_better_practical", f"CI lies entirely above +{margin:g}."
    elif U < -margin:
        label, why = "B_better_practical", f"CI lies entirely below -{margin:g}."
    elif L > 0:
        label, why = (
            "A_better_maybe_trivial",
            "CI is above 0 but reaches into the margin: A is better, but the gain may be trivial.",
        )
    elif U < 0:
        label, why = (
            "B_better_maybe_trivial",
            "CI is below 0 but reaches into the margin: B is better, but the gain may be trivial.",
        )
    elif -margin < L and U < margin:
        label, why = (
            "equivalent",
            f"CI lies inside +/-{margin:g}: practically equivalent "
            "(uses the (1-alpha) CI, which is conservative relative to TOST).",
        )
    else:
        label, why = "inconclusive", "CI straddles 0 and extends beyond the margin."
        if cv is not None:
            needed, note = cv_n_needed(L, U, margin, cv.n_scores, cv.n_train, cv.n_test)
        elif n is not None:
            needed, note = n_needed(L, U, margin, n)
    return Verdict(label=label, explanation=why, n_needed=needed, n_needed_note=note)
