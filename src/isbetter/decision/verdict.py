"""Three-way verdict with a practical-equivalence margin, plus ``n_needed``.

Convention: the CI is for ``delta = A - B`` oriented so that positive means A is better.
``min_effect`` (the margin) is the smallest difference that matters in practice.
"""

from __future__ import annotations

from isbetter.core.result import Verdict

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


def n_needed(ci_low: float, ci_high: float, margin: float, n: float) -> tuple[float | None, str]:
    """Total n for the CI to reach a decisive outcome, assuming half-width ~ 1/sqrt(n).

    Candidate targets for the half-width h*: separate from 0 (|c|), clear the margin
    (|c| - margin), or fit inside the equivalence band (margin - |c|). The largest
    positive target is the cheapest decisive outcome.
    """
    h = (ci_high - ci_low) / 2
    c = (ci_high + ci_low) / 2
    targets = [abs(c), abs(c) - margin, margin - abs(c)]
    best = max(targets)
    if best <= 0 or h <= 0:
        return None, "point estimate sits exactly on a decision boundary; no finite n suffices"
    need = n * (h / best) ** 2
    return float(need), "assumes CI half-width shrinks like 1/sqrt(n) at the current estimate"


def decide(ci_low: float, ci_high: float, margin: float, n: float | None = None) -> Verdict:
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
        if n is not None:
            needed, note = n_needed(L, U, margin, n)
    return Verdict(label=label, explanation=why, n_needed=needed, n_needed_note=note)
