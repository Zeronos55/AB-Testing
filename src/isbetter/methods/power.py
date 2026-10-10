"""Sample-size helpers: how much more data would make an inconclusive CI decisive?"""

from __future__ import annotations

from isbetter.methods.cv import cv_variance_factor

TEST_SHARE_LIMIT = 0.8


def target_half_width(center: float, margin: float) -> float:
    """Largest CI half-width that would still give a decisive verdict.

    Candidates: separate from 0 (|c|), clear the margin (|c| - margin) or fit inside the
    equivalence band (margin - |c|). The largest positive one is the cheapest decisive outcome.
    """
    return max(abs(center), abs(center) - margin, margin - abs(center))


def n_needed(ci_low: float, ci_high: float, margin: float, n: float) -> tuple[float | None, str]:
    """Total n for a decisive CI, assuming half-width shrinks like 1/sqrt(n)."""
    h = (ci_high - ci_low) / 2
    best = target_half_width((ci_high + ci_low) / 2, margin)
    if best <= 0 or h <= 0:
        return None, "point estimate sits exactly on a decision boundary; no finite n suffices"
    return float(n * (h / best) ** 2), (
        "assumes CI half-width shrinks like 1/sqrt(n) at the current estimate"
    )


def cv_n_needed(
    ci_low: float, ci_high: float, margin: float, n_scores: int, n_train: int, n_test: int
) -> tuple[float | None, str | None]:
    """Number of CV scores (repeats x folds) needed, or a "get more data" note.

    Under Nadeau-Bengio the half-width scales like sqrt(1/J + n_test/n_train). The second
    term does not shrink with more repeats, so even infinitely many repeats leave a floor.
    """
    factor, share = cv_variance_factor(n_scores, n_train, n_test)
    h = (ci_high - ci_low) / 2
    best = target_half_width((ci_high + ci_low) / 2, margin)
    floor_ratio = n_test / n_train
    note: str | None = None
    if share > TEST_SHARE_LIMIT:
        note = (
            f"more CV repeats won't help — you need more data "
            f"({share:.0%} of the variance factor is the n_test/n_train term)"
        )
    if best <= 0 or h <= 0:
        return None, note or "point estimate sits exactly on a decision boundary"
    target_factor = factor * (best / h) ** 2
    if target_factor <= floor_ratio:
        return None, note or (
            "even infinitely many CV repeats cannot reach a decisive CI — you need more data "
            "(a larger dataset or a smaller test fraction)"
        )
    needed = 1.0 / (target_factor - floor_ratio)
    return float(max(needed, n_scores)), note
