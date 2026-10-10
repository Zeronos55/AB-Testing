"""Data-generating processes for calibration. All take a Generator and return arrays."""

from __future__ import annotations

import numpy as np


def aa_independent(
    rng: np.random.Generator, reps: int, n: int, skew: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Two independent samples from the SAME distribution (a null world), shape (reps, n)."""
    if skew:
        return rng.exponential(1.0, (reps, n)), rng.exponential(1.0, (reps, n))
    return rng.normal(0.0, 1.0, (reps, n)), rng.normal(0.0, 1.0, (reps, n))


def aa_paired(
    rng: np.random.Generator, reps: int, n: int, unit_sd: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Paired null world: shared unit effect plus independent noise; no true difference."""
    unit = rng.normal(0.0, unit_sd, (reps, n))
    return unit + rng.normal(0.0, 1.0, (reps, n)), unit + rng.normal(0.0, 1.0, (reps, n))


def cv_null_diffs(
    rng: np.random.Generator, reps: int, n: int = 100, k: int = 10, r: int = 1
) -> np.ndarray:
    """Per-fold MSE differences between two learners that are equally good by symmetry.

    ``y = x1 + x2 + noise``; learner A regresses y on x1 alone, learner B on x2 alone, so
    their expected test error is identical (the null). Scores come from ``r`` repeats of
    ``k``-fold CV on the same data. Returns an array of shape (reps, r * k) of
    ``mse_A - mse_B`` per fold.
    """
    x1, x2 = rng.normal(size=(reps, n)), rng.normal(size=(reps, n))
    y = x1 + x2 + rng.normal(size=(reps, n))
    out = np.empty((reps, r * k))
    col = 0
    for _ in range(r):
        perm = rng.permutation(n)
        for te in np.array_split(perm, k):
            tr = np.setdiff1d(perm, te, assume_unique=True)
            mse = []
            for x in (x1, x2):
                xt, yt = x[:, tr], y[:, tr]
                xm, ym = xt.mean(1, keepdims=True), yt.mean(1, keepdims=True)
                slope = ((xt - xm) * (yt - ym)).sum(1, keepdims=True) / ((xt - xm) ** 2).sum(
                    1, keepdims=True
                )
                pred = ym + slope * (x[:, te] - xm)
                mse.append(((y[:, te] - pred) ** 2).mean(1))
            out[:, col] = mse[0] - mse[1]
            col += 1
    return out
