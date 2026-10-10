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
