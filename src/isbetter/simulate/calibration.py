"""Type I error / calibration harness, vectorised across replications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

Dgp = Callable[[np.random.Generator, int, int], tuple[np.ndarray, np.ndarray]]
PValues = Callable[[np.ndarray, np.ndarray], np.ndarray]


def welch_pvalues(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Welch p-values for (reps, n) matrices."""
    na, nb = a.shape[1], b.shape[1]
    va, vb = a.var(axis=1, ddof=1) / na, b.var(axis=1, ddof=1) / nb
    t = (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    return np.asarray(2 * stats.t.sf(np.abs(t), df))


def paired_pvalues(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise paired-t p-values for (reps, n) matrices."""
    d = a - b
    n = d.shape[1]
    t = d.mean(axis=1) / (d.std(axis=1, ddof=1) / np.sqrt(n))
    return np.asarray(2 * stats.t.sf(np.abs(t), n - 1))


def mc_band(alpha: float, reps: int) -> tuple[float, float]:
    """+/- 2 Monte Carlo standard errors around the nominal rate."""
    half = 2 * np.sqrt(alpha * (1 - alpha) / reps)
    return alpha - half, alpha + half


@dataclass
class CalibrationResult:
    rate: float
    reps: int
    alpha: float
    band: tuple[float, float]

    @property
    def passed(self) -> bool:
        return self.band[0] <= self.rate <= self.band[1]


def type1_rate(
    pvalues: PValues,
    dgp: Dgp,
    rng: np.random.Generator,
    reps: int = 10_000,
    n: int = 30,
    alpha: float = 0.05,
    chunk: int = 2_000,
) -> CalibrationResult:
    """Rejection rate under a null DGP; replications simulated in chunks to bound memory."""
    rejections = 0
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        a, b = dgp(rng, m, n)
        rejections += int((pvalues(a, b) < alpha).sum())
        done += m
    return CalibrationResult(rejections / reps, reps, alpha, mc_band(alpha, reps))
