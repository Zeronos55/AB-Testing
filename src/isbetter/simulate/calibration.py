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


def naive_cv_pvalues(diffs: np.ndarray) -> np.ndarray:
    """Row-wise p-values of the NAIVE paired t-test on (reps, J) fold differences."""
    J = diffs.shape[1]
    t = diffs.mean(1) / (diffs.std(1, ddof=1) / np.sqrt(J))
    return np.asarray(2 * stats.t.sf(np.abs(t), J - 1))


def nadeau_bengio_pvalues(diffs: np.ndarray, n_train: int, n_test: int) -> np.ndarray:
    """Row-wise p-values of the Nadeau-Bengio corrected t-test on (reps, J) differences."""
    J = diffs.shape[1]
    se = np.sqrt((1.0 / J + n_test / n_train) * diffs.var(1, ddof=1))
    return np.asarray(2 * stats.t.sf(np.abs(diffs.mean(1) / se), J - 1))


def cv_type1(
    rng: np.random.Generator,
    reps: int,
    n: int = 100,
    k: int = 10,
    r: int = 1,
    alpha: float = 0.05,
    chunk: int = 500,
) -> tuple[CalibrationResult, CalibrationResult]:
    """Type I error of (naive, Nadeau-Bengio) on the symmetric-learners CV null."""
    from isbetter.simulate.dgp import cv_null_diffs

    n_test = n // k
    n_train = n - n_test
    rej = [0, 0]
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        d = cv_null_diffs(rng, m, n, k, r)
        rej[0] += int((naive_cv_pvalues(d) < alpha).sum())
        rej[1] += int((nadeau_bengio_pvalues(d, n_train, n_test) < alpha).sum())
        done += m
    band = mc_band(alpha, reps)
    return (
        CalibrationResult(rej[0] / reps, reps, alpha, band),
        CalibrationResult(rej[1] / reps, reps, alpha, band),
    )
