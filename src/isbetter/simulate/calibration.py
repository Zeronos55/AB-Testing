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


def cuped_pvalues(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise CUPED p-values for (reps, n, 2) arms whose channels are (y, x)."""
    ya, xa, yb, xb = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    na, nb = ya.shape[1], yb.shape[1]
    sxx = ((xa - xa.mean(1, keepdims=True)) ** 2).sum(1) + (
        (xb - xb.mean(1, keepdims=True)) ** 2
    ).sum(1)
    sxy = ((xa - xa.mean(1, keepdims=True)) * (ya - ya.mean(1, keepdims=True))).sum(1) + (
        (xb - xb.mean(1, keepdims=True)) * (yb - yb.mean(1, keepdims=True))
    ).sum(1)
    theta = (sxy / sxx)[:, None]
    wa, wb = ya - theta * xa, yb - theta * xb
    se2a, se2b = wa.var(1, ddof=1) / na, wb.var(1, ddof=1) / nb
    est = wa.mean(1) - wb.mean(1)
    df = np.maximum((se2a + se2b) ** 2 / (se2a**2 / (na - 1) + se2b**2 / (nb - 1)) - 1.0, 1.0)
    return np.asarray(2 * stats.t.sf(np.abs(est / np.sqrt(se2a + se2b)), df))


def cuped_baseline_pvalues(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise unadjusted Welch p-values on the outcome channel of (reps, n, 2) arms."""
    return welch_pvalues(a[..., 0], b[..., 0])


def two_proportion_pvalues(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise pooled two-proportion z-test p-values for (reps, n) 0/1 matrices."""
    na, nb = a.shape[1], b.shape[1]
    p1, p2 = a.mean(1), b.mean(1)
    pooled = (a.sum(1) + b.sum(1)) / (na + nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (p1 - p2) / np.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
    p = 2 * stats.norm.sf(np.abs(z))
    return np.asarray(np.where(np.isnan(p), 1.0, p))  # no successes anywhere: cannot reject


def srm_rejection_rate(
    rng: np.random.Generator,
    reps: int,
    total: int,
    shares: tuple[float, ...],
    alpha: float = 0.001,
) -> CalibrationResult:
    """How often the SRM chi-square rejects when assignment really follows ``shares``."""
    counts = rng.multinomial(total, shares, size=reps).astype(np.float64)
    exp = total * np.asarray(shares)
    chi2 = ((counts - exp) ** 2 / exp).sum(1)
    p = stats.chi2.sf(chi2, len(shares) - 1)
    return CalibrationResult(float((p < alpha).mean()), reps, alpha, mc_band(alpha, reps))


def clustered_pvalues_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Welch p-values that IGNORE clustering: all sessions treated as independent rows."""
    return welch_pvalues(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))


def clustered_pvalues_cr1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise CR1 cluster-robust p-values with users as clusters, t(G - 1) inference.

    With equal sessions per user the CR1 two-arm difference reduces to a Welch-type test on
    per-user means weighted equally; this follows the same algebra as
    :func:`isbetter.methods.cluster.cluster_robust_diff` (checked against it in the tests).
    """
    reps, ga, s = a.shape
    gb = b.shape[1]
    ma, mb = a.mean(2), b.mean(2)  # per-user means
    N, G = (ga + gb) * s, ga + gb
    est = ma.mean(1) - mb.mean(1)
    # OLS of rows on [1, arm]: residual of a row = row - arm mean; cluster score = sum of rows.
    ua = (a - a.mean((1, 2), keepdims=True)).sum(2)  # (reps, ga) cluster sums of residuals
    ub = (b - b.mean((1, 2), keepdims=True)).sum(2)
    n_a, n_b = ga * s, gb * s
    # Sandwich variance of the arm coefficient for X = [1, arm]: each user's influence is
    # U_g / n_arm, so Var = sum_a(U^2) / n_a^2 + sum_b(U^2) / n_b^2.
    v = (ua**2).sum(1) / n_a**2 + (ub**2).sum(1) / n_b**2
    v = v * G / (G - 1) * (N - 1) / (N - 2)
    return np.asarray(2 * stats.t.sf(np.abs(est / np.sqrt(v)), G - 1))


def cluster_type1(
    rng: np.random.Generator,
    reps: int,
    users: int = 100,
    sessions: int = 10,
    icc: float = 0.3,
    alpha: float = 0.05,
    chunk: int = 250,
) -> tuple[CalibrationResult, CalibrationResult]:
    """Type I error of (naive row-level Welch, cluster-robust CR1) when randomised by user.

    Each arm has ``users`` users with ``sessions`` correlated rows each (intra-class
    correlation ``icc``) and there is no treatment effect, so any rejection is a false positive.
    """
    from isbetter.simulate.dgp import aa_clustered

    rej = [0, 0]
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        a, b = aa_clustered(rng, m, users, sessions, icc)
        rej[0] += int((clustered_pvalues_naive(a, b) < alpha).sum())
        rej[1] += int((clustered_pvalues_cr1(a, b) < alpha).sum())
        done += m
    band = mc_band(alpha, reps)
    return (
        CalibrationResult(rej[0] / reps, reps, alpha, band),
        CalibrationResult(rej[1] / reps, reps, alpha, band),
    )


def dm_pvalues(la: np.ndarray, lb: np.ndarray, horizon: int, hln: bool = True) -> np.ndarray:
    """Row-wise Diebold-Mariano p-values for (reps, T) loss matrices.

    Mirrors :func:`isbetter.methods.forecasting.diebold_mariano` (rectangular variance with
    ``horizon - 1`` lags, divided by T; HLN factor and t(T - 1) when ``hln``). A replication
    whose long-run variance estimate is not positive cannot be tested and counts as no rejection.
    """
    d = la - lb
    T = d.shape[1]
    c = d - d.mean(1, keepdims=True)
    lrv = sum(
        (1.0 if k == 0 else 2.0) * (c[:, k:] * c[:, : T - k]).sum(1) / T for k in range(horizon)
    )
    ok = lrv > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = d.mean(1) / np.sqrt(lrv / T)
    if hln:
        factor = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
        p = 2 * stats.t.sf(np.abs(factor * dm), T - 1)
    else:
        p = 2 * stats.norm.sf(np.abs(dm))
    return np.asarray(np.where(ok, p, 1.0))


def naive_dm_pvalues(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """Row-wise p-values of the NAIVE paired t-test on the loss differential."""
    return paired_pvalues(la, lb)


def dm_type1(
    rng: np.random.Generator,
    reps: int,
    T: int = 100,
    horizon: int = 3,
    alpha: float = 0.05,
    rho: float = 0.5,
    chunk: int = 2_000,
) -> tuple[CalibrationResult, CalibrationResult]:
    """Type I error of (naive paired t, Diebold-Mariano + HLN) for equally good forecasters."""
    from isbetter.simulate.dgp import dm_null_losses

    rej = [0, 0]
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        la, lb = dm_null_losses(rng, m, T, horizon, rho)
        rej[0] += int((naive_dm_pvalues(la, lb) < alpha).sum())
        rej[1] += int((dm_pvalues(la, lb, horizon) < alpha).sum())
        done += m
    band = mc_band(alpha, reps)
    return (
        CalibrationResult(rej[0] / reps, reps, alpha, band),
        CalibrationResult(rej[1] / reps, reps, alpha, band),
    )
