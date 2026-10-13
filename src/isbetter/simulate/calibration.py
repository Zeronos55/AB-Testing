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


def sharpe_ci_coverage(
    rng: np.random.Generator,
    reps: int,
    T: int = 500,
    sharpe: float = 0.05,
    phi: float = 0.2,
    block_len: int | None = None,
    n_resamples: int = 499,
    alpha: float = 0.05,
) -> tuple[CalibrationResult, CalibrationResult]:
    """Coverage of (iid bootstrap, block bootstrap) Sharpe intervals on AR(1)-GARCH returns.

    ``block_len=None`` lets the block bootstrap choose its own length (Politis-White); the iid
    bootstrap is the same procedure with block length 1. ``rate`` is the share of intervals that
    contain the true per-period Sharpe ratio, and the band is +/- 2 Monte Carlo standard errors
    around ``1 - alpha``.
    """
    from isbetter.methods.finance import sharpe_block_bootstrap
    from isbetter.simulate.dgp import ar1_garch_returns

    r = ar1_garch_returns(rng, reps, T, sharpe=sharpe, phi=phi)
    hits = [0, 0]
    for row in r:
        for j, length in enumerate((1, block_len)):
            res = sharpe_block_bootstrap(
                row, rng, block_len=length, n_resamples=n_resamples, alpha=alpha
            )
            assert res.ci_low is not None and res.ci_high is not None
            hits[j] += int(res.ci_low <= sharpe <= res.ci_high)
    band = mc_band(1 - alpha, reps)
    return (
        CalibrationResult(hits[0] / reps, reps, 1 - alpha, band),
        CalibrationResult(hits[1] / reps, reps, 1 - alpha, band),
    )


Interval = Callable[[int, int, float], tuple[float, float]]


def clt_proportion_interval(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """The normal-theory interval for an accuracy: ``clt_interval`` on x ones and n - x zeros."""
    from isbetter.methods.small_n import clt_interval

    return clt_interval(np.r_[np.ones(x), np.zeros(n - x)], alpha)


def exact_proportion_coverage_curve(
    interval: Interval, n: int, ps: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Coverage of ``interval(x, n, alpha)`` at every true accuracy in ``ps``, with no simulation.

    A proportion's interval depends on the data only through ``x ~ Binomial(n, p)``, so the
    coverage is ``sum_x P(x) * 1[p in interval(x)]`` over the n + 1 possible counts. The
    intervals are computed once per count and reused for every ``p``.
    """
    xs = np.arange(n + 1)
    lo, hi = np.array([interval(int(x), n, alpha) for x in xs]).T
    ps = np.asarray(ps, dtype=np.float64)
    covered = (lo[:, None] <= ps[None, :]) & (ps[None, :] <= hi[:, None])
    return np.asarray((stats.binom.pmf(xs[:, None], n, ps[None, :]) * covered).sum(0))


def exact_proportion_coverage(interval: Interval, n: int, p: float, alpha: float = 0.05) -> float:
    """Coverage of ``interval`` at one true accuracy ``p`` (see the ``_curve`` version)."""
    return float(exact_proportion_coverage_curve(interval, n, np.array([p]), alpha)[0])


def paired_binary_coverage(
    rng: np.random.Generator,
    reps: int,
    n: int = 50,
    p_a: float = 0.8,
    p_b: float = 0.75,
    rho: float = 0.4,
    alpha: float = 0.05,
) -> tuple[CalibrationResult, CalibrationResult]:
    """Coverage of (normal-theory paired interval, Tango score interval) for ``p_a - p_b``.

    Items are right/wrong for two models with correlation ``rho``. Tables repeat, so each
    distinct ``(b, c)`` interval is computed once.
    """
    from isbetter.methods.small_n import paired_score_interval
    from isbetter.simulate.dgp import paired_binary_cells

    b, c = paired_binary_cells(rng, reps, n, p_a, p_b, rho)
    truth = p_a - p_b
    z = float(stats.norm.ppf(1 - alpha / 2))
    cache: dict[tuple[int, int], tuple[float, float]] = {}
    hits = [0, 0]
    for bi, ci in zip(b.tolist(), c.tolist(), strict=True):
        if (bi, ci) not in cache:
            cache[(bi, ci)] = paired_score_interval(bi, ci, n, alpha)
        lo, hi = cache[(bi, ci)]
        hits[1] += int(lo <= truth <= hi)
        var_d = max(bi + ci - (bi - ci) ** 2 / n, 0.0) / (n - 1)  # sample variance of d_i
        half = z * np.sqrt(var_d / n)
        hits[0] += int((bi - ci) / n - half <= truth <= (bi - ci) / n + half)
    band = mc_band(1 - alpha, reps)
    return (
        CalibrationResult(hits[0] / reps, reps, 1 - alpha, band),
        CalibrationResult(hits[1] / reps, reps, 1 - alpha, band),
    )


def paired_cluster_type1(
    rng: np.random.Generator,
    reps: int,
    groups: int = 30,
    per_group: int = 5,
    icc: float = 0.5,
    alpha: float = 0.05,
    chunk: int = 2_000,
) -> tuple[CalibrationResult, CalibrationResult]:
    """Type I error of (paired t ignoring clusters, paired cluster-robust CR1) for equal models.

    Follows :func:`isbetter.methods.cluster.paired_cluster_robust`: totals of centred
    differences per cluster, ``G/(G - 1)`` factor, t(G - 1) inference (checked against it in the
    tests).
    """
    from isbetter.simulate.dgp import aa_paired_clustered

    rej = [0, 0]
    done = 0
    n = groups * per_group
    while done < reps:
        m = min(chunk, reps - done)
        d = aa_paired_clustered(rng, m, groups, per_group, icc)
        flat = d.reshape(m, -1)
        mean = flat.mean(1)
        t_naive = mean / (flat.std(1, ddof=1) / np.sqrt(n))
        totals = (d - mean[:, None, None]).sum(2)
        se = np.sqrt(groups / (groups - 1) * (totals**2).sum(1)) / n
        rej[0] += int((2 * stats.t.sf(np.abs(t_naive), n - 1) < alpha).sum())
        rej[1] += int((2 * stats.t.sf(np.abs(mean / se), groups - 1) < alpha).sum())
        done += m
    band = mc_band(alpha, reps)
    return (
        CalibrationResult(rej[0] / reps, reps, alpha, band),
        CalibrationResult(rej[1] / reps, reps, alpha, band),
    )


@dataclass
class PeekingResult:
    """False-positive rates of one A/A test monitored at a set of looks."""

    n_looks: int
    n_max: int
    alpha: float
    reps: int
    single_look: CalibrationResult  # fixed-horizon test, looked at once at n_max (the valid use)
    peeked: CalibrationResult  # same test, stopped at the first look with p < alpha
    sequence: CalibrationResult  # normal-mixture confidence sequence, plug-in variance
    sequence_known: CalibrationResult  # same with the variance known (exact guarantee)


def look_schedule(n_max: int, n_looks: int | None, min_n: int) -> np.ndarray:
    """Sample sizes per arm at which an experiment is read: all of ``min_n..n_max`` or evenly
    spaced ``n_looks`` of them (the last one is always ``n_max``)."""
    if n_looks is None:
        return np.arange(min_n, n_max + 1)
    if n_looks < 1 or n_looks > n_max - min_n + 1:
        raise ValueError("n_looks must be between 1 and n_max - min_n + 1")
    if n_looks == 1:
        return np.array([n_max])
    return np.unique(np.linspace(min_n, n_max, n_looks).round().astype(int))


def peeking_type1(
    rng: np.random.Generator,
    reps: int,
    n_max: int = 2_000,
    n_looks: int | None = None,
    alpha: float = 0.05,
    n_target: int | None = None,
    min_n: int = 20,
    chunk: int = 250,
) -> PeekingResult:
    """How often an A/A test (two N(0, 1) arms) declares a difference when it is monitored.

    The fixed-horizon Welch test is valid if looked at once at ``n_max``; stopping at the first
    look with ``p < alpha`` (``peeked``) is what happens when a dashboard is checked daily, and
    its false-positive rate grows with the number of looks. The confidence sequences, tuned for
    ``n_target`` (default ``n_max``) in advance, may be read at every look. Setting
    ``n_looks=None`` reads every sample size from ``min_n`` to ``n_max``.
    """
    from isbetter.methods.sequential import mixture_boundary, optimal_rho

    looks = look_schedule(n_max, n_looks, min_n)
    t = looks.astype(np.float64)
    rho = optimal_rho(float(n_target or n_max), alpha)
    boundary = mixture_boundary(t, rho, alpha)
    crit = stats.t.ppf(
        1 - alpha / 2, 2 * (looks - 1)
    )  # fixed-horizon Welch (equal n) critical value
    hits = np.zeros(4)
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        a, b = rng.normal(size=(m, n_max)), rng.normal(size=(m, n_max))
        ma, mb = np.cumsum(a, 1)[:, looks - 1] / t, np.cumsum(b, 1)[:, looks - 1] / t
        den = np.maximum(t - 1, 1)
        va = np.maximum(np.cumsum(a**2, 1)[:, looks - 1] - t * ma**2, 0.0) / den
        vb = np.maximum(np.cumsum(b**2, 1)[:, looks - 1] - t * mb**2, 0.0) / den
        diff = np.abs(ma - mb)
        s = np.sqrt(va + vb)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = diff / (s / np.sqrt(t))
        hits[0] += int((tstat[:, -1] > crit[-1]).sum())
        hits[1] += int((tstat > crit).any(1).sum())
        hits[2] += int(((diff > s * boundary / t) & (s > 0)).any(1).sum())
        hits[3] += int((diff > np.sqrt(2.0) * boundary / t).any(1).sum())
        done += m
    band = mc_band(alpha, reps)
    r = [CalibrationResult(float(h) / reps, reps, alpha, band) for h in hits]
    return PeekingResult(len(looks), n_max, alpha, reps, r[0], r[1], r[2], r[3])
