"""Oracle tests for the LLM-eval methods (Miller 2024 clustered SEs, Bowyer et al. 2025 small N).

The papers' worked numbers could not be fetched where this was built, so the references are
independent of the implementations instead: statsmodels' cluster-robust covariance, SciPy's
exact interval, a numeric optimiser for the restricted MLE, McNemar's chi-square (the score test
at delta = 0), analytic Dirichlet moments and a literal transcription of Miller's formula.
``test_errorbars_oracle.py`` adds a cross-check against the third-party ``errorbars`` package.
"""

import itertools

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

from isbetter.methods.cluster import paired_cluster_bootstrap, paired_cluster_robust
from isbetter.methods.parametric import paired_t
from isbetter.methods.proportions import jeffreys_interval
from isbetter.methods.small_n import (
    _q_tilde,
    _score_stat,
    beta_posterior_interval,
    clopper_pearson_interval,
    clt_interval,
    paired_beta_posterior,
    paired_binary_score,
    paired_score_interval,
)


def _items(seed, G=20, per=(2, 8), delta=0.4):
    rng = np.random.default_rng(seed)
    a, b, cl = [], [], []
    for g in range(G):
        passage = rng.normal(0, 1)
        for _ in range(int(rng.integers(*per))):
            difficulty = rng.normal(0, 1)
            a.append(delta + passage + difficulty + rng.normal(0, 0.5))
            b.append(passage + difficulty + rng.normal(0, 0.5))
            cl.append(f"p{g}")
    return np.array(a), np.array(b), np.array(cl)


# -- paired + clustered (Miller 2024) --------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_paired_cluster_robust_matches_statsmodels_cr1(seed):
    a, b, cl = _items(seed)
    r = paired_cluster_robust(a, b, cl)
    fit = sm.OLS(a - b, np.ones(len(a))).fit(
        cov_type="cluster", cov_kwds={"groups": cl, "use_correction": True}, use_t=True
    )
    assert r.estimate == pytest.approx(fit.params[0], abs=1e-12)
    assert r.diagnostics["se"] == pytest.approx(fit.bse[0], abs=1e-12)
    assert r.p_value == pytest.approx(fit.pvalues[0], abs=1e-12)
    assert (r.ci_low, r.ci_high) == pytest.approx(tuple(fit.conf_int()[0]), abs=1e-10)
    assert r.df == len(set(cl)) - 1


def test_singleton_clusters_reduce_to_the_paired_t_test():
    a, b, _ = _items(5)
    r = paired_cluster_robust(a, b, np.arange(len(a)))
    pt = paired_t(a, b)
    assert r.estimate == pytest.approx(pt.estimate, abs=1e-12)
    assert r.diagnostics["se"] == pytest.approx(pt.diagnostics["se"], abs=1e-12)
    assert (r.ci_low, r.ci_high) == pytest.approx((pt.ci_low, pt.ci_high), abs=1e-10)
    assert r.p_value == pytest.approx(pt.p_value, abs=1e-12)
    assert r.diagnostics["design_effect"] == pytest.approx(1.0, abs=1e-12)


def test_miller_decomposition_clt_plus_cross_terms_equals_cluster_total_form():
    """Miller's SE_clustered^2 = SE_CLT^2 + (1/n^2) sum_c sum_i sum_{j!=i} e_i e_j, written as
    triple loops, equals the cluster-total form (before the G/(G-1) factor)."""
    a, b, cl = _items(7, G=6, per=(2, 5))
    d = a - b
    n, e = len(d), d - d.mean()
    se_clt2 = (e**2).sum() / n**2  # (1/n^2) sum e^2, the ddof=0 CLT variance of the mean
    cross = 0.0
    for g in set(cl):
        idx = np.flatnonzero(cl == g)
        for i, j in itertools.product(idx, idx):
            if i != j:
                cross += e[i] * e[j]
    miller2 = se_clt2 + cross / n**2
    G = len(set(cl))
    r = paired_cluster_robust(a, b, cl)
    assert r.diagnostics["se"] ** 2 == pytest.approx(G / (G - 1) * miller2, rel=1e-12)


def test_paired_se_removes_shared_difficulty():
    """SE of the paired difference = sqrt(var(A) + var(B) - 2 cov(A, B)) / sqrt(n) (iid case)."""
    a, b, _ = _items(9)
    n = len(a)
    expected = np.sqrt(a.var(ddof=1) + b.var(ddof=1) - 2 * np.cov(a, b)[0, 1]) / np.sqrt(n)
    r = paired_cluster_robust(a, b, np.arange(n))
    assert r.diagnostics["se_iid"] == pytest.approx(expected, rel=1e-12)
    unpaired = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
    assert expected < unpaired  # items are positively correlated across the two models


def test_clustered_se_exceeds_iid_se_when_clusters_are_correlated():
    a, b, cl = _items(11, G=15, per=(6, 10))
    # add a shared passage effect to the *difference* so clusters matter
    rng = np.random.default_rng(0)
    shift = {g: rng.normal(0, 0.8) for g in set(cl)}
    a = a + np.array([shift[g] for g in cl])
    r = paired_cluster_robust(a, b, cl)
    assert r.diagnostics["design_effect"] > 1.5


def test_paired_cluster_bootstrap_agrees_with_robust():
    a, b, cl = _items(13, G=60)
    rob = paired_cluster_robust(a, b, cl)
    bs = paired_cluster_bootstrap(a, b, cl, np.random.default_rng(0), n_resamples=3000)
    assert bs.estimate == pytest.approx(rob.estimate, abs=1e-12)
    assert bs.diagnostics["se"] == pytest.approx(rob.diagnostics["se"], rel=0.12)


# -- exact / Bayesian per-arm intervals ------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_clopper_pearson_matches_scipy_exact(alpha):
    for n in (1, 5, 30, 200):
        for x in sorted({0, 1, n // 2, max(n - 1, 0), n}):
            ref = stats.binomtest(x, n).proportion_ci(confidence_level=1 - alpha, method="exact")
            assert clopper_pearson_interval(x, n, alpha) == pytest.approx(
                (ref.low, ref.high), abs=1e-10
            )


def test_beta_posterior_interval_jeffreys_prior_equals_jeffreys_and_other_priors():
    for n in (10, 50):
        for x in range(1, n):
            assert beta_posterior_interval(x, n) == pytest.approx(
                jeffreys_interval(x, n), abs=1e-12
            )
    lo, hi = beta_posterior_interval(7, 20, prior=(1.0, 1.0))
    assert (lo, hi) == pytest.approx(tuple(stats.beta(8, 14).interval(0.95)), abs=1e-12)
    # at the boundary the posterior interval is not forced to 0 / 1
    assert beta_posterior_interval(0, 20)[0] > 0
    with pytest.raises(ValueError):
        beta_posterior_interval(3, 10, prior=(0, 1))


def test_clt_interval_matches_scipy_sem():
    x = np.random.default_rng(0).normal(1, 2, 40)
    z = stats.norm.ppf(0.975)
    assert clt_interval(x) == pytest.approx(
        (x.mean() - z * stats.sem(x), x.mean() + z * stats.sem(x)), abs=1e-12
    )


# -- paired binary score interval ------------------------------------------------------------
TABLES = [(0, 0, 20), (3, 0, 30), (0, 4, 30), (5, 5, 40), (2, 9, 60), (15, 4, 100), (1, 1, 10)]


@pytest.mark.parametrize(("b", "c", "n"), TABLES)
def test_score_statistic_at_zero_is_mcnemar_uncorrected(b, c, n):
    if b + c == 0:
        assert _score_stat(b, c, n, 0.0) == 0.0
        return
    assert _score_stat(b, c, n, 0.0) ** 2 == pytest.approx((b - c) ** 2 / (b + c), abs=1e-10)
    table = np.array([[n - b - c, b], [c, 0]])
    ref = sm_mcnemar(table, exact=False, correction=False)
    assert _score_stat(b, c, n, 0.0) ** 2 == pytest.approx(ref.statistic, abs=1e-10)


@pytest.mark.parametrize(("b", "c", "n"), TABLES)
def test_interval_contains_zero_iff_uncorrected_mcnemar_does_not_reject(b, c, n):
    lo, hi = paired_score_interval(b, c, n)
    if b + c == 0:
        assert lo < 0 < hi
        return
    p = stats.chi2.sf((b - c) ** 2 / (b + c), 1)
    assert (lo < 0 < hi) == (p > 0.05)


@pytest.mark.parametrize("b", [0, 1, 4, 11])
@pytest.mark.parametrize("c", [0, 2, 9])
@pytest.mark.parametrize("delta", [-0.6, -0.2, 0.0, 0.15, 0.5, 0.9])
def test_restricted_mle_matches_numeric_optimiser(b, c, delta):
    n = 40
    s = n - b - c

    def nll(q):
        pb, pc, ps = q + delta, q, 1 - 2 * q - delta
        if min(pb, pc, ps) < 0:
            return np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = [(b, pb), (c, pc), (s, ps)]
            return -sum(k * np.log(p) if k else 0.0 for k, p in terms)

    lo, hi = max(0.0, -delta), (1 - delta) / 2
    if lo >= hi:
        return
    ref = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    assert _q_tilde(b, c, n, delta) == pytest.approx(ref.x, abs=1e-6)


@pytest.mark.parametrize(("b", "c", "n"), [t for t in TABLES if 0 < t[0] + t[1]])
def test_interval_endpoints_solve_the_score_equation(b, c, n):
    lo, hi = paired_score_interval(b, c, n)
    z = stats.norm.ppf(0.975)
    est = (b - c) / n
    if lo > -1:
        assert _score_stat(b, c, n, lo) == pytest.approx(z, abs=1e-8)
    if hi < 1:
        assert _score_stat(b, c, n, hi) == pytest.approx(-z, abs=1e-8)
    assert -1 <= lo <= est <= hi <= 1


def test_interval_is_antisymmetric_under_swapping_models_and_nests_by_level():
    for b, c, n in TABLES:
        lo, hi = paired_score_interval(b, c, n)
        slo, shi = paired_score_interval(c, b, n)
        assert (slo, shi) == pytest.approx((-hi, -lo), abs=1e-9)
        lo90, hi90 = paired_score_interval(b, c, n, alpha=0.10)
        assert lo <= lo90 and hi90 <= hi


def test_interval_has_positive_width_when_there_are_no_discordant_items():
    lo, hi = paired_score_interval(0, 0, 30)  # the Wald interval has zero width here
    assert lo < 0 < hi and hi - lo > 0.1
    assert (lo, hi) == pytest.approx((-hi, -lo), abs=1e-9)  # symmetric about 0


def test_paired_binary_score_reports_mcnemar_p_and_per_arm_intervals():
    rng = np.random.default_rng(3)
    a = rng.random(80) < 0.7
    b = np.where(rng.random(80) < 0.6, a, rng.random(80) < 0.6)
    r = paired_binary_score(a, b)
    bb, cc = int((a & ~b).sum()), int((~a & b).sum())
    ref = sm_mcnemar(np.array([[(a & b).sum(), bb], [cc, (~a & ~b).sum()]]), exact=bb + cc < 25)
    assert r.p_value == pytest.approx(ref.pvalue, abs=1e-10)
    assert r.estimate == pytest.approx(a.mean() - b.mean(), abs=1e-12)
    assert r.n_effective == bb + cc
    for key in ("wilson_a", "wilson_b", "jeffreys_a", "jeffreys_b"):
        assert len(r.diagnostics[key]) == 2
    with pytest.raises(ValueError, match="not binary"):
        paired_binary_score(np.array([0.5, 1.0]), np.array([1, 0]))
    with pytest.raises(ValueError, match="same number"):
        paired_binary_score(np.array([1, 0]), np.array([1, 0, 1]))


# -- Dirichlet posterior ---------------------------------------------------------------------
def test_paired_beta_posterior_matches_dirichlet_moments():
    a = np.array([1] * 30 + [1] * 9 + [0] * 3 + [0] * 8, dtype=bool)
    b = np.array([1] * 30 + [0] * 9 + [1] * 3 + [0] * 8, dtype=bool)  # cells 30, 9, 3, 8
    r = paired_beta_posterior(a, b, np.random.default_rng(0), draws=1_000_000)
    alpha = np.array([30, 9, 3, 8]) + 0.5
    a0 = alpha.sum()
    mean = (alpha[1] - alpha[2]) / a0
    var = (alpha[1] * (a0 - alpha[1]) + alpha[2] * (a0 - alpha[2]) + 2 * alpha[1] * alpha[2]) / (
        a0**2 * (a0 + 1)
    )
    assert r.estimate == pytest.approx(mean, abs=4 * np.sqrt(var / 1e6) + 1e-6)
    assert r.diagnostics["se"] == pytest.approx(np.sqrt(var), rel=0.01)
    # the reported interval holds 2.5% posterior mass in each tail (independent draws)
    ref = np.random.default_rng(99).dirichlet(alpha, 1_000_000)
    delta = ref[:, 1] - ref[:, 2]
    assert (delta < r.ci_low).mean() == pytest.approx(0.025, abs=0.002)
    assert (delta > r.ci_high).mean() == pytest.approx(0.025, abs=0.002)
    assert r.diagnostics["prob_a_better"] == pytest.approx((delta > 0).mean(), abs=0.004)
