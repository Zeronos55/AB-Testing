"""Cross-check against ``errorbars``, a third-party implementation of the eval statistics.

``errorbars`` (PyPI, by Anton Soloviev, MIT) is a separate package written from the same
literature as our LLM-eval methods (Miller 2024 clustered and paired standard errors; Wilson
intervals; exact McNemar). It is **not** Miller's own code and it is not the papers' worked
numbers, so this is one more independent implementation, not a paper fixture: agreement shows
two programmers reading the same formulas got the same numbers.

Compared here: the paired difference with a cluster-robust standard error (G/(G-1) times the
squared cluster totals over n^2, t with G - 1 degrees of freedom), the plain paired t interval,
the Wilson interval, the CLT interval and the exact McNemar p-value.
"""

import numpy as np
import pytest
from errorbars.compare import mcnemar_exact, paired_compare
from errorbars.stats import mean_ci_clt, wilson_ci

from isbetter.methods.classification import mcnemar
from isbetter.methods.cluster import paired_cluster_robust
from isbetter.methods.parametric import paired_t
from isbetter.methods.proportions import wilson_interval
from isbetter.methods.small_n import clt_interval

# errorbars computes the t quantile with its own incomplete-beta routine, so allow a tolerance
# a little looser than machine precision.
REL = 1e-6


def clustered_scores(seed, groups=25, binary=False):
    rng = np.random.default_rng(seed)
    a, b, cl = [], [], []
    for g in range(groups):
        shared = rng.normal(0, 0.8)
        for _ in range(int(rng.integers(1, 9))):
            x, y = shared + rng.normal(0.2, 1), shared + rng.normal(0, 1)
            a.append(float(x > 0) if binary else x)
            b.append(float(y > 0) if binary else y)
            cl.append(f"p{g}")
    return np.array(a), np.array(b), np.array(cl)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("binary", [False, True])
def test_paired_cluster_robust_matches_errorbars(seed, binary):
    a, b, cl = clustered_scores(seed, binary=binary)
    ours = paired_cluster_robust(a, b, cl)
    theirs = paired_compare(a, b, clusters=cl)
    assert ours.estimate == pytest.approx(theirs.mean_diff, rel=1e-12, abs=1e-14)
    assert ours.diagnostics["se"] == pytest.approx(theirs.se_clustered, rel=1e-10)
    assert ours.ci_low == pytest.approx(theirs.ci_low_clustered, rel=REL, abs=1e-9)
    assert ours.ci_high == pytest.approx(theirs.ci_high_clustered, rel=REL, abs=1e-9)
    assert ours.p_value == pytest.approx(theirs.p_value_clustered, rel=REL, abs=1e-12)
    assert ours.df == len(set(cl)) - 1


@pytest.mark.parametrize("seed", range(5))
def test_plain_paired_interval_matches_errorbars(seed):
    a, b, _ = clustered_scores(seed)
    ours, theirs = paired_t(a, b), paired_compare(a, b)
    assert ours.estimate == pytest.approx(theirs.mean_diff, rel=1e-12, abs=1e-14)
    assert ours.ci_low == pytest.approx(theirs.ci_low, rel=REL, abs=1e-9)
    assert ours.ci_high == pytest.approx(theirs.ci_high, rel=REL, abs=1e-9)
    assert ours.p_value == pytest.approx(theirs.p_value, rel=REL, abs=1e-12)


def test_singleton_clusters_reduce_to_the_paired_interval_in_both_packages():
    a, b, _ = clustered_scores(3)
    singles = np.arange(len(a))
    ours = paired_cluster_robust(a, b, singles)
    theirs = paired_compare(a, b, clusters=singles)
    assert ours.ci_low == pytest.approx(theirs.ci_low_clustered, rel=REL, abs=1e-9)
    assert theirs.se_clustered == pytest.approx(theirs.se_paired, rel=1e-10)


@pytest.mark.parametrize(("x", "n"), [(0, 30), (1, 30), (27, 30), (30, 30), (45, 100), (95, 100)])
def test_wilson_matches_errorbars(x, n):
    theirs = wilson_ci(x, n)
    lo, hi = wilson_interval(x, n)
    assert lo == pytest.approx(theirs.ci_low, rel=REL, abs=1e-9)
    assert hi == pytest.approx(theirs.ci_high, rel=REL, abs=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_clt_interval_matches_errorbars(seed):
    v = np.random.default_rng(seed).normal(size=40)
    theirs = mean_ci_clt(v)
    lo, hi = clt_interval(v)
    assert lo == pytest.approx(theirs.ci_low, rel=REL, abs=1e-9)
    assert hi == pytest.approx(theirs.ci_high, rel=REL, abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_exact_mcnemar_matches_errorbars(seed):
    a, b, _ = clustered_scores(seed, groups=6, binary=True)  # few discordant pairs: exact branch
    theirs = mcnemar_exact(a, b)
    ours = mcnemar(a.astype(bool), b.astype(bool), exact=True)
    assert ours.p_value == pytest.approx(theirs.p_value, rel=1e-12, abs=1e-14)
    assert (ours.diagnostics["b"], ours.diagnostics["c"]) == (theirs.n10, theirs.n01)
