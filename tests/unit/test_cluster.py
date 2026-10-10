import numpy as np
import pytest

from isbetter.methods.cluster import cluster_bootstrap, cluster_robust_diff


def test_needs_two_clusters():
    with pytest.raises(ValueError, match="2 clusters"):
        cluster_robust_diff(np.arange(4.0), np.zeros(4), np.arange(4.0), np.zeros(4))


def test_bad_lengths_and_nan():
    with pytest.raises(ValueError, match="equal length"):
        cluster_robust_diff(np.arange(3.0), np.zeros(2), np.arange(3.0), np.zeros(3))
    with pytest.raises(ValueError, match="NaN"):
        cluster_robust_diff(
            np.array([1.0, np.nan]), np.array([0, 1]), np.array([1.0, 2.0]), np.array([0, 1])
        )


def test_bootstrap_reproducible():
    va = np.arange(20.0)
    ca = np.repeat(np.arange(10), 2)
    vb = va[::-1] + 1
    a = cluster_bootstrap(va, ca, vb, ca, np.random.default_rng(1), n_resamples=200)
    b = cluster_bootstrap(va, ca, vb, ca, np.random.default_rng(1), n_resamples=200)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
