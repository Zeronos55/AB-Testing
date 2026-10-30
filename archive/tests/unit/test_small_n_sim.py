import numpy as np
import pytest

from isbetter.methods.cluster import paired_cluster_robust
from isbetter.simulate.calibration import paired_cluster_type1
from isbetter.simulate.dgp import aa_paired_clustered, paired_binary_cells


def test_paired_binary_cells_have_requested_marginals_and_correlation():
    rng = np.random.default_rng(0)
    b, c = paired_binary_cells(rng, 2000, 100, 0.8, 0.7, rho=0.4)
    assert (b + c <= 100).all()
    assert (b - c).mean() / 100 == pytest.approx(0.1, abs=0.005)
    # discordance shrinks as the models become more correlated
    b2, c2 = paired_binary_cells(rng, 2000, 100, 0.8, 0.7, rho=0.7)
    assert (b2 + c2).mean() < (b + c).mean()
    with pytest.raises(ValueError, match="infeasible"):
        paired_binary_cells(rng, 10, 50, 0.9, 0.1, rho=0.9)


def test_vectorised_cr1_equals_the_method_row_by_row():
    from scipy import stats

    rng = np.random.default_rng(1)
    d = aa_paired_clustered(rng, 5, 12, 4, 0.5)
    G, m = 12, 4
    for row in d:
        clusters = np.repeat(np.arange(G), m)
        res = paired_cluster_robust(row.ravel(), np.zeros(G * m), clusters)
        mean = row.mean()
        totals = (row - mean).sum(1)
        se = np.sqrt(G / (G - 1) * (totals**2).sum()) / (G * m)
        assert res.p_value == pytest.approx(2 * stats.t.sf(abs(mean / se), G - 1), abs=1e-12)


def test_paired_cluster_type1_runs_small():
    naive, cr1 = paired_cluster_type1(np.random.default_rng(0), 400, groups=20, per_group=5)
    assert naive.rate > cr1.rate
