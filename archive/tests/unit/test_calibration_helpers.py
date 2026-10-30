import numpy as np
import pytest

from isbetter.methods.parametric import paired_t, welch_t
from isbetter.simulate.calibration import mc_band, paired_pvalues, welch_pvalues


def test_vectorised_pvalues_match_scalar_methods():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 25)), rng.normal(0.3, 2, size=(5, 25))
    for i in range(5):
        assert welch_pvalues(a, b)[i] == pytest.approx(welch_t(a[i], b[i]).p_value, abs=1e-10)
        assert paired_pvalues(a, b)[i] == pytest.approx(paired_t(a[i], b[i]).p_value, abs=1e-10)


def test_band_width():
    lo, hi = mc_band(0.05, 2000)
    assert hi - lo == pytest.approx(4 * np.sqrt(0.05 * 0.95 / 2000))


def test_vectorised_cuped_matches_method():
    from isbetter.core.sufficient import SufficientStats
    from isbetter.methods.variance_reduction import cuped
    from isbetter.simulate.calibration import cuped_baseline_pvalues, cuped_pvalues
    from isbetter.simulate.dgp import aa_covariate

    a, b = aa_covariate(np.random.default_rng(1), 6, 60, rho=0.6)
    for i in range(6):
        m = cuped(
            SufficientStats.from_arrays(a[i, :, 0], a[i, :, 1]),
            SufficientStats.from_arrays(b[i, :, 0], b[i, :, 1]),
        )
        assert cuped_pvalues(a, b)[i] == pytest.approx(m.p_value, rel=1e-8)
        assert cuped_baseline_pvalues(a, b)[i] == pytest.approx(
            welch_t(a[i, :, 0], b[i, :, 0]).p_value, rel=1e-8
        )


def test_vectorised_two_proportion_matches_method():
    from isbetter.methods.proportions import two_proportion_z
    from isbetter.simulate.calibration import two_proportion_pvalues
    from isbetter.simulate.dgp import aa_binary

    a, b = aa_binary(np.random.default_rng(2), 5, 300, p=0.2)
    for i in range(5):
        m = two_proportion_z(int(a[i].sum()), 300, int(b[i].sum()), 300)
        assert two_proportion_pvalues(a, b)[i] == pytest.approx(m.p_value, rel=1e-8)


def test_two_proportion_pvalues_handle_all_zero_rows():
    from isbetter.simulate.calibration import two_proportion_pvalues

    z = np.zeros((2, 50), dtype=np.int8)
    assert (two_proportion_pvalues(z, z) == 1.0).all()


def test_vectorised_cluster_pvalues_match_cluster_robust_diff():
    from isbetter.methods.cluster import cluster_robust_diff
    from isbetter.simulate.calibration import clustered_pvalues_cr1, clustered_pvalues_naive
    from isbetter.simulate.dgp import aa_clustered

    a, b = aa_clustered(np.random.default_rng(3), 4, 12, sessions=4, icc=0.6)
    ga, gb, s = a.shape[1], b.shape[1], a.shape[2]
    for i in range(4):
        users_a = np.repeat([f"a{u}" for u in range(ga)], s)
        users_b = np.repeat([f"b{u}" for u in range(gb)], s)
        m = cluster_robust_diff(a[i].ravel(), users_a, b[i].ravel(), users_b)
        assert clustered_pvalues_cr1(a, b)[i] == pytest.approx(m.p_value, rel=1e-8)
        assert clustered_pvalues_naive(a, b)[i] == pytest.approx(
            welch_t(a[i].ravel(), b[i].ravel()).p_value, rel=1e-8
        )
