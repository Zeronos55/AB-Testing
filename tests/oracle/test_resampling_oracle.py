import numpy as np
import pytest
from scipy import stats

from isbetter.methods.resampling import paired_bootstrap, sign_flip_permutation


def _scipy_perm_p(a, b, n_resamples):
    res = stats.permutation_test(
        (a, b),
        lambda x, y, axis: np.mean(x - y, axis=axis),
        permutation_type="samples",
        n_resamples=n_resamples,
        alternative="two-sided",
        random_state=1,
    )
    return res.pvalue


def test_exact_permutation_equals_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 1, 12)
    b = rng.normal(0, 1, 12)
    ours = sign_flip_permutation(a, b, np.random.default_rng(1))
    assert ours.diagnostics["exact"]
    assert ours.p_value == pytest.approx(_scipy_perm_p(a, b, np.inf), abs=1e-12)


def test_mc_permutation_close_to_scipy():
    rng = np.random.default_rng(2)
    a = rng.normal(0.3, 1, 60)
    b = rng.normal(0, 1, 60)
    ours = sign_flip_permutation(a, b, np.random.default_rng(3), n_resamples=19_999)
    ref = _scipy_perm_p(a, b, 19_999)
    se = np.sqrt(ref * (1 - ref) / 19_999)
    assert abs(ours.p_value - ref) < 5 * se + 1e-3


def test_bootstrap_percentile_close_to_t_interval():
    rng = np.random.default_rng(4)
    a = rng.normal(0.4, 1, 400)
    b = rng.normal(0, 1, 400)
    boot = paired_bootstrap(a, b, np.random.default_rng(5), n_resamples=4_000)
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    assert boot.ci_low == pytest.approx(d.mean() - 1.96 * se, abs=0.02)
    assert boot.ci_high == pytest.approx(d.mean() + 1.96 * se, abs=0.02)


def test_bca_close_to_percentile_symmetric_data():
    rng = np.random.default_rng(6)
    a = rng.normal(0.4, 1, 200)
    b = rng.normal(0, 1, 200)
    p = paired_bootstrap(a, b, np.random.default_rng(7), n_resamples=3_000)
    c = paired_bootstrap(a, b, np.random.default_rng(7), n_resamples=3_000, method="bca")
    assert c.ci_low == pytest.approx(p.ci_low, abs=0.03)
    assert c.ci_high == pytest.approx(p.ci_high, abs=0.03)
