import numpy as np
import pytest
from scipy import stats

from isbetter.methods.parametric import cohens_d, hedges_g, paired_t, welch_t

RNG = np.random.default_rng(123)


@pytest.mark.parametrize("na,nb", [(5, 9), (30, 30), (200, 50)])
def test_welch_matches_scipy(na, nb):
    a = RNG.normal(1.0, 2.0, na)
    b = RNG.normal(0.5, 0.7, nb)
    r = welch_t(a, b, alpha=0.05)
    ref = stats.ttest_ind(a, b, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert r.statistic == pytest.approx(ref.statistic, abs=1e-8)
    assert r.p_value == pytest.approx(ref.pvalue, abs=1e-8)
    assert r.df == pytest.approx(ref.df, abs=1e-8)
    assert r.ci_low == pytest.approx(ci.low, abs=1e-8)
    assert r.ci_high == pytest.approx(ci.high, abs=1e-8)


@pytest.mark.parametrize("n", [4, 25, 500])
def test_paired_matches_scipy(n):
    a = RNG.normal(0.3, 1.0, n)
    b = a + RNG.normal(0.1, 0.5, n)
    r = paired_t(a, b)
    ref = stats.ttest_rel(a, b)
    ci = ref.confidence_interval(0.95)
    assert r.statistic == pytest.approx(ref.statistic, abs=1e-8)
    assert r.p_value == pytest.approx(ref.pvalue, abs=1e-8)
    assert r.ci_low == pytest.approx(ci.low, abs=1e-8)
    assert r.ci_high == pytest.approx(ci.high, abs=1e-8)


def test_effect_sizes_hand_computed():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    # pooled sd = sqrt(5/3); diff = -1
    d = -1 / np.sqrt(5 / 3)
    assert cohens_d(a, b) == pytest.approx(d, abs=1e-12)
    assert hedges_g(a, b) == pytest.approx(d * (1 - 3 / (4 * 6 - 1)), abs=1e-12)
