"""Oracle tests: proportions methods against statsmodels.stats.proportion."""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

from isbetter.core.sufficient import SufficientStats
from isbetter.methods.proportions import (
    binary_counts,
    fisher_exact,
    fisher_exact_stats,
    jeffreys_interval,
    newcombe_interval,
    two_proportion_z,
    two_proportion_z_stats,
    wilson_interval,
)

RNG = np.random.default_rng(20261010)
# Random (x1, n1, x2, n2) with interior counts, small and large n.
CASES = [
    (int(a), int(n1), int(b), int(n2))
    for n1, n2 in zip(RNG.integers(15, 5000, 60), RNG.integers(15, 5000, 60), strict=True)
    for a, b in [(RNG.integers(1, n1), RNG.integers(1, n2))]
]


@pytest.mark.parametrize("alpha", [0.05, 0.01, 0.2])
def test_wilson_matches_statsmodels(alpha):
    for x1, n1, *_ in CASES:
        lo, hi = wilson_interval(x1, n1, alpha)
        slo, shi = proportion_confint(x1, n1, alpha=alpha, method="wilson")
        assert lo == pytest.approx(slo, abs=1e-12) and hi == pytest.approx(shi, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_jeffreys_matches_statsmodels_in_interior(alpha):
    for x1, n1, *_ in CASES:
        lo, hi = jeffreys_interval(x1, n1, alpha)
        slo, shi = proportion_confint(x1, n1, alpha=alpha, method="jeffreys")
        assert lo == pytest.approx(slo, abs=1e-10) and hi == pytest.approx(shi, abs=1e-10)


def test_jeffreys_clamps_at_boundaries():
    assert jeffreys_interval(0, 30)[0] == 0.0
    assert jeffreys_interval(30, 30)[1] == 1.0
    lo, hi = jeffreys_interval(0, 30)
    assert 0 < hi < 0.2


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_newcombe_matches_statsmodels(alpha):
    for x1, n1, x2, n2 in CASES:
        lo, hi = newcombe_interval(x1, n1, x2, n2, alpha)
        slo, shi = confint_proportions_2indep(
            x1, n1, x2, n2, method="newcomb", compare="diff", alpha=alpha
        )
        assert lo == pytest.approx(slo, abs=1e-10) and hi == pytest.approx(shi, abs=1e-10)


def test_z_test_matches_statsmodels_pooled_ztest():
    for x1, n1, x2, n2 in CASES:
        r = two_proportion_z(x1, n1, x2, n2)
        z, p = proportions_ztest([x1, x2], [n1, n2])
        assert r.statistic == pytest.approx(z, abs=1e-10)
        assert r.p_value == pytest.approx(p, abs=1e-12)
        assert r.estimate == pytest.approx(x1 / n1 - x2 / n2)


def test_z_test_wald_ci_matches_statsmodels_wald():
    for x1, n1, x2, n2 in CASES:
        r = two_proportion_z(x1, n1, x2, n2, ci="wald")
        slo, shi = confint_proportions_2indep(x1, n1, x2, n2, method="wald", compare="diff")
        assert r.ci_low == pytest.approx(slo, abs=1e-10)
        assert r.ci_high == pytest.approx(shi, abs=1e-10)


def test_auto_ci_switches_to_newcombe_for_small_cells():
    big = two_proportion_z(500, 5000, 450, 5000)
    small = two_proportion_z(4, 200, 12, 210)
    assert big.diagnostics["ci_method"] == "wald"
    assert small.diagnostics["ci_method"] == "newcombe"
    assert (small.ci_low, small.ci_high) == pytest.approx(newcombe_interval(4, 200, 12, 210))
    forced = two_proportion_z(500, 5000, 450, 5000, ci="newcombe")
    assert forced.diagnostics["ci_method"] == "newcombe"


def test_fisher_matches_scipy_and_agrees_with_z_at_large_n():
    for x1, n1, x2, n2 in CASES[:20]:
        r = fisher_exact(x1, n1, x2, n2)
        _, p = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])
        assert r.p_value == pytest.approx(p)
        assert r.ci_low is None
    z = two_proportion_z(520, 5000, 450, 5000)
    f = fisher_exact(520, 5000, 450, 5000)
    assert f.p_value == pytest.approx(z.p_value, rel=0.1)


def test_stats_wrappers_equal_count_versions():
    rng = np.random.default_rng(1)
    ya, yb = (rng.random(800) < 0.12).astype(float), (rng.random(900) < 0.10).astype(float)
    a, b = SufficientStats.from_arrays(ya), SufficientStats.from_arrays(yb)
    assert binary_counts(a) == (int(ya.sum()), 800)
    assert (
        two_proportion_z_stats(a, b).p_value
        == two_proportion_z(int(ya.sum()), 800, int(yb.sum()), 900).p_value
    )
    assert (
        fisher_exact_stats(a, b).p_value
        == fisher_exact(int(ya.sum()), 800, int(yb.sum()), 900).p_value
    )


def test_non_binary_stats_rejected():
    s = SufficientStats.from_arrays(np.array([0.0, 0.5, 1.0, 2.0]))
    with pytest.raises(ValueError, match="not binary"):
        binary_counts(s)


@pytest.mark.parametrize(
    "args",
    [(-1, 10, 1, 10), (11, 10, 1, 10), (1, 0, 1, 10), (1.5, 10, 1, 10)],
)
def test_bad_counts_rejected(args):
    with pytest.raises(ValueError):
        two_proportion_z(*args)


def test_degenerate_tables_raise():
    with pytest.raises(ValueError, match="undefined"):
        two_proportion_z(0, 50, 0, 60)
    with pytest.raises(ValueError, match="undefined"):
        two_proportion_z(50, 50, 60, 60)
