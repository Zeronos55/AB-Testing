"""Invariances of the Phase 2 methods (hypothesis)."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from isbetter.core.sufficient import SufficientStats
from isbetter.methods.proportions import (
    jeffreys_interval,
    newcombe_interval,
    two_proportion_z,
    wilson_interval,
)
from isbetter.methods.srm import srm_check
from isbetter.methods.variance_reduction import cuped, regression_adjustment

counts = st.integers(1, 5000)


@st.composite
def two_tables(draw):
    n1, n2 = draw(counts), draw(counts)
    return draw(st.integers(0, n1)), n1, draw(st.integers(0, n2)), n2


@given(two_tables())
def test_swap_flips_sign_and_mirrors_interval(t):
    x1, n1, x2, n2 = t
    assume((x1 + x2) not in (0, n1 + n2))
    a, b = two_proportion_z(x1, n1, x2, n2), two_proportion_z(x2, n2, x1, n1)
    assert a.estimate == pytest.approx(-b.estimate)
    assert a.p_value == pytest.approx(b.p_value)
    assert a.ci_low == pytest.approx(-b.ci_high, abs=1e-12)
    assert a.ci_high == pytest.approx(-b.ci_low, abs=1e-12)


@given(counts.flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))))
def test_interval_contains_sample_proportion_and_stays_in_unit_interval(t):
    x, n = t
    for lo, hi in (wilson_interval(x, n), jeffreys_interval(x, n)):
        assert 0.0 <= lo <= x / n <= hi <= 1.0


@given(two_tables())
def test_newcombe_contains_estimate_and_is_bounded(t):
    x1, n1, x2, n2 = t
    lo, hi = newcombe_interval(x1, n1, x2, n2)
    assert -1.0 <= lo <= x1 / n1 - x2 / n2 <= hi <= 1.0


@given(two_tables(), st.floats(0.01, 0.2))
def test_wider_confidence_never_narrower(t, alpha):
    x1, n1, x2, n2 = t
    lo1, hi1 = newcombe_interval(x1, n1, x2, n2, alpha)
    lo2, hi2 = newcombe_interval(x1, n1, x2, n2, alpha / 2)
    assert lo2 <= lo1 + 1e-12 and hi2 >= hi1 - 1e-12


@settings(deadline=None)
@given(st.integers(0, 10_000), st.floats(0.5, 20.0), st.floats(-100, 100))
def test_cuped_estimate_unchanged_by_affine_covariate_transform(seed, scale, shift):
    rng = np.random.default_rng(seed)
    xa, xb = rng.normal(0, 1, 60), rng.normal(0, 1, 70)
    ya, yb = xa + rng.normal(0, 1, 60) + 0.3, xb + rng.normal(0, 1, 70)
    base = cuped(SufficientStats.from_arrays(ya, xa), SufficientStats.from_arrays(yb, xb))
    moved = cuped(
        SufficientStats.from_arrays(ya, scale * xa + shift),
        SufficientStats.from_arrays(yb, scale * xb + shift),
    )
    assert moved.estimate == pytest.approx(base.estimate, abs=1e-6)
    assert moved.p_value == pytest.approx(base.p_value, abs=1e-6)


@settings(deadline=None)
@given(st.integers(0, 10_000), st.floats(0.1, 10.0))
def test_adjusted_estimates_scale_with_the_outcome(seed, c):
    rng = np.random.default_rng(seed)
    xa, xb = rng.normal(0, 1, 50), rng.normal(0, 1, 55)
    ya, yb = xa + rng.normal(0, 1, 50), xb + rng.normal(0, 1, 55)
    for fn in (cuped, regression_adjustment):
        r1 = fn(SufficientStats.from_arrays(ya, xa), SufficientStats.from_arrays(yb, xb))
        r2 = fn(SufficientStats.from_arrays(c * ya, xa), SufficientStats.from_arrays(c * yb, xb))
        assert r2.estimate == pytest.approx(c * r1.estimate, rel=1e-6, abs=1e-9)
        assert r2.p_value == pytest.approx(r1.p_value, rel=1e-6, abs=1e-12)


@given(st.lists(st.integers(1, 10_000), min_size=2, max_size=6), st.integers(1, 50))
def test_srm_is_invariant_to_arm_order_and_scales_with_evidence(raw, k):
    arms = {f"a{i}": c for i, c in enumerate(raw)}
    r = srm_check(arms)
    rev = srm_check(dict(reversed(list(arms.items()))))
    assert r.p_value == pytest.approx(rev.p_value)
    # multiplying every count by k keeps the shares but strengthens the evidence
    assert srm_check({a: c * k for a, c in arms.items()}).statistic == pytest.approx(
        k * r.statistic, rel=1e-9
    )
