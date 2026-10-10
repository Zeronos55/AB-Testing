import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from isbetter.decision.verdict import decide, mirror_label
from isbetter.methods.parametric import paired_t, welch_t

finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
arrays = st.lists(finite, min_size=4, max_size=40).map(np.array)


@given(st.floats(-5, 5), st.floats(0, 5), st.floats(0, 2))
def test_verdict_mirror_property(c, h, m):
    lo, hi = c - h, c + h
    assert decide(-hi, -lo, m).label == mirror_label(decide(lo, hi, m).label)


@settings(deadline=None)
@given(arrays, arrays)
def test_swap_flips_sign_welch(a, b):
    if a.var() + b.var() < 1e-9:  # skip numerically degenerate (underflowing) variances
        return
    r1, r2 = welch_t(a, b), welch_t(b, a)
    assert np.isclose(r1.estimate, -r2.estimate)
    assert np.isclose(r1.p_value, r2.p_value)
    assert np.isclose(r1.ci_low, -r2.ci_high)


@settings(deadline=None)
@given(arrays, st.floats(-100, 100))
def test_shift_invariance_paired(a, k):
    b = a[::-1].copy()
    if np.std(a - b) < 1e-9:
        return
    r1, r2 = paired_t(a, b), paired_t(a + k, b + k)
    assert np.isclose(r1.p_value, r2.p_value, atol=1e-6)


@settings(deadline=None)
@given(arrays, st.floats(0.5, 10))
def test_scale_equivariance_welch(a, c):
    b = a[::-1] + 1.0
    if a.std() < 1e-6:
        return
    r1, r2 = welch_t(a, b), welch_t(c * a, c * b)
    assert np.isclose(r2.estimate, c * r1.estimate, rtol=1e-6, atol=1e-6)
