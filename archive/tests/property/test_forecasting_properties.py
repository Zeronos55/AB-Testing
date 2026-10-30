import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from isbetter.methods.forecasting import diebold_mariano

finite = st.floats(0, 100, allow_nan=False, allow_infinity=False)
losses = st.lists(finite, min_size=12, max_size=60).map(np.array)
horizons = st.integers(1, 3)


def _ok(la, lb, h):
    d = la - lb
    return d.std() > 1e-3 and la.shape == lb.shape


@settings(deadline=None)
@given(losses, horizons, st.randoms(use_true_random=False))
def test_swap_flips_sign_and_keeps_p(la, h, rnd):
    lb = np.array([x * (0.5 + rnd.random()) for x in la])
    if not _ok(la, lb, h):
        return
    try:
        r1, r2 = (
            diebold_mariano(la, lb, h, hac_lags=h - 1),
            diebold_mariano(lb, la, h, hac_lags=h - 1),
        )
    except ValueError:
        return
    assert np.isclose(r1.estimate, -r2.estimate)
    assert np.isclose(r1.statistic, -r2.statistic)
    assert np.isclose(r1.p_value, r2.p_value)
    assert np.isclose(r1.ci_low, -r2.ci_high)


@settings(deadline=None)
@given(losses, horizons, st.floats(-50, 50), st.floats(0.5, 20))
def test_common_shift_and_scale(la, h, k, c):
    lb = la[::-1].copy() + 1.0
    if not _ok(la, lb, h):
        return
    r1 = diebold_mariano(la, lb, h, hac_lags=h - 1)
    shifted = diebold_mariano(la + k, lb + k, h, hac_lags=h - 1)
    scaled = diebold_mariano(c * la, c * lb, h, hac_lags=h - 1)
    assert np.isclose(r1.p_value, shifted.p_value, atol=1e-6)
    assert np.isclose(r1.statistic, scaled.statistic, rtol=1e-6, atol=1e-6)  # scale-free
    assert np.isclose(scaled.estimate, c * r1.estimate, rtol=1e-6, atol=1e-6)
