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
