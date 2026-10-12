import numpy as np
import pytest

from isbetter.methods.forecasting import diebold_mariano, naive_dm_t
from isbetter.simulate.calibration import dm_pvalues, dm_type1, naive_dm_pvalues
from isbetter.simulate.dgp import dm_null_losses


@pytest.mark.parametrize("h", [1, 3, 7])
def test_null_losses_have_equal_means_and_ma_dependence(h):
    la, lb = dm_null_losses(np.random.default_rng(0), 4000, 60, h)
    assert la.shape == (4000, 60)
    assert la.mean() == pytest.approx(h, rel=0.03) and lb.mean() == pytest.approx(h, rel=0.03)
    d = la - lb
    c = d - d.mean(1, keepdims=True)
    acf = [(c[:, k:] * c[:, : 60 - k]).mean() for k in range(h + 2)]
    assert abs(acf[h]) < 0.1 * acf[0]  # independent beyond lag h - 1
    if h > 1:
        assert acf[1] > 0.1 * acf[0]  # dependent inside the window


@pytest.mark.parametrize("h", [1, 3, 5])
def test_rowwise_pvalues_match_the_method(h):
    la, lb = dm_null_losses(np.random.default_rng(1), 20, 80, h)
    p = dm_pvalues(la, lb, h)
    q = naive_dm_pvalues(la, lb)
    for i in range(len(p)):
        assert p[i] == pytest.approx(diebold_mariano(la[i], lb[i], h).p_value, rel=1e-10)
        assert q[i] == pytest.approx(naive_dm_t(la[i], lb[i]).p_value, rel=1e-10)
    plain = dm_pvalues(la, lb, h, hln=False)
    assert plain[0] == pytest.approx(diebold_mariano(la[0], lb[0], h, hln=False).p_value, rel=1e-10)


def test_small_dm_type1_run_shows_the_inflation():
    naive, dm = dm_type1(np.random.default_rng(2), reps=1500, T=150, horizon=5, chunk=500)
    assert naive.rate > 0.2 > dm.rate
