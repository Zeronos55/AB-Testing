import numpy as np
import pytest

from isbetter.methods.forecasting import (
    autocovariances,
    diebold_mariano,
    hln_factor,
    long_run_variance,
    naive_dm_t,
)


def test_autocovariances_divide_by_T_not_T_minus_lag():
    d = np.array([1.0, -1.0, 1.0, -1.0])  # centred already
    g = autocovariances(d, 2)
    assert g.tolist() == [1.0, -0.75, 0.5]  # (-3)/4 and 2/4


def test_long_run_variance_kernels_and_validation():
    d = np.array([1.0, -1.0, 1.0, -1.0])
    assert long_run_variance(d, 2, "rectangular") == pytest.approx(1 + 2 * (-0.75 + 0.5))
    assert long_run_variance(d, 2, "bartlett") == pytest.approx(
        1 + 2 * (-0.75 * (2 / 3) + 0.5 * (1 / 3))
    )
    with pytest.raises(ValueError, match="kernel"):
        long_run_variance(d, 1, "parzen")


def test_hln_factor_is_one_at_h1_large_T():
    assert hln_factor(10_000, 1) == pytest.approx(1.0, abs=1e-4)
    assert hln_factor(100, 5) < 1.0
    with pytest.raises(ValueError, match="too large"):
        hln_factor(3, 3)


def test_validation():
    ok = np.arange(10.0)
    with pytest.raises(ValueError, match="equal length"):
        diebold_mariano(ok, ok[:-1])
    with pytest.raises(ValueError, match="at least 3"):
        diebold_mariano(ok[:2], ok[:2] + 1)
    with pytest.raises(ValueError, match="NaN"):
        diebold_mariano(ok, np.where(ok == 3, np.nan, ok))
    with pytest.raises(ValueError, match="horizon"):
        diebold_mariano(ok, ok[::-1], horizon=0)
    with pytest.raises(ValueError, match="hac_lags"):
        diebold_mariano(ok, ok[::-1], hac_lags=-1)
    with pytest.raises(ValueError, match="more than"):
        diebold_mariano(ok, ok[::-1], hac_lags=10)


def test_identical_forecasts_have_no_defined_statistic():
    x = np.random.default_rng(0).standard_normal(30) ** 2
    with pytest.raises(ValueError, match="not positive"):
        diebold_mariano(x, x)


def test_negative_rectangular_variance_points_to_bartlett():
    # strong negative lag-1 autocovariance with h = 2 makes gamma_0 + 2*gamma_1 < 0
    d = np.tile([1.0, -1.0], 6)
    with pytest.raises(ValueError, match="hac_lags"):
        diebold_mariano(d, np.zeros(12), horizon=2)
    ok = diebold_mariano(d, np.zeros(12), horizon=2, hac_lags=1)
    assert ok.diagnostics["kernel"] == "bartlett" and ok.p_value is not None


def test_result_fields_and_rationale():
    rng = np.random.default_rng(1)
    la, lb = rng.standard_normal(60) ** 2, rng.standard_normal(60) ** 2 * 1.3
    r = diebold_mariano(la, lb, horizon=3)
    assert r.method == "diebold_mariano_hln"
    assert r.ci_low < r.estimate < r.ci_high and r.ci_level == 0.95
    assert r.n_effective == 60 and r.df == 59
    assert "horizon 3" in r.rationale and "Harvey-Leybourne-Newbold" in r.rationale
    assert r.diagnostics["lags"] == 2 and r.diagnostics["kernel"] == "rectangular"
    plain = diebold_mariano(la, lb, horizon=3, hln=False)
    assert plain.method == "diebold_mariano" and plain.df is None
    assert plain.diagnostics["hln_factor"] == 1.0


def test_hln_is_more_conservative_than_plain_dm():
    rng = np.random.default_rng(2)
    la, lb = rng.standard_normal(40) ** 2, rng.standard_normal(40) ** 2 * 1.4
    hln, plain = diebold_mariano(la, lb, horizon=4), diebold_mariano(la, lb, horizon=4, hln=False)
    assert hln.p_value >= plain.p_value
    assert hln.ci_high - hln.ci_low > plain.ci_high - plain.ci_low


def test_naive_flagged_and_validated():
    rng = np.random.default_rng(3)
    r = naive_dm_t(rng.standard_normal(20) ** 2, rng.standard_normal(20) ** 2 + 0.5)
    assert r.diagnostics["naive"] and "NAIVE" in r.rationale
    with pytest.raises(ValueError, match="zero variance"):
        naive_dm_t(np.arange(5.0), np.arange(5.0) - 1.0)
