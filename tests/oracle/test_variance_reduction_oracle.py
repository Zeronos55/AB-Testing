"""Oracle tests for CUPED, regression adjustment and the delta method (vs statsmodels/NumPy)."""

import numpy as np
import pytest
import statsmodels.api as sm

from isbetter.core.sufficient import SufficientStats
from isbetter.methods.parametric import welch_t, welch_t_stats
from isbetter.methods.variance_reduction import (
    cuped,
    delta_method_ratio,
    ratio_bootstrap,
    regression_adjustment,
)


def make_arms(seed, na=4000, nb=3500, rho=0.6, effect=0.3, shift_b=0.0):
    rng = np.random.default_rng(seed)
    xa, xb = rng.normal(10, 3, na), rng.normal(10 + shift_b, 3, nb)
    ya = 5 + effect + rho * (xa - 10) + rng.normal(0, 2, na)
    yb = 5 + rho * (xb - 10) + rng.normal(0, 2, nb)
    return ya, xa, yb, xb


def stats_of(ya, xa, yb, xb):
    return SufficientStats.from_arrays(ya, xa), SufficientStats.from_arrays(yb, xb)


@pytest.mark.parametrize("seed", range(5))
def test_cuped_estimate_equals_ols_treatment_coefficient(seed):
    ya, xa, yb, xb = make_arms(seed, shift_b=0.2)
    y = np.r_[ya, yb]
    T = np.r_[np.ones(len(ya)), np.zeros(len(yb))]
    xc = np.r_[xa, xb] - np.r_[xa, xb].mean()  # centred covariate
    ols = sm.OLS(y, sm.add_constant(np.column_stack([T, xc]))).fit()
    r = cuped(*stats_of(ya, xa, yb, xb))
    assert r.estimate == pytest.approx(ols.params[1], abs=1e-9)
    # theta is the OLS slope on the covariate
    assert r.diagnostics["theta"] == pytest.approx(ols.params[2], abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_cuped_se_matches_explicit_adjusted_welch_and_hc_ols(seed):
    ya, xa, yb, xb = make_arms(seed)
    r = cuped(*stats_of(ya, xa, yb, xb))
    th = r.diagnostics["theta"]
    adj = welch_t(ya - th * xa, yb - th * xb)
    assert r.diagnostics["se"] == pytest.approx(adj.diagnostics["se"], rel=1e-9)
    y = np.r_[ya, yb]
    T = np.r_[np.ones(len(ya)), np.zeros(len(yb))]
    X = sm.add_constant(np.column_stack([T, np.r_[xa, xb]]))
    hc = sm.OLS(y, X).fit(cov_type="HC1").bse[1]
    assert r.diagnostics["se"] == pytest.approx(hc, rel=0.02)


@pytest.mark.parametrize("seed", range(5))
def test_regression_adjustment_equals_lin_interaction_coefficient(seed):
    ya, xa, yb, xb = make_arms(seed, shift_b=0.3)
    y = np.r_[ya, yb]
    T = np.r_[np.ones(len(ya)), np.zeros(len(yb))]
    x = np.r_[xa, xb]
    xc = x - x.mean()
    ols = sm.OLS(y, sm.add_constant(np.column_stack([T, xc, T * xc]))).fit()
    r = regression_adjustment(*stats_of(ya, xa, yb, xb))
    assert r.estimate == pytest.approx(ols.params[1], abs=1e-9)
    hc = sm.OLS(y, sm.add_constant(np.column_stack([T, xc, T * xc]))).fit(cov_type="HC1").bse[1]
    assert r.diagnostics["se"] == pytest.approx(hc, rel=0.02)


def test_cuped_and_adjustment_reduce_variance_by_about_rho_squared():
    ya, xa, yb, xb = make_arms(7, na=20000, nb=20000, rho=1.0)  # corr = 3/sqrt(9+4)
    rho2 = 9 / 13
    for fn in (cuped, regression_adjustment):
        r = fn(*stats_of(ya, xa, yb, xb))
        assert r.diagnostics["variance_reduction"] == pytest.approx(rho2, abs=0.02)
    assert cuped(*stats_of(ya, xa, yb, xb)).diagnostics["predicted_variance_reduction"] == (
        pytest.approx(rho2, abs=0.02)
    )


def test_cuped_with_uncorrelated_covariate_changes_nothing_much():
    ya, xa, yb, xb = make_arms(8, rho=0.0)
    a, b = stats_of(ya, xa, yb, xb)
    assert cuped(a, b).diagnostics["se"] == pytest.approx(
        welch_t_stats(a, b).diagnostics["se"], rel=0.01
    )


def test_cuped_estimate_invariant_to_covariate_shift_and_scale_of_x():
    ya, xa, yb, xb = make_arms(9)
    base = cuped(*stats_of(ya, xa, yb, xb))
    moved = cuped(*stats_of(ya, 3 * xa + 100, yb, 3 * xb + 100))
    assert moved.estimate == pytest.approx(base.estimate, abs=1e-8)
    assert moved.diagnostics["se"] == pytest.approx(base.diagnostics["se"], rel=1e-8)


def test_antisymmetry_when_arms_swap():
    ya, xa, yb, xb = make_arms(10)
    a, b = stats_of(ya, xa, yb, xb)
    for fn in (cuped, regression_adjustment):
        f, r = fn(a, b), fn(b, a)
        assert f.estimate == pytest.approx(-r.estimate, abs=1e-10)
        assert f.p_value == pytest.approx(r.p_value, rel=1e-8)


def test_welch_from_stats_equals_welch_from_arrays():
    ya, _, yb, _ = make_arms(11)
    f = welch_t(ya, yb)
    s = welch_t_stats(SufficientStats.from_arrays(ya), SufficientStats.from_arrays(yb))
    for field in ("estimate", "ci_low", "ci_high", "p_value", "df"):
        assert getattr(s, field) == pytest.approx(getattr(f, field), rel=1e-10)
    assert s.diagnostics["hedges_g"] == pytest.approx(f.diagnostics["hedges_g"], rel=1e-9)


# ---------------------------------------------------------------- delta method


def ratio_arms(seed, na=3000, nb=3200, lift=0.1):
    rng = np.random.default_rng(seed)
    out = []
    for n, mult in ((na, 1 + lift), (nb, 1.0)):
        visits = rng.poisson(3, n) + 1  # positive denominators
        revenue = rng.gamma(2.0, 1.5 * mult, n) * visits * (rng.random(n) < 0.3)
        out += [revenue, visits.astype(float)]
    return out  # num_a, den_a, num_b, den_b


def test_delta_method_matches_explicit_linearisation():
    ya, za, yb, zb = ratio_arms(1)
    r = delta_method_ratio(SufficientStats.from_arrays(ya, za), SufficientStats.from_arrays(yb, zb))
    var = 0.0
    for y, z in ((ya, za), (yb, zb)):
        R = y.sum() / z.sum()
        lin = (y - R * z) / z.mean()  # influence-function values
        var += lin.var(ddof=1) / len(y)
    assert r.estimate == pytest.approx(ya.sum() / za.sum() - yb.sum() / zb.sum())
    assert r.diagnostics["se"] == pytest.approx(np.sqrt(var), rel=1e-9)


def test_delta_method_agrees_with_bootstrap():
    ya, za, yb, zb = ratio_arms(2)
    d = delta_method_ratio(SufficientStats.from_arrays(ya, za), SufficientStats.from_arrays(yb, zb))
    b = ratio_bootstrap(ya, za, yb, zb, np.random.default_rng(0), n_resamples=2000)
    assert b.estimate == pytest.approx(d.estimate)
    assert b.diagnostics["se"] == pytest.approx(d.diagnostics["se"], rel=0.07)
    assert (b.ci_low, b.ci_high) == pytest.approx((d.ci_low, d.ci_high), rel=0.15, abs=0.03)


def test_delta_method_with_constant_denominator_reduces_to_welch_on_scaled_mean():
    rng = np.random.default_rng(3)
    ya, yb = rng.normal(5, 2, 800), rng.normal(4.8, 2, 900)
    one_a, one_b = np.ones(800), np.ones(900)
    d = delta_method_ratio(
        SufficientStats.from_arrays(ya, one_a), SufficientStats.from_arrays(yb, one_b)
    )
    w = welch_t(ya, yb)
    assert d.estimate == pytest.approx(w.estimate)
    assert d.diagnostics["se"] == pytest.approx(w.diagnostics["se"], rel=1e-9)


def test_ratio_bootstrap_is_reproducible_and_batched():
    ya, za, yb, zb = ratio_arms(4, na=500, nb=500)
    r1 = ratio_bootstrap(ya, za, yb, zb, np.random.default_rng(5), n_resamples=300, batch=7)
    r2 = ratio_bootstrap(ya, za, yb, zb, np.random.default_rng(5), n_resamples=300, batch=7)
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)


def test_error_paths():
    ya, xa, yb, xb = make_arms(12, na=50, nb=50)
    plain = SufficientStats.from_arrays(ya), SufficientStats.from_arrays(yb)
    for fn in (cuped, regression_adjustment, delta_method_ratio):
        with pytest.raises(ValueError, match="needs x moments"):
            fn(*plain)
    flat = (
        SufficientStats.from_arrays(ya, np.ones(50)),
        SufficientStats.from_arrays(yb, np.ones(50)),
    )
    with pytest.raises(ValueError, match="no variance"):
        cuped(*flat)
    with pytest.raises(ValueError, match="no variance"):
        regression_adjustment(*flat)
    zero = (
        SufficientStats.from_arrays(ya, np.zeros(50)),
        SufficientStats.from_arrays(yb, np.zeros(50)),
    )
    with pytest.raises(ValueError, match="denominator sums to zero"):
        delta_method_ratio(*zero)
    with pytest.raises(ValueError, match="equal length"):
        ratio_bootstrap(ya, xa[:10], yb, xb, np.random.default_rng(0))
