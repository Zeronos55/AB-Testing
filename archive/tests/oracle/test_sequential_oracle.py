"""Oracle tests for the normal-mixture confidence sequence.

The references are independent of the closed forms in the implementation: numerical integration
of the mixture over lambda, brute-force minimisation of the boundary, and Howard et al.'s
closed-form tuning of rho. (The papers' published numbers could not be fetched here.)
"""

import numpy as np
import pytest
from scipy import integrate, stats

from isbetter.methods.sequential import (
    confidence_sequence_diff,
    confidence_sequence_mean,
    mixture_boundary,
    mixture_martingale,
    optimal_rho,
)


@pytest.mark.parametrize("t", [1, 7, 50, 400])
@pytest.mark.parametrize("rho", [0.5, 20.0, 300.0])
@pytest.mark.parametrize("dev", [0.0, 0.1, -0.7])
def test_mixture_martingale_equals_numerical_integral_over_lambda(t, rho, dev):
    s = 1.3
    s_std = t * dev / s  # S_t / s

    def integrand(lam):
        return np.exp(lam * s_std - lam**2 * t / 2) * stats.norm.pdf(lam, scale=rho**-0.5)

    ref, _ = integrate.quad(integrand, -np.inf, np.inf, epsabs=0, epsrel=1e-10, limit=500)
    assert mixture_martingale(t, dev, s, rho) == pytest.approx(ref, rel=1e-7)


def test_martingale_has_expectation_one_under_the_truth():
    rng = np.random.default_rng(0)
    t, rho, s = 40, 25.0, 2.0
    x = rng.normal(0.3, s, size=(400_000, t))
    m = mixture_martingale(t, x.mean(1) - 0.3, s, rho)
    assert m.mean() == pytest.approx(1.0, abs=4 * m.std() / np.sqrt(len(m)))


@pytest.mark.parametrize("alpha", [0.05, 0.01, 0.2])
def test_interval_edge_is_where_the_martingale_reaches_one_over_alpha(alpha):
    rho, s = 80.0, 1.7
    t = np.array([5.0, 30.0, 200.0, 5000.0])
    edge = s * mixture_boundary(t, rho, alpha) / t
    assert mixture_martingale(t, edge, s, rho) == pytest.approx(np.full(4, 1 / alpha), rel=1e-10)


@pytest.mark.parametrize("n_target", [30, 100, 1000, 50_000])
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_optimal_rho_minimises_the_boundary_and_matches_howard_closed_form(n_target, alpha):
    rho = optimal_rho(n_target, alpha)
    grid = rho * np.exp(np.linspace(-1.5, 1.5, 2001))
    width = mixture_boundary(n_target, grid, alpha)
    assert mixture_boundary(n_target, rho, alpha) <= width.min() * (1 + 1e-9)
    howard = n_target / (2 * np.log(1 / alpha) + np.log(1 + 2 * np.log(1 / alpha)))
    assert rho == pytest.approx(howard, rel=0.06)
    assert mixture_boundary(n_target, howard, alpha) == pytest.approx(
        mixture_boundary(n_target, rho, alpha), rel=1e-3
    )


def test_boundary_is_wider_than_the_fixed_horizon_interval_everywhere():
    rho = optimal_rho(500)
    t = np.arange(5, 5000)
    cs = mixture_boundary(t, rho) / t
    fixed = stats.norm.ppf(0.975) / np.sqrt(t)
    assert (cs > fixed).all()
    # at the tuned sample size the price of being allowed to peek is about 3.035 vs 1.96 SDs
    assert mixture_boundary(500, rho) / np.sqrt(500) == pytest.approx(3.035, abs=2e-3)


def test_always_valid_p_value_is_dual_to_the_sequence():
    rng = np.random.default_rng(3)
    for shift in (0.0, 0.25, 0.5):
        x = rng.normal(shift, 1.0, 600)
        cs = confidence_sequence_mean(x, 0.05, n_target=600, sigma=1.0, intersect=False)
        excluded_by_p = cs.p_value <= 0.05
        excluded_by_cs = np.maximum.accumulate(
            (0 < cs.estimate - cs.radius) | (0 > cs.estimate + cs.radius)
        )
        assert (excluded_by_p == excluded_by_cs).all()
        first = cs.first_exclusion()
        assert (first is None) == (not excluded_by_p.any())
        if first is not None:
            assert first == cs.n[np.argmax(excluded_by_p)]
        assert (np.diff(cs.p_value) <= 1e-15).all()  # the running minimum only falls


def test_intersection_only_tightens_and_still_contains_the_final_interval_target():
    x = np.random.default_rng(4).normal(0.2, 1.0, 800)
    raw = confidence_sequence_mean(x, n_target=800, sigma=1.0, intersect=False)
    inter = confidence_sequence_mean(x, n_target=800, sigma=1.0, intersect=True)
    assert (inter.lower >= raw.lower - 1e-15).all() and (inter.upper <= raw.upper + 1e-15).all()
    assert (np.diff(inter.lower) >= -1e-15).all() and (np.diff(inter.upper) <= 1e-15).all()
    assert (inter.lower <= inter.upper).all()


def test_two_arm_sequence_with_known_sigma_is_the_one_stream_sequence_on_the_difference():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.4, 1.0, 300), rng.normal(0.0, 2.0, 300)
    two = confidence_sequence_diff(a, b, n_target=300, sigma=(1.0, 2.0), min_n=1)
    one = confidence_sequence_mean(a - b, n_target=300, sigma=np.sqrt(5.0), min_n=1)
    for name in ("estimate", "lower", "upper", "radius", "p_value"):
        assert getattr(two, name) == pytest.approx(getattr(one, name), abs=1e-12)


def test_plug_in_variance_matches_running_sample_std():
    x = np.random.default_rng(6).normal(1, 3, 200)
    cs = confidence_sequence_mean(x, rho=50.0, min_n=20, intersect=False)
    t = cs.n
    run_sd = np.array([x[:k].std(ddof=1) for k in t])
    assert cs.radius == pytest.approx(run_sd * mixture_boundary(t, 50.0) / t, abs=1e-12)
    assert cs.estimate == pytest.approx([x[:k].mean() for k in t], abs=1e-12)
