"""Oracles for PSR / DSR / MinTRL.

The Bailey & Lopez de Prado papers' own worked examples could not be fetched in the
environment this was written in, so the references here are independent of the implementation:
numerical quadrature and Monte Carlo for the expected maximum of N normals (plus the classical
tabulated values of that quantity), a Monte Carlo check of the Sharpe ratio's sampling variance
formula, a coverage check of the PSR as a probability, and exact algebraic identities.
"""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from isbetter.methods.finance import (
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record,
    min_track_record_length,
    probabilistic_sharpe,
    psr,
    return_moments,
    sharpe_variance_factor,
)

# Expected maximum of N iid standard normals (Harter 1961 tables).
HARTER = {2: 0.56419, 5: 1.16296, 10: 1.53875, 50: 2.24907, 100: 2.50759, 1000: 3.24144}


def exact_expected_max(n: int) -> float:
    """N * int z phi(z) Phi(z)^(N-1) dz by quadrature."""
    f = lambda z: n * z * stats.norm.pdf(z) * stats.norm.cdf(z) ** (n - 1)  # noqa: E731
    return float(integrate.quad(f, -12, 12, epsabs=1e-12, epsrel=1e-12, limit=200)[0])


@pytest.mark.parametrize("n", sorted(HARTER))
def test_quadrature_oracle_reproduces_the_tabulated_values(n):
    assert exact_expected_max(n) == pytest.approx(HARTER[n], abs=2e-5)


@pytest.mark.parametrize("n", [3, 5, 10, 50, 100, 1000])
def test_expected_max_sharpe_matches_exact_expected_maximum_within_the_approximation(n):
    # The closed form approximates E[max]; for N >= 3 it is slightly high (0.5% to 3%).
    exact = exact_expected_max(n)
    assert 0 <= expected_max_sharpe(n, 1.0) / exact - 1 < 0.03


def test_expected_max_sharpe_approximation_is_poor_only_at_two_trials():
    # Known limit of the paper's asymptotic formula: about 8% too low at N = 2.
    assert expected_max_sharpe(2, 1.0) / exact_expected_max(2) - 1 == pytest.approx(
        -0.079, abs=0.005
    )


def test_expected_max_sharpe_formula_hand_values():
    g = 0.5772156649015329
    for n in (10, 100, 1000):
        want = (1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e))
        assert expected_max_sharpe(n, 1.0) == pytest.approx(want, rel=1e-12)
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2.5309, abs=5e-4)
    assert expected_max_sharpe(100, 0.04) == pytest.approx(0.2 * 2.5309, abs=1e-4)  # sqrt(V) scale


def test_expected_max_sharpe_matches_simulation_of_the_best_of_n_null_trials():
    rng = np.random.default_rng(7)
    n, reps, T = 100, 4000, 250
    r = rng.standard_normal((reps, n, T))
    sr = r.mean(2) / r.std(2, ddof=1)
    var = sr.var(1, ddof=1).mean()
    sim = sr.max(1).mean()
    assert expected_max_sharpe(n, var) == pytest.approx(sim, rel=0.03)


def test_expected_max_sharpe_edge_cases():
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert expected_max_sharpe(50, 0.0) == 0.0
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -0.1)


def test_psr_hand_computed_fixture():
    """Direct arithmetic with math.erf, independent of scipy and of the implementation."""
    sr, bench, T, g3, g4 = 0.2, 0.1, 60, -0.7, 5.0
    denom = math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)  # sqrt(1 + 0.14 + 0.04) = sqrt(1.18)
    assert denom == pytest.approx(math.sqrt(1.18))
    z = (sr - bench) * math.sqrt(T - 1) / denom
    want = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert psr(sr, bench, T, g3, g4) == pytest.approx(want, rel=1e-12)
    assert want == pytest.approx(0.8, abs=0.05)  # sanity: z ~ 0.707


def test_psr_normal_case_reduces_to_one_plus_half_sr_squared():
    sr, T = 0.3, 101
    z = sr * math.sqrt(T - 1) / math.sqrt(1 + sr**2 / 2)
    assert psr(sr, 0.0, T) == pytest.approx(stats.norm.cdf(z), rel=1e-12)
    assert sharpe_variance_factor(sr, 0.0, 3.0) == pytest.approx(1 + sr**2 / 2)


def test_sharpe_sampling_variance_formula_against_monte_carlo():
    """Var(SR_hat) ~ (1 - skew SR + (kurt-1)/4 SR^2) / T for skewed, fat-tailed returns."""
    rng = np.random.default_rng(11)
    T, reps = 1000, 20_000
    # Shifted exponential: skew 2, raw kurtosis 9; SR = mu / sigma = 0.1.
    sigma = 0.01
    mu = 0.1 * sigma
    x = mu + sigma * (rng.exponential(1.0, (reps, T)) - 1.0)
    sr_hat = x.mean(1) / x.std(1, ddof=1)
    factor = sharpe_variance_factor(0.1, 2.0, 9.0)  # 1 - 0.2 + 2 * 0.01 = 0.82
    assert factor == pytest.approx(0.82)
    assert sr_hat.var(ddof=1) == pytest.approx(factor / T, rel=0.05)
    # the normal-returns formula (1 + SR^2/2)/T overstates the variance by 22% here
    assert sr_hat.var(ddof=1) < (1 + 0.01 / 2) / T / 1.15


def test_psr_at_the_true_sharpe_ratio_is_calibrated_as_a_probability():
    """P(PSR(true SR) < 0.05) should be about 0.05 over repeated samples."""
    rng = np.random.default_rng(13)
    T, reps = 750, 6000
    sigma = 0.01
    x = 0.08 * sigma + sigma * (rng.exponential(1.0, (reps, T)) - 1.0)
    mean, sd = x.mean(1), x.std(1, ddof=1)
    c = x - mean[:, None]
    skew = (c**3).mean(1) / (c**2).mean(1) ** 1.5
    kurt = (c**4).mean(1) / (c**2).mean(1) ** 2
    sr = mean / sd
    factor = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    p = stats.norm.cdf((sr - 0.08) * np.sqrt(T - 1) / np.sqrt(factor))
    # the same vectorised formula as psr(); check one row against the function too
    assert psr(sr[0], 0.08, T, skew[0], kurt[0]) == pytest.approx(p[0], rel=1e-9)
    assert (p < 0.05).mean() == pytest.approx(0.05, abs=0.012)
    assert (p > 0.95).mean() == pytest.approx(0.05, abs=0.012)


def test_probabilistic_sharpe_matches_function_on_a_return_series():
    r = np.random.default_rng(3).standard_t(5, 400) * 0.01 + 0.0006
    m = return_moments(r)
    res = probabilistic_sharpe(r, sr_benchmark=0.02)
    assert res.diagnostics["psr"] == pytest.approx(
        psr(m.sharpe, 0.02, m.n, m.skew, m.kurtosis), rel=1e-12
    )
    assert res.p_value == pytest.approx(1 - res.diagnostics["psr"])
    assert res.estimate == pytest.approx(r.mean() / r.std(ddof=1))
    assert res.ci_low < res.estimate < res.ci_high


def test_return_moments_match_scipy_definitions():
    r = np.random.default_rng(5).standard_normal(300) ** 3
    m = return_moments(r)
    assert m.skew == pytest.approx(stats.skew(r))
    assert m.kurtosis == pytest.approx(stats.kurtosis(r) + 3.0)
    assert m.std == pytest.approx(np.std(r, ddof=1))


@pytest.mark.parametrize(
    ("sr", "bench", "g3", "g4", "alpha"),
    [(0.2, 0.0, 0.0, 3.0, 0.05), (0.15, 0.05, -1.5, 8.0, 0.05), (0.4, 0.1, 0.3, 4.0, 0.01)],
)
def test_psr_at_min_track_record_length_is_exactly_one_minus_alpha(sr, bench, g3, g4, alpha):
    t = min_track_record_length(sr, bench, g3, g4, alpha)
    # PSR depends on T - 1 only through sqrt(T - 1); evaluate it at the (non-integer) length
    factor = sharpe_variance_factor(sr, g3, g4)
    z = (sr - bench) * math.sqrt(t - 1) / math.sqrt(factor)
    assert stats.norm.cdf(z) == pytest.approx(1 - alpha, abs=1e-12)
    # and with whole observations: ceil(MinTRL) clears the bar, floor(MinTRL) - 1 does not
    assert psr(sr, bench, math.ceil(t), g3, g4) >= 1 - alpha - 1e-12
    assert psr(sr, bench, math.floor(t) - 1, g3, g4) < 1 - alpha


def test_min_track_record_hand_value_for_normal_returns():
    # SR = 0.2 vs 0: 1 + (1 + 0.02) * (1.6449 / 0.2)^2
    want = 1 + 1.02 * (stats.norm.ppf(0.95) / 0.2) ** 2
    assert min_track_record_length(0.2, 0.0) == pytest.approx(want, rel=1e-12)
    assert want == pytest.approx(70.0, abs=0.5)


def test_min_track_record_grows_with_fat_tails_and_negative_skew():
    base = min_track_record_length(0.2, 0.0, 0.0, 3.0)
    assert min_track_record_length(0.2, 0.0, -1.0, 3.0) > base
    assert min_track_record_length(0.2, 0.0, 0.0, 9.0) > base
    assert math.isinf(min_track_record_length(0.1, 0.1))
    assert math.isinf(min_track_record_length(-0.1, 0.0))


def test_min_track_record_result_reports_whether_the_record_is_long_enough():
    r = np.random.default_rng(2).normal(0.002, 0.01, 500)  # SR ~ 0.2
    res = min_track_record(r)
    assert res.diagnostics["enough"] is (500 >= res.estimate)
    short = min_track_record(r[:20])
    assert short.estimate > 20 and short.diagnostics["enough"] is False


def test_dsr_is_psr_at_the_expected_max_sharpe():
    r = np.random.default_rng(9).normal(0.0015, 0.01, 1000)
    m = return_moments(r)
    sr0 = expected_max_sharpe(200, 0.0004)
    res = deflated_sharpe(r, n_trials=200, sr_variance=0.0004)
    assert res.diagnostics["sr_star"] == pytest.approx(sr0)
    assert res.diagnostics["dsr"] == pytest.approx(
        psr(m.sharpe, sr0, m.n, m.skew, m.kurtosis), rel=1e-12
    )
    assert res.diagnostics["psr_vs_zero"] == pytest.approx(
        psr(m.sharpe, 0.0, m.n, m.skew, m.kurtosis), rel=1e-12
    )
    assert res.diagnostics["dsr"] < res.diagnostics["psr_vs_zero"]
    assert res.p_value == pytest.approx(1 - res.diagnostics["dsr"])


def test_dsr_with_one_trial_equals_psr_against_zero():
    r = np.random.default_rng(10).normal(0.001, 0.01, 500)
    res = deflated_sharpe(r, n_trials=1, sr_variance=0.5)
    assert res.diagnostics["sr_star"] == 0.0
    assert res.diagnostics["dsr"] == pytest.approx(probabilistic_sharpe(r).diagnostics["psr"])
