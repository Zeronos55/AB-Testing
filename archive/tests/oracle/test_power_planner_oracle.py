"""Sample-size planner vs statsmodels.stats.power (ticket 2.6: within 1%)."""

import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from isbetter.methods.power import (
    mde_means,
    mde_proportions,
    power_means,
    power_proportions,
    sample_size_means,
    sample_size_proportions,
)
from isbetter.simulate.calibration import two_proportion_pvalues

GRID = [
    (d, a, pw, ratio, two)
    for d in (0.05, 0.2, 0.5, 1.0)
    for a in (0.05, 0.01)
    for pw in (0.8, 0.9)
    for ratio in (1.0, 0.5, 3.0)
    for two in (True, False)
]


@pytest.mark.parametrize("d, alpha, power, ratio, two", GRID)
def test_means_sample_size_within_1pct_of_statsmodels(d, alpha, power, ratio, two):
    sd = 7.0
    ours = sample_size_means(d * sd, sd, alpha, power, ratio, two)
    ref = TTestIndPower().solve_power(
        effect_size=d, alpha=alpha, power=power, ratio=ratio,
        alternative="two-sided" if two else "larger",
    )  # fmt: skip
    assert ours.n_control_exact == pytest.approx(ref, rel=0.01)
    assert ours.n_control >= ours.n_control_exact  # rounded up, never down
    assert ours.power_achieved >= power - 1e-9


@pytest.mark.parametrize(
    "n, d, ratio", [(40, 0.5, 1.0), (400, 0.2, 1.0), (150, 0.4, 2.0), (30, 1.0, 0.5)]
)
def test_means_power_matches_statsmodels(n, d, ratio):
    ref = TTestIndPower().power(effect_size=d, nobs1=n, alpha=0.05, ratio=ratio)
    assert power_means(n, d * 3.0, 3.0, 0.05, ratio) == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize("n, ratio", [(100, 1.0), (500, 1.0), (300, 2.0)])
def test_means_mde_within_1pct_of_statsmodels(n, ratio):
    ref = TTestIndPower().solve_power(nobs1=n, alpha=0.05, power=0.8, ratio=ratio) * 5.0
    assert mde_means(n, 5.0, 0.05, 0.8, ratio) == pytest.approx(ref, rel=0.01)


PGRID = [
    (p, lift, a, pw, ratio, two)
    for p in (0.02, 0.1, 0.4)
    for lift in (0.005, 0.02, -0.01)
    for a in (0.05, 0.01)
    for pw in (0.8, 0.9)
    for ratio in (1.0, 2.0)
    for two in (True, False)
]


@pytest.mark.parametrize("p, lift, alpha, power, ratio, two", PGRID)
def test_proportions_arcsine_within_1pct_of_statsmodels(p, lift, alpha, power, ratio, two):
    if p + lift <= 0:
        pytest.skip("rate would be non-positive")
    ours = sample_size_proportions(p, lift, alpha, power, ratio, two, method="arcsine")
    h = proportion_effectsize(p + lift, p)
    ref = NormalIndPower().solve_power(
        effect_size=h, alpha=alpha, power=power, ratio=ratio,
        alternative="two-sided" if two else ("larger" if h > 0 else "smaller"),
    )  # fmt: skip
    assert ours.n_control_exact == pytest.approx(ref, rel=0.01)


def test_proportions_power_and_mde_match_statsmodels():
    h = proportion_effectsize(0.12, 0.10)
    ref = NormalIndPower().power(effect_size=h, nobs1=3000, alpha=0.05, ratio=1.0)
    assert power_proportions(3000, 0.10, 0.02, 0.05, 1.0) == pytest.approx(ref, abs=1e-8)
    lift = mde_proportions(3000, 0.10, 0.05, 0.8, 1.0)
    h_ref = NormalIndPower().solve_power(nobs1=3000, alpha=0.05, power=0.8, ratio=1.0)
    # statsmodels' root finder is only accurate to about 1e-6 relative; ours is a bracketed solve
    h_ours = 2 * np.arcsin(np.sqrt(0.10 + lift)) - 2 * np.arcsin(np.sqrt(0.10))
    assert h_ours == pytest.approx(h_ref, rel=1e-5)
    assert power_proportions(3000, 0.10, lift, 0.05, 1.0) == pytest.approx(0.8, abs=1e-9)


def test_pooled_method_matches_hand_calculation_and_simulation():
    plan = sample_size_proportions(0.10, 0.02, method="pooled")
    # Fleiss: (z_a sqrt(2 pbar qbar) + z_b sqrt(p1 q1 + p2 q2))^2 / delta^2
    za, zb = 1.959964, 0.841621
    n = (za * np.sqrt(2 * 0.11 * 0.89) + zb * np.sqrt(0.1 * 0.9 + 0.12 * 0.88)) ** 2 / 0.02**2
    assert plan.n_control_exact == pytest.approx(n, rel=1e-4)
    # Monte Carlo: the z-test at this n rejects about 80% of the time
    rng = np.random.default_rng(0)
    m = plan.n_control
    a = (rng.random((20_000, m)) < 0.12).astype(np.int8)
    b = (rng.random((20_000, m)) < 0.10).astype(np.int8)
    assert (two_proportion_pvalues(a, b) < 0.05).mean() == pytest.approx(0.80, abs=0.015)


def test_cuped_rho_shrinks_required_n_by_one_minus_rho_squared():
    base = sample_size_means(1.0, 5.0)
    adj = sample_size_means(1.0, 5.0, rho=0.6)
    assert adj.n_control_exact / base.n_control_exact == pytest.approx(1 - 0.36, rel=0.02)
    pb = sample_size_proportions(0.1, 0.02)
    pa = sample_size_proportions(0.1, 0.02, rho=0.5)
    assert pa.n_control_exact == pytest.approx(0.75 * pb.n_control_exact)


def test_plan_totals_and_rounding():
    p = sample_size_means(1.0, 5.0, ratio=2.0)
    assert p.n_treatment >= 2 * p.n_control_exact - 1e-9
    assert p.n_total == p.n_control + p.n_treatment


@pytest.mark.parametrize(
    "call",
    [
        lambda: sample_size_means(0.0, 1.0),
        lambda: sample_size_means(1.0, -1.0),
        lambda: sample_size_means(1.0, 1.0, alpha=1.5),
        lambda: sample_size_means(1.0, 1.0, rho=1.0),
        lambda: sample_size_proportions(0.0, 0.01),
        lambda: sample_size_proportions(0.99, 0.02),
        lambda: sample_size_proportions(0.1, 0.01, method="exact"),
        lambda: sample_size_proportions(0.1, 0.01, ratio=0),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
