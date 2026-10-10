"""Calibration of the Phase 2 methods: Type I error on null worlds must sit at alpha."""

import numpy as np
import pytest

from isbetter.simulate.calibration import (
    cuped_baseline_pvalues,
    cuped_pvalues,
    srm_rejection_rate,
    two_proportion_pvalues,
    type1_rate,
)
from isbetter.simulate.dgp import aa_binary, aa_covariate

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("rho, curvature", [(0.0, 0.0), (0.5, 0.0), (0.9, 0.0), (0.7, 1.0)])
def test_cuped_type1_stays_nominal(request, rho, curvature):
    """Includes a curved covariate-outcome relation that the linear adjustment cannot remove."""
    reps = request.config.getoption("--reps")
    seed = 210 + int(rho * 10) + int(curvature * 100)
    r = type1_rate(
        cuped_pvalues,
        lambda rng, m, n: aa_covariate(rng, m, n, rho=rho, curvature=curvature),
        np.random.default_rng(seed),
        reps=reps,
        n=400,
    )
    msg = f"CUPED (rho={rho}, curvature={curvature}) rate {r.rate:.4f} outside {r.band}"
    assert r.passed, msg


def test_unadjusted_baseline_also_nominal(request):
    reps = request.config.getoption("--reps")
    r = type1_rate(
        cuped_baseline_pvalues,
        lambda rng, m, n: aa_covariate(rng, m, n, rho=0.7),
        np.random.default_rng(22),
        reps=reps,
        n=400,
    )
    assert r.passed, f"unadjusted rejection rate {r.rate:.4f} outside {r.band}"


@pytest.mark.parametrize("p", [0.1, 0.3])
def test_two_proportion_type1(request, p):
    reps = request.config.getoption("--reps")
    r = type1_rate(
        two_proportion_pvalues,
        lambda rng, m, n: aa_binary(rng, m, n, p=p),
        np.random.default_rng(23),
        reps=reps,
        n=1500,
    )
    assert r.passed, f"two-proportion z (p={p}) rejection rate {r.rate:.4f} outside {r.band}"


def test_srm_false_alarm_rate_at_nominal(request):
    reps = request.config.getoption("--reps")
    r = srm_rejection_rate(np.random.default_rng(24), reps, 20_000, (0.4, 0.3, 0.3), alpha=0.05)
    assert r.passed, f"SRM false-alarm rate {r.rate:.4f} outside {r.band}"


def test_cluster_robust_nominal_while_naive_row_analysis_is_inflated(request):
    """Gallery #5 as a test: users randomised, 10 sessions each, ICC 0.3 (design effect 3.7).

    CR1 with t(G - 1) inference is known to be marginally liberal at finite G (its variance is
    about 0.5% small at G = 200; pooled over 80,000 replications the rate is 0.051), so this
    test uses a 3 Monte Carlo SE band instead of the 2 SE band of the exact methods above.
    """
    from isbetter.simulate.calibration import cluster_type1

    reps = request.config.getoption("--reps")
    naive, cr1 = cluster_type1(np.random.default_rng(25), reps, users=100, sessions=10, icc=0.3)
    half = 3 * np.sqrt(0.05 * 0.95 / reps)
    assert abs(cr1.rate - 0.05) <= half, f"CR1 rate {cr1.rate:.4f} outside 0.05 +/- {half:.4f}"
    assert naive.rate > 0.20, f"naive row-level rate {naive.rate:.3f} should be badly inflated"
