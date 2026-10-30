"""Coverage of intervals at small N (LLM evals with 30-200 items).

The single-arm checks are exact (the count is binomial, so coverage is a finite sum) and run in
the fast suite; the paired and clustered checks simulate and are marked slow.
"""

import numpy as np
import pytest

from isbetter.methods.proportions import jeffreys_interval, wilson_interval
from isbetter.methods.small_n import clopper_pearson_interval
from isbetter.simulate.calibration import (
    clt_proportion_interval,
    exact_proportion_coverage,
    paired_binary_coverage,
    paired_cluster_type1,
)


def test_clt_interval_undercovers_at_small_n_and_extreme_accuracy():
    # n = 30, true accuracy 0.95: the normal interval misses far more than 5% of the time.
    cov = exact_proportion_coverage(clt_proportion_interval, 30, 0.95)
    assert cov < 0.85
    assert exact_proportion_coverage(clt_proportion_interval, 30, 0.5) > 0.9  # fine mid-range


def test_clopper_pearson_never_undercovers_and_wilson_jeffreys_stay_close():
    ps = np.linspace(0.02, 0.98, 49)
    for n in (30, 50, 100, 200):
        cp = [exact_proportion_coverage(clopper_pearson_interval, n, p) for p in ps]
        assert min(cp) >= 0.95 - 1e-9  # guaranteed coverage
        for interval in (wilson_interval, jeffreys_interval):
            cov = [exact_proportion_coverage(interval, n, p) for p in ps]
            assert np.mean(cov) > 0.935 and min(cov) > 0.80


def test_clt_is_worse_than_wilson_on_average_coverage_for_every_n_in_range():
    ps = np.linspace(0.02, 0.98, 49)
    for n in (30, 50, 100, 200):
        clt = np.mean([exact_proportion_coverage(clt_proportion_interval, n, p) for p in ps])
        wil = np.mean([exact_proportion_coverage(wilson_interval, n, p) for p in ps])
        assert clt < wil


@pytest.mark.slow
@pytest.mark.parametrize(("n", "p_a", "p_b"), [(40, 0.8, 0.75), (60, 0.95, 0.9)])
def test_score_interval_covers_paired_difference_near_nominal(request, n, p_a, p_b):
    reps = min(request.config.getoption("--reps"), 4000)
    wald, score = paired_binary_coverage(np.random.default_rng(11), reps, n, p_a, p_b, 0.4)
    assert score.rate > 0.935, f"score interval coverage {score.rate:.3f}"
    assert score.rate >= wald.rate - 0.005


@pytest.mark.slow
def test_paired_cluster_robust_controls_type1_while_naive_paired_t_does_not(request):
    reps = min(request.config.getoption("--reps"), 6000)
    naive, cr1 = paired_cluster_type1(np.random.default_rng(5), reps, groups=30, per_group=5)
    assert naive.rate > 0.15, f"naive paired t rejects {naive.rate:.3f}"
    assert 0.04 < cr1.rate < 0.065, f"CR1 rejects {cr1.rate:.3f}"  # mildly liberal at G = 30
