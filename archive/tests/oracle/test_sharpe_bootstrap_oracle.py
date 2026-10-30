"""Oracle: on iid returns the block bootstrap (block length 1) must reproduce the analytic
normal-approximation interval of the Sharpe ratio, and an independent plain-NumPy bootstrap."""

import numpy as np
import pytest

from isbetter.methods.finance import probabilistic_sharpe, return_moments, sharpe_block_bootstrap


def test_iid_bootstrap_interval_matches_the_analytic_interval():
    r = np.random.default_rng(3).standard_t(8, 2000) * 0.01 + 0.0008
    boot = sharpe_block_bootstrap(r, np.random.default_rng(1), block_len=1, n_resamples=9999)
    analytic = probabilistic_sharpe(r)
    assert boot.diagnostics["se"] == pytest.approx(analytic.diagnostics["se"], rel=0.06)
    assert boot.ci_low == pytest.approx(analytic.ci_low, abs=0.012)
    assert boot.ci_high == pytest.approx(analytic.ci_high, abs=0.012)


def test_matches_a_literal_numpy_bootstrap_of_the_same_statistic():
    r = np.random.default_rng(4).normal(0.0005, 0.01, 400)
    ref_rng = np.random.default_rng(99)
    ref = np.array(
        [
            (lambda s: s.mean() / s.std(ddof=1))(r[ref_rng.integers(0, len(r), len(r))])
            for _ in range(6000)
        ]
    )
    boot = sharpe_block_bootstrap(r, np.random.default_rng(2), block_len=1, n_resamples=6000)
    lo, hi = np.quantile(ref, [0.025, 0.975])
    assert boot.estimate == pytest.approx(return_moments(r).sharpe)
    assert boot.ci_low == pytest.approx(lo, abs=0.015)
    assert boot.ci_high == pytest.approx(hi, abs=0.015)


def test_block_bootstrap_is_wider_than_iid_for_autocorrelated_returns():
    rng = np.random.default_rng(5)
    e = rng.normal(0, 0.01, 3000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = 0.4 * x[t - 1] + e[t]
    x += 0.0006
    iid = sharpe_block_bootstrap(x, np.random.default_rng(0), block_len=1, n_resamples=2000)
    blk = sharpe_block_bootstrap(x, np.random.default_rng(0), n_resamples=2000)
    # long-run variance inflation (1 + phi) / (1 - phi) = 2.33 -> SE factor ~ 1.53
    ratio = blk.diagnostics["se"] / iid.diagnostics["se"]
    assert ratio == pytest.approx(1.53, rel=0.2)
