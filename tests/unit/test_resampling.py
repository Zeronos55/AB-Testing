import tracemalloc

import numpy as np
import pytest

from isbetter.methods.resampling import bootstrap_means, paired_bootstrap, sign_flip_permutation


def test_reproducible_with_seed():
    a, b = np.arange(30.0), np.arange(30.0)[::-1]
    r1 = paired_bootstrap(a, b, np.random.default_rng(1), n_resamples=500)
    r2 = paired_bootstrap(a, b, np.random.default_rng(1), n_resamples=500)
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)


def test_bad_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap(np.arange(3.0), np.arange(4.0), np.random.default_rng(0))
    with pytest.raises(ValueError, match="method"):
        paired_bootstrap(np.arange(3.0), np.arange(3.0), np.random.default_rng(0), method="x")
    with pytest.raises(ValueError, match="BCa limited"):
        paired_bootstrap(np.zeros(60_000), np.ones(60_000), np.random.default_rng(0), method="bca")


def test_null_permutation_p_not_small():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1, 40)
    r = sign_flip_permutation(d, np.zeros(40), np.random.default_rng(1), n_resamples=999)
    assert 0 < r.p_value <= 1


def test_bootstrap_memory_is_bounded_by_block_not_n_resamples():
    """n=300k with batch=200 must not allocate an n x batch index matrix (~480 MB)."""
    d = np.random.default_rng(0).normal(size=300_000)
    tracemalloc.start()
    bootstrap_means(d, n_resamples=200, rng=np.random.default_rng(1), batch=200)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 600e6
    assert peak < 100e6  # block cap keeps it far below the budget
