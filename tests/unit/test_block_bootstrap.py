import numpy as np
import pytest

from isbetter.methods.resampling import _block_indices, block_bootstrap


@pytest.mark.parametrize("kind", ["stationary", "circular"])
def test_indices_are_valid_and_blocks_are_contiguous(kind):
    n = 50
    idx = _block_indices(kind, n, 7, 40, np.random.default_rng(0))
    assert idx.shape == (40, n) and idx.min() >= 0 and idx.max() < n
    step = np.diff(idx, axis=1) % n
    # most steps continue a block (+1); block boundaries are the rest
    assert (step == 1).mean() > 0.7


def test_circular_blocks_have_fixed_length():
    idx = _block_indices("circular", 40, 8, 25, np.random.default_rng(1))
    jumps = np.diff(idx, axis=1) % 40 != 1
    assert not jumps[:, np.r_[0:7, 8:15, 16:23, 24:31, 32:39]].any()  # inside the 5 blocks


def test_block_len_equal_to_n_resamples_whole_series_shifts():
    d = np.arange(10.0)
    r = block_bootstrap(d, np.zeros(10), np.random.default_rng(2), 10, "circular", 200)
    assert r.ci_low == r.ci_high == pytest.approx(4.5)  # a circular shift keeps the mean


def test_reproducible_and_batch_invariant_in_distribution():
    d = np.random.default_rng(3).standard_normal(80)
    a = block_bootstrap(d, np.zeros(80), np.random.default_rng(9), 5, n_resamples=500)
    b = block_bootstrap(d, np.zeros(80), np.random.default_rng(9), 5, n_resamples=500)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert a.p_value is None and a.method == "stationary_block_bootstrap"
    assert a.ci_low < d.mean() < a.ci_high and "block" in a.rationale


def test_validation():
    d = np.arange(10.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="kind"):
        block_bootstrap(d, d, rng, 3, kind="moving")
    with pytest.raises(ValueError, match="block_len"):
        block_bootstrap(d, d, rng, 0)
    with pytest.raises(ValueError, match="block_len"):
        block_bootstrap(d, d, rng, 11)
    with pytest.raises(ValueError, match="equal length"):
        block_bootstrap(d, d[:-1], rng, 3)


@pytest.mark.parametrize("kind", ["stationary", "circular"])
def test_block_bootstrap_memory_is_bounded_by_block_not_n_resamples(kind):
    """n = 300k with batch = 200 must not allocate a batch x n index matrix (~480 MB)."""
    import tracemalloc

    d = np.random.default_rng(0).normal(size=300_000)
    tracemalloc.start()
    block_bootstrap(d, np.zeros_like(d), np.random.default_rng(1), 50, kind, n_resamples=40)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 150e6
