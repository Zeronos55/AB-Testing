"""Oracles for the block bootstrap: Politis-Romano closed form and arch's bootstrap classes."""

import numpy as np
import pytest
from arch.bootstrap import CircularBlockBootstrap, StationaryBootstrap

from isbetter.methods.resampling import block_bootstrap

B = 30_000


def ma_series(seed, T=200):
    e = np.random.default_rng(seed).standard_normal(T + 2)
    return e[2:] + 0.8 * e[1:-1] + 0.5 * e[:-2] + 0.2


def stationary_bootstrap_variance(d, block_len):
    """Exact variance of the stationary-bootstrap mean (Politis & Romano 1994, Lemma 1)."""
    T, p = len(d), 1.0 / block_len
    c = d - d.mean()
    R = np.array([(c * np.roll(c, -k)).sum() / T for k in range(T)])  # circular autocovariances
    k = np.arange(1, T)
    return (R[0] + 2 * ((1 - k / T) * (1 - p) ** k * R[1:]).sum()) / T


@pytest.mark.parametrize("block_len", [1, 5, 12])
def test_stationary_se_matches_closed_form(block_len):
    d = ma_series(block_len)
    r = block_bootstrap(
        d, np.zeros(len(d)), np.random.default_rng(1), block_len, "stationary", n_resamples=B
    )
    exact = np.sqrt(stationary_bootstrap_variance(d, block_len))
    assert r.diagnostics["se"] == pytest.approx(exact, rel=0.02)  # MC rel. SE ~ 0.4%


@pytest.mark.parametrize("kind", ["stationary", "circular"])
@pytest.mark.parametrize("block_len", [1, 5, 12])
def test_se_matches_arch(kind, block_len):
    d = ma_series(10 + block_len)
    r = block_bootstrap(d, np.zeros(len(d)), np.random.default_rng(2), block_len, kind, B)
    cls = StationaryBootstrap if kind == "stationary" else CircularBlockBootstrap
    bs = cls(block_len, d, seed=np.random.default_rng(3))
    ref = float(np.sqrt(bs.var(lambda x: x.mean(), reps=B)[0]))
    assert r.diagnostics["se"] == pytest.approx(ref, rel=0.03)


def test_block_len_one_is_the_iid_bootstrap():
    d = ma_series(7)
    r = block_bootstrap(d, np.zeros(len(d)), np.random.default_rng(4), 1, "circular", B)
    assert r.diagnostics["se"] == pytest.approx(d.std(ddof=0) / np.sqrt(len(d)), rel=0.02)


def test_percentile_ci_matches_arch_interval():
    d = ma_series(8)
    r = block_bootstrap(d, np.zeros(len(d)), np.random.default_rng(5), 6, "stationary", B)
    bs = StationaryBootstrap(6, d, seed=np.random.default_rng(6))
    lo, hi = bs.conf_int(lambda x: x.mean(), reps=B, method="percentile", size=0.95).ravel()
    width = hi - lo
    assert r.ci_low == pytest.approx(lo, abs=0.04 * width)
    assert r.ci_high == pytest.approx(hi, abs=0.04 * width)
