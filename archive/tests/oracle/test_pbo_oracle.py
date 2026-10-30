"""Oracle for CSCV: a slow, literal transcription of the algorithm in the paper."""

import time
from itertools import combinations

import numpy as np
import pytest

from isbetter.methods.pbo import cscv, n_splits, pbo_cscv


def reference_cscv(x, S, metric="sharpe"):
    """Loop over splits, slice the returns, rank with scipy-free plain numpy."""
    T, N = x.shape
    blocks = np.array_split(np.arange(T), S)

    def perf(rows):
        sub = x[rows]
        if metric == "mean":
            return sub.mean(0)
        sd = sub.std(0, ddof=1)
        return np.where(sd > 0, sub.mean(0) / np.where(sd > 0, sd, 1), 0.0)

    logits = []
    for js in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[j] for j in js])
        oos_rows = np.concatenate([blocks[j] for j in range(S) if j not in js])
        best = int(np.argmax(perf(is_rows)))
        oos = perf(oos_rows)
        rank = (
            sum(1.0 for v in oos if v < oos[best]) + sum(0.5 for v in oos if v == oos[best]) + 0.5
        )
        omega = rank / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
    return np.array(logits)


@pytest.mark.parametrize("metric", ["sharpe", "mean"])
@pytest.mark.parametrize(("T", "N", "S"), [(60, 5, 6), (97, 8, 8), (200, 30, 10)])
def test_vectorised_cscv_equals_the_literal_algorithm(T, N, S, metric):
    x = np.random.default_rng(T + N).standard_t(6, (T, N)) * 0.01 + 0.0003
    ours = cscv(x, S, metric)
    ref = reference_cscv(x, S, metric)
    assert ours.n_splits == n_splits(S) == len(ref)
    np.testing.assert_allclose(ours.logits, ref, atol=1e-12)


def test_uneven_blocks_and_flat_strategies_match_the_reference():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 0.01, (103, 6))
    x[:, 2] = 0.0  # a configuration that never trades
    np.testing.assert_allclose(cscv(x, 6).logits, reference_cscv(x, 6), atol=1e-12)


def test_pure_noise_gives_pbo_near_one_half():
    # One dataset's PBO is noisy (sd ~ 0.14: the 12,870 splits overlap), its mean is 0.5.
    pbos = [
        pbo_cscv(np.random.default_rng(s).normal(0, 0.01, (1600, 100)), 16).estimate
        for s in range(30)
    ]
    sem = np.std(pbos, ddof=1) / np.sqrt(len(pbos))
    assert abs(np.mean(pbos) - 0.5) < 3 * sem + 0.01, (np.mean(pbos), sem)


def test_one_planted_strong_edge_gives_low_pbo():
    pbos = []
    for seed in range(10):
        x = np.random.default_rng(100 + seed).normal(0, 0.01, (1600, 100))
        x[:, 17] += 0.002  # per-period Sharpe 0.2: t ~ 8 over the sample
        res = pbo_cscv(x, 16, names=[f"c{j}" for j in range(100)])
        pbos.append(res.estimate)
        assert res.diagnostics["most_selected"][0][0] == "c17"
    assert max(pbos) < 0.05


def test_s16_on_a_few_hundred_configurations_runs_in_under_30_seconds():
    x = np.random.default_rng(1).normal(0, 0.01, (2500, 400))
    t0 = time.perf_counter()
    res = pbo_cscv(x, 16)
    elapsed = time.perf_counter() - t0
    assert res.diagnostics["n_splits"] == 12_870
    assert elapsed < 30, f"took {elapsed:.1f}s"
