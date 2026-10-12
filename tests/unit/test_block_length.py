import builtins

import numpy as np
import pytest
from arch.bootstrap import optimal_block_length

from isbetter.methods.forecasting import BlockLength, suggest_block_length


def ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_wrapper_returns_archs_politis_white_lengths():
    x = ar1(0.5, 400, 0)
    ref = optimal_block_length(x)
    got = suggest_block_length(x)
    assert got.source == "politis_white"
    assert got.stationary == pytest.approx(float(ref["stationary"].iloc[0]))
    assert got.circular == pytest.approx(float(ref["circular"].iloc[0]))


def test_more_autocorrelation_gives_longer_blocks():
    weak, strong = suggest_block_length(ar1(0.0, 600, 1)), suggest_block_length(ar1(0.8, 600, 1))
    assert strong.stationary > 2 * weak.stationary


def test_integer_rounding_is_clipped():
    bl = BlockLength(stationary=3.2, circular=0.2, source="politis_white")
    assert bl.integer("stationary", 100) == 3
    assert bl.integer("circular", 100) == 1
    assert BlockLength(500.0, 500.0, "politis_white").integer("circular", 50) == 50


def test_constant_series_falls_back_to_rule_of_thumb():
    got = suggest_block_length(np.ones(64))
    assert got.source == "rule_of_thumb" and got.stationary == 4.0


def test_falls_back_when_arch_is_not_installed(monkeypatch):
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("arch"):
            raise ImportError("no arch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    got = suggest_block_length(ar1(0.5, 125, 2))
    assert got == BlockLength(5.0, 5.0, "rule_of_thumb")


def test_validation():
    with pytest.raises(ValueError, match="at least 3"):
        suggest_block_length(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        suggest_block_length(np.array([1.0, np.nan, 2.0, 3.0]))
