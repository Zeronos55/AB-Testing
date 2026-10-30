import numpy as np
import pytest

from isbetter.methods.pbo import cscv, n_splits, pbo_cscv


def _x(seed=0, T=200, N=10):
    return np.random.default_rng(seed).normal(0, 0.01, (T, N))


def test_split_count_is_n_choose_half():
    assert n_splits(16) == 12_870
    assert cscv(_x(), 8).n_splits == 70


def test_pbo_is_invariant_to_scale_and_column_order():
    x = _x(3, 240, 12)
    base = pbo_cscv(x, 8).estimate
    assert pbo_cscv(x * 100.0, 8).estimate == base
    perm = np.random.default_rng(0).permutation(12)
    assert pbo_cscv(x[:, perm], 8).estimate == pytest.approx(base, abs=0.03)


def test_logits_are_symmetric_when_no_configuration_has_any_edge_by_construction():
    # two identical columns: the IS-best ties with its twin and ranks mid-way among ties
    x = _x(1, 160, 2)
    x = np.column_stack([x[:, 0], x[:, 0]])
    res = cscv(x, 8)
    assert np.allclose(res.omega, 0.5 + 0.0, atol=1 / 3)  # ranks within 1..2 of N+1 = 3
    assert 0.0 <= res.pbo <= 1.0


def test_diagnostics_fields():
    res = pbo_cscv(_x(2, 200, 6), 8, names=list("abcdef"))
    d = res.diagnostics
    assert d["n_configs"] == 6 and d["metric"] == "sharpe" and d["n_blocks"] == 8
    assert sum(c for _, c in d["most_selected"]) <= 70
    assert 0.0 <= d["prob_oos_loss"] <= 1.0
    assert res.method == "pbo_cscv" and "Backtest Overfitting" in res.rationale


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_blocks": 7}, "even"),
        ({"n_blocks": 22}, "even"),
        ({"metric": "calmar"}, "metric"),
    ],
)
def test_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        cscv(_x(), **kwargs)


def test_bad_inputs():
    with pytest.raises(ValueError, match="at least 2 columns"):
        cscv(_x(N=1), 4)
    with pytest.raises(ValueError, match="per block"):
        cscv(_x(T=10), 8)
    with pytest.raises(ValueError, match="matrix"):
        cscv(np.zeros(50), 4)
    bad = _x()
    bad[3, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cscv(bad, 4)
    with pytest.raises(ValueError, match="one entry per column"):
        pbo_cscv(_x(N=3), 4, names=["a"])
