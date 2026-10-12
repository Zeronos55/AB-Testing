"""Helpers of case_studies/04_trading_edge/run.py: leak-free signals, registry, data loaders."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from isbetter.adapters.strategy_returns import TrialRegistry

RUN = Path(__file__).resolve().parents[2] / "case_studies" / "04_trading_edge" / "run.py"
spec = importlib.util.spec_from_file_location("trading_case_study", RUN)
cs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cs  # dataclasses resolves string annotations through sys.modules
spec.loader.exec_module(cs)


def _prices(seed=0, n=420):
    return 100 * np.cumprod(1 + np.random.default_rng(seed).normal(0.0002, 0.01, n))


def test_grid_has_a_few_hundred_variants_with_fast_below_slow():
    assert 250 <= len(cs.GRID) <= 400
    assert all(f < s for f, s in cs.GRID) and len(set(cs.GRID)) == len(cs.GRID)


def test_excess_return_matches_a_literal_loop():
    p = _prices()
    grid = cs.MovingAverageGrid(p)
    fast, slow = 3, 7
    r = p[1:] / p[:-1] - 1
    want = []
    for i in range(cs.WARMUP - 1, len(r)):  # decided at the close of price i, earns r[i]
        pos = float(p[i - fast + 1 : i + 1].mean() > p[i - slow + 1 : i + 1].mean())
        want.append((pos - 1.0) * r[i])
    np.testing.assert_allclose(grid.compute(fast, slow), want, atol=1e-14)
    assert len(grid.buy_and_hold) == len(want)


def test_signal_uses_no_future_prices():
    p = _prices(1)
    q = p.copy()
    cut = 330
    q[cut + 1 :] *= np.random.default_rng(5).uniform(
        0.5, 2.0, len(p) - cut - 1
    )  # change the future
    a = cs.MovingAverageGrid(p).compute(10, 50)
    b = cs.MovingAverageGrid(q).compute(10, 50)
    # the return over day i depends on the position set at close i and on p[i + 1]: the first
    # i whose inputs are all at positions <= cut is i = cut - 1
    k = cut - 1 - (cs.WARMUP - 1)
    np.testing.assert_allclose(a[:k], b[:k], atol=1e-14)


def test_excess_return_is_zero_when_always_long_and_negative_buy_and_hold_when_flat():
    rising = 100 * 1.001 ** np.arange(400)  # fast average always above slow
    x = cs.MovingAverageGrid(rising).compute(2, 20)
    assert np.allclose(x, 0.0)
    falling = 100 * 0.999 ** np.arange(400)
    g = cs.MovingAverageGrid(falling)
    np.testing.assert_allclose(g.compute(2, 20), -g.buy_and_hold)  # flat: miss the (negative) move


def test_every_evaluation_is_logged_with_its_parameters(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    grid = cs.MovingAverageGrid(_prices(2), reg)
    for f, s in cs.GRID[:5]:
        grid.excess_return(fast=f, slow=s)
    assert reg.n_trials() == 5
    got = {(t.params["fast"], t.params["slow"]) for t in reg.trials().values()}
    assert got == set(cs.GRID[:5])


def test_synthetic_worlds_are_reproducible_and_the_trend_world_is_predictable():
    a = cs.synthetic_returns(3, "luck", 1500)
    assert np.array_equal(a, cs.synthetic_returns(3, "luck", 1500))
    assert not np.array_equal(a, cs.synthetic_returns(4, "luck", 1500))
    lag = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]  # noqa: E731
    luck = np.mean([lag(cs.synthetic_returns(s, "luck", 1500)) for s in range(30)])
    trend = np.mean([lag(cs.synthetic_returns(s, "trend", 1500)) for s in range(30)])
    # lag-1 autocorrelation of e + drift: rho * var(drift) / (var(e) + var(drift))
    theory = 0.97 * 0.0015**2 / (0.01**2 + 0.0015**2)
    assert abs(luck) < 0.015
    assert trend == pytest.approx(theory, abs=0.012)
    with pytest.raises(ValueError, match="kind"):
        cs.synthetic_returns(0, "bull")


def test_load_prices(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("date,close\n2020-01-03,102\n2020-01-01,100\n2020-01-02,101\n")
    prices, dates = cs.load_prices(p)
    assert prices.tolist() == [100.0, 101.0, 102.0] and dates[0] == "2020-01-01"
    (tmp_path / "bad.csv").write_text("day,px\n1,2\n")
    with pytest.raises(ValueError, match="needs 'date'"):
        cs.load_prices(tmp_path / "bad.csv")
