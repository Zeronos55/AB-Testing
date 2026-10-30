import numpy as np
import pytest

from isbetter.methods.finance import (
    return_moments,
    sharpe_block_bootstrap,
    spa_vs_buy_and_hold,
    stepm_vs_buy_and_hold,
)
from isbetter.simulate.dgp import ar1_garch_returns


def _r(seed, n=500, mu=0.0005, sd=0.01):
    return np.random.default_rng(seed).normal(mu, sd, n)


def test_interval_contains_estimate_and_reports_the_block_length():
    r = _r(1)
    res = sharpe_block_bootstrap(r, np.random.default_rng(0), n_resamples=500)
    assert res.ci_low < res.estimate < res.ci_high
    assert res.estimate == pytest.approx(return_moments(r).sharpe)
    assert res.diagnostics["block_len"] >= 1
    assert res.diagnostics["block_len_source"] in ("politis_white", "rule_of_thumb")
    assert res.method == "stationary_block_bootstrap_sharpe"


def test_user_block_length_is_respected_and_block_length_one_is_the_iid_bootstrap():
    r = _r(2)
    user = sharpe_block_bootstrap(r, np.random.default_rng(0), block_len=7, n_resamples=300)
    assert user.diagnostics["block_len"] == 7 and user.diagnostics["block_len_source"] == "user"
    iid = sharpe_block_bootstrap(r, np.random.default_rng(0), block_len=1, n_resamples=300)
    assert iid.method == "iid_bootstrap_sharpe"


def test_same_seed_gives_the_same_interval_and_circular_kind_works():
    r = _r(3)
    a = sharpe_block_bootstrap(r, np.random.default_rng(5), n_resamples=300)
    b = sharpe_block_bootstrap(r, np.random.default_rng(5), n_resamples=300)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    c = sharpe_block_bootstrap(r, np.random.default_rng(5), kind="circular", n_resamples=300)
    assert c.method == "circular_block_bootstrap_sharpe" and c.ci_low < c.ci_high


def test_batch_size_changes_the_random_stream_but_not_the_interval_beyond_mc_error():
    r = _r(4, 300)
    a = sharpe_block_bootstrap(r, np.random.default_rng(1), block_len=5, n_resamples=4000, batch=7)
    b = sharpe_block_bootstrap(
        r, np.random.default_rng(2), block_len=5, n_resamples=4000, batch=400
    )
    se = a.diagnostics["se"]
    assert abs(a.ci_low - b.ci_low) < 0.2 * se and abs(a.ci_high - b.ci_high) < 0.2 * se


def test_interval_for_a_difference_versus_a_benchmark():
    rng = np.random.default_rng(6)
    bench = rng.normal(0.0003, 0.01, 800)
    strat = 0.5 * bench + rng.normal(0.0008, 0.006, 800)
    res = sharpe_block_bootstrap(strat, np.random.default_rng(0), benchmark=bench, n_resamples=999)
    want = return_moments(strat).sharpe - return_moments(bench).sharpe
    assert res.estimate == pytest.approx(want)
    assert res.ci_low < want < res.ci_high
    assert res.diagnostics["versus_benchmark"] is True and "benchmark" in res.rationale
    # shared resampling indices: the difference is far tighter than two independent intervals
    solo = sharpe_block_bootstrap(strat, np.random.default_rng(0), n_resamples=999)
    assert res.diagnostics["se"] < 2 * solo.diagnostics["se"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "moving"}, "kind"),
        ({"alpha": 0.0}, "alpha"),
        ({"n_resamples": 10}, "n_resamples"),
        ({"block_len": 0}, "block_len"),
        ({"block_len": 10_000}, "block_len"),
        ({"benchmark": np.zeros(5)}, "same length"),
    ],
)
def test_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sharpe_block_bootstrap(_r(1, 100), np.random.default_rng(0), **kwargs)


def test_constant_returns_are_rejected():
    with pytest.raises(ValueError, match="constant"):
        sharpe_block_bootstrap(np.full(50, 0.001), np.random.default_rng(0))


def test_spa_and_stepm_find_a_planted_edge_over_buy_and_hold():
    rng = np.random.default_rng(10)
    bh = rng.normal(0.0003, 0.01, 1000)
    strategies = {f"s{j}": rng.normal(0.0003, 0.01, 1000) for j in range(20)}
    strategies["edge"] = bh + rng.normal(0.0020, 0.004, 1000)
    spa = spa_vs_buy_and_hold(strategies, bh, np.random.default_rng(0), reps=300)
    assert spa.p_value < 0.01
    assert spa.diagnostics["best_model"] == "edge"
    assert spa.estimate == pytest.approx(
        float(np.mean(strategies["edge"] - bh)), rel=1e-9
    )  # positive: a mean excess RETURN, not a loss
    sm = stepm_vs_buy_and_hold(strategies, bh, np.random.default_rng(0), reps=300)
    # StepM controls the family-wise error at 5%, so a lucky noise strategy may join the edge
    assert "edge" in sm.diagnostics["superior_models"]
    assert "loss = -return" in spa.rationale


def test_spa_does_not_reward_the_best_of_many_noise_strategies():
    rng = np.random.default_rng(11)
    bh = rng.normal(0.0003, 0.01, 1000)
    noise = {f"s{j}": rng.normal(0.0003, 0.01, 1000) for j in range(50)}
    best = max(float(np.mean(v - bh)) for v in noise.values())
    spa = spa_vs_buy_and_hold(noise, bh, np.random.default_rng(0), reps=300)
    assert best > 0  # the winner looks better than buy-and-hold ...
    assert spa.p_value > 0.05  # ... but the search-aware test is not fooled
    assert (
        stepm_vs_buy_and_hold(noise, bh, np.random.default_rng(0), reps=300).diagnostics[
            "superior_models"
        ]
        == []
    )


def test_ar1_garch_dgp_hits_its_target_moments():
    r = ar1_garch_returns(np.random.default_rng(0), 4000, 400, sharpe=0.08, vol=0.02, phi=0.3)
    assert r.std() == pytest.approx(0.02, rel=0.03)
    assert r.mean() / r.std() == pytest.approx(0.08, abs=0.01)
    lag1 = np.mean([np.corrcoef(x[1:], x[:-1])[0, 1] for x in r[:1000]])
    assert lag1 == pytest.approx(0.3, abs=0.03)
    sq = (r - r.mean(1, keepdims=True)) ** 2
    vol_cluster = np.mean([np.corrcoef(x[1:], x[:-1])[0, 1] for x in sq[:1000]])
    assert vol_cluster > 0.05  # volatility clustering is present
    with pytest.raises(ValueError, match="stationary"):
        ar1_garch_returns(np.random.default_rng(0), 2, 10, alpha=0.5, beta=0.6)
