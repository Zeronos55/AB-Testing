"""SRM check: oracle against scipy.stats.chisquare plus the ticket's Hillstrom/planted cases."""

import numpy as np
import pytest
from scipy import stats

from isbetter.methods.srm import srm_check

HILLSTROM = {"No E-Mail": 21_306, "Mens E-Mail": 21_307, "Womens E-Mail": 21_387}


def test_matches_scipy_chisquare_on_random_tables():
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = int(rng.integers(2, 6))
        share = rng.dirichlet(np.ones(k) * 5)
        counts = rng.multinomial(int(rng.integers(500, 100_000)), share * 0.9 + 0.1 / k)
        arms = [f"a{i}" for i in range(k)]
        r = srm_check(
            dict(zip(arms, counts.tolist(), strict=True)), dict(zip(arms, share, strict=True))
        )
        ref = stats.chisquare(counts, counts.sum() * share)
        assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
        assert r.p_value == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-300)


def test_hillstrom_arm_counts_pass():
    r = srm_check(HILLSTROM)
    assert not r.flagged and r.p_value > 0.5
    assert r.df == 2


def test_planted_50_5_split_at_one_million_fails():
    r = srm_check({"control": 505_000, "treatment": 495_000}, {"control": 0.5, "treatment": 0.5})
    assert r.flagged
    assert r.statistic == pytest.approx(100.0)
    assert r.p_value < 1e-20
    assert "SRM DETECTED" in r.rationale


def test_same_skew_is_not_flagged_at_small_n():
    r = srm_check({"control": 505, "treatment": 495})
    assert not r.flagged  # 1% imbalance on 1,000 users is ordinary noise


def test_unequal_planned_split_is_respected():
    ok = srm_check({"0": 150_200, "1": 849_800}, {"0": 0.15, "1": 0.85})
    assert not ok.flagged
    bad = srm_check({"0": 150_200, "1": 849_800})  # judged against the wrong plan
    assert bad.flagged


def test_small_expected_counts_are_marked():
    assert not srm_check({"a": 3, "b": 4}).approx_ok
    assert srm_check({"a": 30, "b": 40}).approx_ok


@pytest.mark.parametrize(
    "counts, share, msg",
    [
        ({"a": 5}, None, "at least 2"),
        ({"a": -1, "b": 5}, None, "non-negative"),
        ({"a": 0, "b": 0}, None, "no units"),
        ({"a": 5, "b": 5}, {"a": 0.5}, "do not match"),
        ({"a": 5, "b": 5}, {"a": 0.6, "b": 0.6}, "sum to 1"),
        ({"a": 5, "b": 5}, {"a": 1.0, "b": 0.0}, "positive"),
    ],
)
def test_invalid_inputs_raise(counts, share, msg):
    with pytest.raises(ValueError, match=msg):
        srm_check(counts, share)
