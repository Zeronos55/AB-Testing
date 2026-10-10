import numpy as np
import pytest
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests

from isbetter.methods.classification import mcnemar
from isbetter.methods.multiplicity import adjust


def _table(a, b):
    return np.array([[(a & b).sum(), (a & ~b).sum()], [(~a & b).sum(), (~a & ~b).sum()]])


@pytest.mark.parametrize("n,pa,pb", [(40, 0.8, 0.6), (300, 0.9, 0.88), (1000, 0.7, 0.72)])
def test_mcnemar_matches_statsmodels(n, pa, pb):
    rng = np.random.default_rng(n)
    a, b = rng.random(n) < pa, rng.random(n) < pb
    t = _table(a, b)
    r_exact = mcnemar(a, b, exact=True)
    r_chi = mcnemar(a, b, exact=False)
    assert r_exact.p_value == pytest.approx(sm_mcnemar(t, exact=True).pvalue, abs=1e-12)
    ref = sm_mcnemar(t, exact=False, correction=True)
    assert r_chi.p_value == pytest.approx(ref.pvalue, abs=1e-12)
    assert r_chi.statistic == pytest.approx(ref.statistic, abs=1e-10)


def test_mcnemar_auto_switch_and_no_discordance():
    a = np.array([True] * 10)
    assert mcnemar(a, a).p_value == 1.0
    rng = np.random.default_rng(0)
    big_a, big_b = rng.random(500) < 0.8, rng.random(500) < 0.7
    assert mcnemar(big_a, big_b).method == "mcnemar_chi2"
    assert mcnemar(big_a[:20], big_b[:20]).method == "mcnemar_exact"


@pytest.mark.parametrize(
    "ours,theirs", [("bonferroni", "bonferroni"), ("holm", "holm"), ("bh", "fdr_bh")]
)
def test_multiplicity_matches_statsmodels(ours, theirs):
    rng = np.random.default_rng(42)
    # statsmodels' holm is ~50 ms/call, so use fewer vectors for it
    for _ in range(150 if theirs == "holm" else 1000):
        m = int(rng.integers(1, 30))
        p = rng.random(m) ** rng.uniform(1, 4)
        ref = multipletests(p, method=theirs)[1]
        assert np.allclose(adjust(p, ours), ref, atol=1e-12)
