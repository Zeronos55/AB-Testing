import numpy as np
import pytest
from mlxtend.evaluate import paired_ttest_5x2cv
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from isbetter.methods.cv import cv_variance_factor, five_by_two_cv, nadeau_bengio, naive_kfold_t


def test_nadeau_bengio_hand_computed():
    # d = [0.02, 0.04, 0.00, 0.06]; mean=0.03; var(ddof=1)=0.0008333...
    d = np.array([0.02, 0.04, 0.00, 0.06])
    r = nadeau_bengio(d, n_train=90, n_test=10)
    factor = 1 / 4 + 10 / 90
    se = np.sqrt(factor * d.var(ddof=1))
    assert r.statistic == pytest.approx(0.03 / se, abs=1e-12)
    assert r.df == 3
    assert r.p_value == pytest.approx(2 * stats.t.sf(0.03 / se, 3), abs=1e-12)
    crit = stats.t.ppf(0.975, 3)
    assert r.ci_low == pytest.approx(0.03 - crit * se, abs=1e-12)


def test_naive_equals_scipy_ttest_1samp():
    d = np.random.default_rng(0).normal(0.01, 0.02, 25)
    r = naive_kfold_t(d)
    ref = stats.ttest_1samp(d, 0.0)
    assert r.statistic == pytest.approx(ref.statistic, abs=1e-10)
    assert r.p_value == pytest.approx(ref.pvalue, abs=1e-10)


def test_nb_wider_than_naive():
    d = np.random.default_rng(1).normal(0.01, 0.02, 100)
    nb, nv = nadeau_bengio(d, 900, 100), naive_kfold_t(d)
    assert nb.ci_high - nb.ci_low > nv.ci_high - nv.ci_low
    assert nb.p_value > nv.p_value


def test_variance_factor_example_from_blueprint():
    factor, share = cv_variance_factor(100, 90, 10)  # 10x10 CV
    assert factor == pytest.approx(0.01 + 1 / 9)
    assert share > 0.9


def test_5x2cv_matches_mlxtend_on_iris():
    X, y = load_iris(return_X_y=True)
    clf1 = LogisticRegression(max_iter=500)
    clf2 = DecisionTreeClassifier(random_state=1)
    t_ref, p_ref = paired_ttest_5x2cv(clf1, clf2, X, y, random_seed=1)

    rng = np.random.RandomState(1)
    diffs = np.empty((5, 2))
    for i in range(5):
        seed = rng.randint(low=0, high=32767)
        X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=seed)
        for j, (a, b, c, e) in enumerate([(X1, y1, X2, y2), (X2, y2, X1, y1)]):
            clf1.fit(a, b)
            clf2.fit(a, b)
            diffs[i, j] = clf1.score(c, e) - clf2.score(c, e)
    ours = five_by_two_cv(diffs)
    assert ours.statistic == pytest.approx(t_ref, abs=1e-10)
    assert ours.p_value == pytest.approx(p_ref, abs=1e-10)
