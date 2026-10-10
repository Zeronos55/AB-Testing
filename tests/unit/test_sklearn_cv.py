import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import make_classification  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from isbetter.adapters.sklearn_cv import run_cv  # noqa: E402
from isbetter.core.dependence import Paired  # noqa: E402
from isbetter.core.schema import build_table  # noqa: E402

X, y = make_classification(600, 8, weights=[0.9], random_state=0)
EST = {
    "logreg": lambda: LogisticRegression(max_iter=200),
    "tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
}


def test_table_valid_and_shape():
    out = run_cv(X, y, EST, n_splits=5, n_repeats=2, seed=1)
    assert len(out.table) == 2 * 5 * 2
    build_table(out.table.columns, Paired())  # passes paired validation
    assert out.oof_proba["logreg"].shape == (2, 600)
    assert not np.isnan(out.oof_proba["tree"]).any()
    assert (out.n_train, out.n_test) == (480, 120)
    assert out.table.columns["value"].min() > 0


def test_deterministic_with_seed():
    a = run_cv(X, y, EST, n_splits=5, n_repeats=2, seed=3)
    b = run_cv(X, y, EST, n_splits=5, n_repeats=2, seed=3)
    assert np.array_equal(a.table.columns["value"], b.table.columns["value"])


def test_rejects_accuracy_and_single_model():
    with pytest.raises(ValueError, match="pr_auc"):
        run_cv(X, y, EST, metric="accuracy")
    with pytest.raises(ValueError, match="two"):
        run_cv(X, y, {"a": EST["logreg"]})
