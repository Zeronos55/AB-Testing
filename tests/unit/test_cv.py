import numpy as np
import pytest

from isbetter.methods.cv import five_by_two_cv, nadeau_bengio, naive_kfold_t


def test_validation():
    with pytest.raises(ValueError, match="at least"):
        nadeau_bengio(np.array([0.1]), 9, 1)
    with pytest.raises(ValueError, match="NaN"):
        nadeau_bengio(np.array([0.1, np.nan]), 9, 1)
    with pytest.raises(ValueError, match="zero variance"):
        naive_kfold_t(np.full(5, 0.1))
    with pytest.raises(ValueError, match="shape"):
        five_by_two_cv(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        nadeau_bengio(np.array([0.1, 0.2]), 0, 1)


def test_naive_flagged():
    r = naive_kfold_t(np.array([0.1, 0.2, 0.15]))
    assert r.diagnostics["naive"] and "NAIVE" in r.rationale
