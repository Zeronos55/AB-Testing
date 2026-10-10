import numpy as np
import pytest

from isbetter.methods.cv import nadeau_bengio, naive_kfold_t
from isbetter.simulate.calibration import nadeau_bengio_pvalues, naive_cv_pvalues
from isbetter.simulate.dgp import cv_null_diffs


def test_shapes_and_symmetry():
    d = cv_null_diffs(np.random.default_rng(0), reps=400, n=60, k=5, r=2)
    assert d.shape == (400, 10)
    assert abs(d.mean()) < 0.05  # symmetric learners: mean difference near zero


def test_vectorised_pvalues_match_scalar():
    d = cv_null_diffs(np.random.default_rng(1), reps=6, n=100, k=10, r=2)
    for i in range(6):
        assert naive_cv_pvalues(d)[i] == pytest.approx(naive_kfold_t(d[i]).p_value, abs=1e-10)
        assert nadeau_bengio_pvalues(d, 90, 10)[i] == pytest.approx(
            nadeau_bengio(d[i], 90, 10).p_value, abs=1e-10
        )
