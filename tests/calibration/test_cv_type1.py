import numpy as np
import pytest

from isbetter.simulate.calibration import cv_type1

pytestmark = pytest.mark.slow


def test_naive_inflated_and_nadeau_bengio_valid(request):
    reps = min(request.config.getoption("--reps"), 3000)
    naive, nb = cv_type1(np.random.default_rng(21), reps=reps, n=100, k=10, r=10)
    assert naive.rate > naive.band[1], f"naive rate {naive.rate:.3f} not inflated"
    assert nb.rate <= nb.band[1], f"NB rate {nb.rate:.3f} above band {nb.band}"
