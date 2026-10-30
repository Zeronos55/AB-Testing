import numpy as np
import pytest

from isbetter.simulate.calibration import mc_band, paired_pvalues, type1_rate, welch_pvalues
from isbetter.simulate.dgp import aa_independent, aa_paired

pytestmark = pytest.mark.slow


def test_welch_type1(request):
    reps = request.config.getoption("--reps")
    r = type1_rate(welch_pvalues, aa_independent, np.random.default_rng(11), reps=reps, n=40)
    assert r.passed, f"Welch rejection rate {r.rate:.4f} outside {r.band}"


def test_paired_type1(request):
    reps = request.config.getoption("--reps")
    r = type1_rate(paired_pvalues, aa_paired, np.random.default_rng(12), reps=reps, n=40)
    assert r.passed, f"paired-t rejection rate {r.rate:.4f} outside {r.band}"


def test_band_matches_spec():
    lo, hi = mc_band(0.05, 10_000)
    assert (round(lo, 4), round(hi, 4)) == (0.0456, 0.0544)
