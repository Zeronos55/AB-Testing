import numpy as np
import pytest

from isbetter.simulate.calibration import dm_type1

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("horizon", [1, 3])
def test_dm_hln_is_calibrated_and_naive_is_not_beyond_one_step(request, horizon):
    reps = min(request.config.getoption("--reps"), 5000)
    naive, dm = dm_type1(np.random.default_rng(31 + horizon), reps=reps, T=250, horizon=horizon)
    assert dm.passed, f"DM-HLN rate {dm.rate:.3f} outside band {dm.band}"
    if horizon == 1:
        assert naive.passed  # one-step errors are not autocorrelated: nothing to correct
    else:
        assert naive.rate > naive.band[1] + 0.05, f"naive rate {naive.rate:.3f} not inflated"


def test_dm_hln_at_horizon_7_is_much_better_than_naive_but_still_mildly_liberal(request):
    """Known finite-sample behaviour: about 5.5-6% at nominal 5% (T=250, h=7); HLN helps a lot."""
    reps = min(request.config.getoption("--reps"), 5000)
    naive, dm = dm_type1(np.random.default_rng(38), reps=reps, T=250, horizon=7)
    assert naive.rate > 0.3
    assert dm.rate < 0.07 and dm.rate < naive.rate / 4
