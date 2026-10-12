import numpy as np
import pytest

from isbetter.simulate.calibration import sharpe_ci_coverage

pytestmark = pytest.mark.slow


def test_block_bootstrap_sharpe_interval_beats_iid_on_ar1_garch_returns(request):
    """AR(1) phi=0.2 + GARCH(1,1) t(5) returns, T=500, true per-period Sharpe 0.05.

    Known finite-sample behaviour, not a calibration claim: the percentile block bootstrap
    covers about 91-93% at nominal 95% (it is still slightly liberal), the iid bootstrap
    about 86-88%.
    """
    reps = min(request.config.getoption("--reps"), 1000)
    iid, blk = sharpe_ci_coverage(np.random.default_rng(41), reps=reps, T=500, phi=0.2)
    assert iid.rate < iid.band[0] - 0.02, f"iid coverage {iid.rate:.3f} not clearly deficient"
    assert blk.rate > iid.rate + 0.03, f"block {blk.rate:.3f} vs iid {iid.rate:.3f}"
    assert blk.rate > 0.89, f"block-bootstrap coverage {blk.rate:.3f} too low"


def test_both_bootstraps_cover_about_equally_without_serial_correlation(request):
    reps = min(request.config.getoption("--reps"), 600)
    iid, blk = sharpe_ci_coverage(np.random.default_rng(43), reps=reps, T=500, phi=0.0)
    assert abs(iid.rate - blk.rate) < 0.05
    assert blk.rate > 0.89
