import numpy as np
import pytest

from isbetter.methods.sequential import confidence_sequence_mean
from isbetter.simulate.calibration import peeking_type1

pytestmark = pytest.mark.slow


def test_confidence_sequence_keeps_false_positives_below_alpha_under_continuous_peeking(request):
    """A/A, read after EVERY observation from 20 to 2,000 per arm: the sequence's false-positive
    rate stays at or below 5% (it is conservative: about 2%), while the same fixed-horizon test
    stopped at the first p < 0.05 rejects in nearly half of the runs."""
    reps = min(request.config.getoption("--reps"), 2000)
    r = peeking_type1(np.random.default_rng(1), reps, n_max=2000, n_looks=None)
    assert r.sequence.rate <= r.alpha, f"sequence {r.sequence.rate:.4f}"
    assert r.sequence_known.rate <= r.alpha, f"known-variance sequence {r.sequence_known.rate:.4f}"
    assert r.peeked.rate > 0.30, f"peeked fixed-horizon test {r.peeked.rate:.3f}"
    lo, hi = r.single_look.band
    assert lo <= r.single_look.rate <= hi  # looked at once, the fixed test is fine


@pytest.mark.parametrize("n_looks", [5, 20])
def test_sequence_stays_valid_and_peeking_still_inflates_with_few_looks(request, n_looks):
    reps = min(request.config.getoption("--reps"), 2000)
    r = peeking_type1(np.random.default_rng(2), reps, n_max=1000, n_looks=n_looks)
    assert r.sequence.rate <= r.alpha
    assert r.peeked.rate > r.alpha + 0.05  # visible inflation even with a handful of looks


def test_plug_in_sequence_is_valid_on_skewed_data(request):
    """Exponential data (skew 2): continuous peeking from 20 to 1,000 observations."""
    reps = min(request.config.getoption("--reps"), 1000)
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(reps):
        x = rng.exponential(2.0, 1000)
        cs = confidence_sequence_mean(x, n_target=1000, null=2.0)
        hits += cs.first_exclusion() is not None
    assert hits / reps <= 0.05, f"false-positive rate {hits / reps:.4f}"
