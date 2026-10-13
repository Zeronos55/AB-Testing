import numpy as np
import pytest

from isbetter.core.result import MethodResult
from isbetter.methods.sequential import (
    confidence_sequence_diff,
    confidence_sequence_mean,
    optimal_rho,
    sequential_mean_result,
)
from isbetter.simulate.calibration import look_schedule, peeking_type1


def test_must_choose_rho_before_looking_at_the_data():
    x = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError, match="exactly one of rho or n_target"):
        confidence_sequence_mean(x)
    with pytest.raises(ValueError, match="exactly one of rho or n_target"):
        confidence_sequence_mean(x, rho=10.0, n_target=100)
    with pytest.raises(ValueError, match="exactly one"):
        confidence_sequence_diff(x, x)


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"rho": -1.0}, "rho must be positive"),
        ({"rho": 1.0, "alpha": 1.5}, "alpha"),
        ({"rho": 1.0, "sigma": 0.0}, "sigma must be positive"),
        ({"rho": 1.0, "min_n": 500}, "at least 500"),
    ],
)
def test_bad_arguments(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        confidence_sequence_mean(np.random.default_rng(0).normal(size=60), **kwargs)


def test_bad_streams():
    with pytest.raises(ValueError, match="NaN"):
        confidence_sequence_mean(np.array([1.0, np.nan] * 20), rho=5.0)
    with pytest.raises(ValueError, match="1-D"):
        confidence_sequence_mean(np.ones((4, 4)), rho=5.0)
    with pytest.raises(ValueError, match="truncate"):
        confidence_sequence_diff(np.zeros(30), np.zeros(31), rho=5.0)
    with pytest.raises(ValueError, match="per arm"):
        confidence_sequence_diff(np.zeros(5), np.zeros(5), rho=5.0)
    with pytest.raises(ValueError, match="n_target"):
        optimal_rho(0)


def test_plug_in_needs_two_observations_even_with_min_n_one():
    with pytest.raises(ValueError, match="at least 2"):
        confidence_sequence_mean(np.array([1.0]), rho=5.0, min_n=1)
    cs = confidence_sequence_mean(np.array([1.0]), rho=5.0, sigma=0.5, min_n=1)
    assert len(cs.n) == 1 and np.isfinite(cs.radius).all()


def test_zero_variance_stream_gives_an_unbounded_interval_not_false_certainty():
    cs = confidence_sequence_mean(np.ones(50), rho=10.0, min_n=2)
    assert np.isinf(cs.radius).all() and (cs.p_value == 1.0).all()
    assert cs.first_exclusion() is None


def test_shifted_stream_excludes_zero_and_reports_when():
    x = np.random.default_rng(1).normal(0.5, 1.0, 1000)
    cs = confidence_sequence_mean(x, n_target=1000, min_n=20)
    first = cs.first_exclusion()
    assert first is not None and first < 400
    assert cs.p_value[-1] < 1e-4 and cs.lower[-1] > 0
    assert cs.first_exclusion(value=0.5) is None  # the true mean is never excluded here


def test_method_result_final_look_and_foil():
    x = np.random.default_rng(2).normal(0.3, 1.0, 500)
    r = sequential_mean_result(x, n_target=500)
    assert isinstance(r, MethodResult) and r.method == "confidence_sequence_normal_mixture"
    assert r.ci_low < r.estimate < r.ci_high and r.n_effective == 500
    fixed_lo, fixed_hi = r.diagnostics["fixed_horizon_ci"]
    assert fixed_hi - fixed_lo < r.ci_high - r.ci_low  # the price of being allowed to peek
    assert r.diagnostics["variance"].startswith("plug-in")
    known = sequential_mean_result(x, n_target=500, sigma=1.0)
    assert known.diagnostics["variance"] == "known"


def test_look_schedule():
    assert list(look_schedule(10, None, 4)) == [4, 5, 6, 7, 8, 9, 10]
    sched = look_schedule(100, 5, 20)
    assert sched[0] == 20 and sched[-1] == 100 and len(sched) == 5
    with pytest.raises(ValueError):
        look_schedule(30, 50, 20)


def test_vectorised_peeking_sequence_matches_the_method_row_by_row():
    """One simulated A/A run: the helper's rejection of the sequence equals the method's."""
    rng = np.random.default_rng(7)
    res = peeking_type1(rng, 40, n_max=300, n_looks=None, min_n=20, n_target=300, chunk=40)
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(40, 300)), rng.normal(size=(40, 300))
    expected = sum(
        confidence_sequence_diff(a[i], b[i], n_target=300, min_n=20).first_exclusion() is not None
        for i in range(40)
    )
    assert res.sequence.rate == pytest.approx(expected / 40)
