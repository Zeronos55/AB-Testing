import numpy as np
import pytest

from isbetter.methods.finance import (
    annualize_sharpe,
    deflated_sharpe,
    min_track_record,
    probabilistic_sharpe,
    psr,
    return_moments,
    sharpe_variance_factor,
)


def test_deflated_sharpe_refuses_to_run_without_the_number_of_trials():
    r = np.random.default_rng(0).normal(0.001, 0.01, 300)
    with pytest.raises(TypeError):
        deflated_sharpe(r)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        deflated_sharpe(r, n_trials=10)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="needs n_trials"):
        deflated_sharpe(r, n_trials=None, sr_variance=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        deflated_sharpe(r, n_trials=0, sr_variance=0.1)


def test_constant_or_tiny_returns_are_rejected():
    with pytest.raises(ValueError, match="constant"):
        return_moments(np.full(20, 0.01))
    with pytest.raises(ValueError, match="at least 3"):
        return_moments([0.1, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        return_moments([0.1, np.nan, 0.2, 0.3])


def test_excess_kurtosis_mistaken_for_raw_kurtosis_is_caught_for_large_sharpe():
    # kurtosis 0 (excess of a normal) with skew 2 and SR = 1 makes the variance factor negative
    with pytest.raises(ValueError, match="inconsistent moments"):
        sharpe_variance_factor(1.0, 2.0, 0.0)
    with pytest.raises(ValueError, match="at least 2"):
        psr(0.1, 0.0, 1)


def test_psr_is_monotone_in_sharpe_benchmark_and_length():
    assert psr(0.2, 0.0, 100) > psr(0.1, 0.0, 100)
    assert psr(0.2, 0.0, 100) > psr(0.2, 0.1, 100)
    assert psr(0.2, 0.0, 400) > psr(0.2, 0.0, 100)
    assert psr(0.0, 0.0, 100) == pytest.approx(0.5)
    assert psr(0.2, 0.0, 100, skew=-1.0) < psr(0.2, 0.0, 100, skew=0.0)  # negative skew hurts


def test_probabilistic_sharpe_is_invariant_to_return_scale():
    r = np.random.default_rng(1).normal(0.001, 0.01, 300)
    a, b = probabilistic_sharpe(r), probabilistic_sharpe(r * 100)
    assert a.estimate == pytest.approx(b.estimate)
    assert a.p_value == pytest.approx(b.p_value)


def test_probabilistic_sharpe_validates_alpha():
    with pytest.raises(ValueError, match="alpha"):
        probabilistic_sharpe([0.1, 0.2, 0.3, 0.1], alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        min_track_record([0.1, 0.2, 0.3, 0.1], alpha=0.0)


def test_annualize_sharpe():
    assert annualize_sharpe(0.1, 252) == pytest.approx(0.1 * 252**0.5)


def test_rationale_names_the_threshold_and_the_trial_count():
    r = np.random.default_rng(4).normal(0.001, 0.01, 300)
    res = deflated_sharpe(r, n_trials=300, sr_variance=0.001)
    assert "300 variants" in res.rationale and res.method == "deflated_sharpe"
