import numpy as np
import pytest

from isbetter.methods.parametric import paired_t, welch_t


def test_too_few():
    with pytest.raises(ValueError, match="at least 2"):
        welch_t(np.array([1.0]), np.array([1.0, 2.0]))


def test_nan_rejected():
    with pytest.raises(ValueError, match="NaN"):
        welch_t(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_zero_variance_explicit():
    with pytest.raises(ValueError, match="zero variance"):
        welch_t(np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="zero variance"):
        paired_t(np.arange(5.0), np.arange(5.0))


def test_paired_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        paired_t(np.arange(3.0), np.arange(4.0))
