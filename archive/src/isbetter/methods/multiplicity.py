"""Multiple-comparison adjustments returning adjusted p-values."""

from __future__ import annotations

import numpy as np


def _check(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("p-values must be a non-empty 1-D array")
    if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
        raise ValueError("p-values must be finite and in [0, 1]")
    return arr


def bonferroni(p: np.ndarray) -> np.ndarray:
    """Bonferroni: ``min(1, m * p)``."""
    arr = _check(p)
    return np.asarray(np.minimum(1.0, arr * len(arr)))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down: sorted ``(m - i + 1) * p_(i)`` made monotone with a running max."""
    arr = _check(p)
    m = len(arr)
    order = np.argsort(arr, kind="stable")
    adj = np.minimum(1.0, (m - np.arange(m)) * arr[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty(m)
    out[order] = adj
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR: sorted ``m * p_(i) / i`` made monotone with a running min."""
    arr = _check(p)
    m = len(arr)
    order = np.argsort(arr, kind="stable")
    adj = np.minimum(1.0, arr[order] * m / np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = adj
    return out


METHODS = {"bonferroni": bonferroni, "holm": holm, "bh": benjamini_hochberg}


def adjust(p: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return _check(p).copy()
    try:
        return METHODS[method](p)
    except KeyError:
        raise ValueError(f"unknown correction {method!r}; use none, {', '.join(METHODS)}") from None
