"""Memory-safe resampling: paired bootstrap, block bootstrap and sign-flip permutation.

Both operate on the vector of paired differences ``d`` (never on both arms), and
process resamples in blocks so memory stays O(block * n), never O(B * n).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult

# Upper bound on elements held in one resampling block (~16 MB of int64 indices).
_MAX_BLOCK_ELEMENTS = 2_000_000
_BCA_MAX_N = 50_000
_EXACT_MAX_N = 16


def _diffs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired samples must be 1-D arrays of equal length")
    d: np.ndarray = a - b
    if len(d) < 2:
        raise ValueError("need at least 2 pairs")
    if not np.isfinite(d).all():
        raise ValueError("paired samples contain NaN or infinite values")
    return d


def _block(batch: int, n: int) -> int:
    if batch < 1:
        raise ValueError("batch must be >= 1")
    return max(1, min(batch, _MAX_BLOCK_ELEMENTS // n))


def bootstrap_means(
    d: np.ndarray, n_resamples: int, rng: np.random.Generator, batch: int = 200
) -> np.ndarray:
    """Means of ``n_resamples`` bootstrap resamples of ``d``, computed in blocks."""
    n = len(d)
    out = np.empty(n_resamples)
    step = _block(batch, n)
    for start in range(0, n_resamples, step):
        stop = min(start + step, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = d[idx].mean(axis=1)
    return out


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 9_999,
    alpha: float = 0.05,
    method: str = "percentile",
    batch: int = 200,
) -> MethodResult:
    """Bootstrap CI for the mean paired difference (``method``: percentile or bca).

    BCa delegates to ``scipy.stats.bootstrap`` and is limited to moderate n because its
    jackknife step is O(n^2); use ``percentile`` for very large n.
    """
    d = _diffs(a, b)
    n = len(d)
    est = float(d.mean())
    if method == "percentile":
        means = bootstrap_means(d, n_resamples, rng, batch)
        lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        low, high = float(lo), float(hi)
        se = float(means.std(ddof=1))
    elif method == "bca":
        if n > _BCA_MAX_N:
            raise ValueError(f"BCa limited to n <= {_BCA_MAX_N}; use method='percentile'")
        res = stats.bootstrap(
            (d,),
            np.mean,
            n_resamples=n_resamples,
            confidence_level=1 - alpha,
            method="BCa",
            vectorized=True,
            batch=_block(batch, n),
            random_state=rng,
        )
        low, high = float(res.confidence_interval.low), float(res.confidence_interval.high)
        se = float(res.standard_error)
    else:
        raise ValueError("method must be 'percentile' or 'bca'")
    return MethodResult(
        method=f"paired_bootstrap_{method}",
        estimate=est,
        ci_low=low,
        ci_high=high,
        ci_level=1 - alpha,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(n),
        rationale=(
            f"Paired {method} bootstrap of the mean difference "
            f"({n_resamples} resamples, batched); makes no normality assumption."
        ),
        diagnostics={"se": se, "n_resamples": n_resamples},
    )


def sign_flip_permutation(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 9_999,
    batch: int = 200,
) -> MethodResult:
    """Two-sided sign-flip permutation test of the mean paired difference.

    Exact (all 2^n sign patterns) for n <= 16, otherwise Monte Carlo with the
    ``(count + 1) / (B + 1)`` correction.
    """
    d = _diffs(a, b)
    n = len(d)
    obs = abs(float(d.mean()))
    tol = 1e-12 * max(1.0, float(np.abs(d).max()))
    if n <= _EXACT_MAX_N:
        codes = np.arange(2**n, dtype=np.int64)
        signs = 1.0 - 2.0 * ((codes[:, None] >> np.arange(n)) & 1)
        perm = np.abs((signs * d).mean(axis=1))
        p = float((perm >= obs - tol).mean())
        exact = True
    else:
        count = 0
        step = _block(batch, n)
        for start in range(0, n_resamples, step):
            m = min(step, n_resamples - start)
            signs = 1.0 - 2.0 * rng.integers(0, 2, size=(m, n))
            count += int((np.abs((signs * d).mean(axis=1)) >= obs - tol).sum())
        p = (count + 1) / (n_resamples + 1)
        exact = False
    how = "exact enumeration" if exact else f"{n_resamples} Monte Carlo draws"
    return MethodResult(
        method="sign_flip_permutation",
        estimate=float(d.mean()),
        ci_low=None,
        ci_high=None,
        ci_level=None,
        p_value=p,
        statistic=float(d.mean()),
        df=None,
        n_effective=float(n),
        rationale=(
            "Sign-flip permutation test: under H0 the sign of each paired difference is "
            f"exchangeable ({how})."
        ),
        diagnostics={"exact": exact},
    )


def _block_indices(
    kind: str, n: int, block_len: int, rows: int, rng: np.random.Generator
) -> np.ndarray:
    """``rows`` resampled index vectors of length ``n`` for a block bootstrap."""
    if kind == "circular":
        n_blocks = -(-n // block_len)  # ceil
        starts = rng.integers(0, n, size=(rows, n_blocks))
        idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
        return idx.reshape(rows, n_blocks * block_len)[:, :n]
    # Stationary (Politis & Romano, 1994): a new block starts with probability 1/block_len,
    # otherwise the series continues, so block lengths are geometric with mean block_len.
    fresh = rng.random((rows, n)) < 1.0 / block_len
    fresh[:, 0] = True
    starts = rng.integers(0, n, size=(rows, n))
    pos = np.arange(n)[None, :]
    last = np.maximum.accumulate(np.where(fresh, pos, 0), axis=1)
    return (np.take_along_axis(starts, last, axis=1) + (pos - last)) % n


def block_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    block_len: int,
    kind: str = "stationary",
    n_resamples: int = 9_999,
    alpha: float = 0.05,
    batch: int = 200,
) -> MethodResult:
    """Percentile block-bootstrap CI for the mean of a serially dependent difference ``a - b``.

    Resampling whole blocks keeps the autocorrelation of the loss differential inside each
    block, so the interval is valid for time series where the iid bootstrap is not.
    ``kind`` is ``"stationary"`` (random geometric block lengths with mean ``block_len``,
    Politis & Romano 1994) or ``"circular"`` (fixed length, wrapping around the end,
    Politis & Romano 1992). Memory is O(batch * n).
    """
    if kind not in ("stationary", "circular"):
        raise ValueError("kind must be 'stationary' or 'circular'")
    d = _diffs(a, b)
    n = len(d)
    if not 1 <= block_len <= n:
        raise ValueError(f"block_len must be between 1 and the number of origins ({n})")
    means = np.empty(n_resamples)
    step = _block(batch, n)
    for start in range(0, n_resamples, step):
        stop = min(start + step, n_resamples)
        means[start:stop] = d[_block_indices(kind, n, block_len, stop - start, rng)].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return MethodResult(
        method=f"{kind}_block_bootstrap",
        estimate=float(d.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=1 - alpha,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(n),
        rationale=(
            f"{kind.capitalize()} block bootstrap of the mean loss differential "
            f"(mean block length {block_len}, {n_resamples} resamples): resamples blocks so "
            "serial dependence is preserved; makes no normality assumption."
        ),
        diagnostics={
            "se": float(means.std(ddof=1)),
            "block_len": block_len,
            "n_resamples": n_resamples,
        },
    )
