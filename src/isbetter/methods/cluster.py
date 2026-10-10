"""Cluster-aware inference for a difference in means.

Rows sharing a cluster id (users, passages, ...) are correlated, so the effective sample
size is the number of clusters, not rows. Two estimators are provided: a cluster-robust
(CR1) regression standard error and a cluster bootstrap.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult


def _prep(
    va: np.ndarray, ca: np.ndarray, vb: np.ndarray, cb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.concatenate([np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64)])
    if len(va) != len(ca) or len(vb) != len(cb):
        raise ValueError("values and cluster ids must have equal length per arm")
    if len(va) < 1 or len(vb) < 1:
        raise ValueError("each arm needs at least one observation")
    if not np.isfinite(y).all():
        raise ValueError("values contain NaN or infinite values")
    arm = np.r_[np.ones(len(va)), np.zeros(len(vb))]
    _, codes = np.unique(
        np.concatenate([np.asarray(ca).astype(str), np.asarray(cb).astype(str)]),
        return_inverse=True,
    )
    if len(np.unique(codes)) < 2:
        raise ValueError("need at least 2 clusters")
    return y, arm, codes


def cluster_robust_diff(
    values_a: np.ndarray,
    clusters_a: np.ndarray,
    values_b: np.ndarray,
    clusters_b: np.ndarray,
    alpha: float = 0.05,
) -> MethodResult:
    """OLS of value on an arm dummy with CR1 cluster-robust SE and t(G - 1) inference.

    ``V = G/(G-1) * (N-1)/(N-k) * (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1`` with k = 2.
    The estimate is ``mean(a) - mean(b)``. Cluster labels are shared across arms, so the
    same label in both arms is one cluster.
    """
    y, arm, codes = _prep(values_a, clusters_a, values_b, clusters_b)
    N = len(y)
    X = np.column_stack([np.ones(N), arm])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    G = len(np.unique(codes))
    scores = np.zeros((G, 2))
    np.add.at(scores, codes, X * u[:, None])
    meat = scores.T @ scores
    k = 2
    scale = G / (G - 1) * (N - 1) / (N - k)
    V = scale * XtX_inv @ meat @ XtX_inv
    est = float(beta[1])
    se = float(np.sqrt(V[1, 1]))
    if se == 0:
        raise ValueError("zero cluster-robust variance: t statistic undefined")
    df = float(G - 1)
    t = est / se
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return MethodResult(
        method="cluster_robust_diff",
        estimate=est,
        ci_low=est - crit * se,
        ci_high=est + crit * se,
        ci_level=1 - alpha,
        p_value=float(2 * stats.t.sf(abs(t), df)),
        statistic=float(t),
        df=df,
        n_effective=float(G),
        rationale=(
            f"Cluster-robust (CR1) difference in means: dependence=clustered, {G} clusters. "
            "Rows in a cluster are correlated, so inference uses G - 1 degrees of freedom "
            "rather than the row count."
        ),
        diagnostics={"se": se, "n_clusters": G, "n_rows": N},
    )


def cluster_bootstrap(
    values_a: np.ndarray,
    clusters_a: np.ndarray,
    values_b: np.ndarray,
    clusters_b: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 9_999,
    alpha: float = 0.05,
    batch: int = 200,
) -> MethodResult:
    """Percentile bootstrap that resamples whole clusters (memory O(batch * G))."""
    y, arm, codes = _prep(values_a, clusters_a, values_b, clusters_b)
    G = len(np.unique(codes))
    sum_a = np.bincount(codes, weights=y * arm, minlength=G)
    n_a = np.bincount(codes, weights=arm, minlength=G)
    sum_b = np.bincount(codes, weights=y * (1 - arm), minlength=G)
    n_b = np.bincount(codes, weights=1 - arm, minlength=G)
    est = float(sum_a.sum() / n_a.sum() - sum_b.sum() / n_b.sum())
    stats_ = np.empty(n_resamples)
    step = max(1, min(batch, 2_000_000 // G))
    for start in range(0, n_resamples, step):
        stop = min(start + step, n_resamples)
        idx = rng.integers(0, G, size=(stop - start, G))
        with np.errstate(invalid="ignore", divide="ignore"):
            stats_[start:stop] = sum_a[idx].sum(1) / n_a[idx].sum(1) - sum_b[idx].sum(1) / n_b[
                idx
            ].sum(1)
    ok = stats_[np.isfinite(stats_)]
    if len(ok) < 0.9 * n_resamples:
        raise ValueError("too many bootstrap resamples lost an arm entirely; need more clusters")
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    return MethodResult(
        method="cluster_bootstrap",
        estimate=est,
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=1 - alpha,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(G),
        rationale=f"Percentile bootstrap resampling whole clusters ({G} clusters).",
        diagnostics={"se": float(ok.std(ddof=1)), "n_resamples": n_resamples},
    )
