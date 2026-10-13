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


def _paired_prep(
    a: np.ndarray, b: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-item differences ``a - b`` and integer cluster codes (items already aligned)."""
    a_, b_ = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a_.ndim != 1 or a_.shape != b_.shape or len(clusters) != len(a_):
        raise ValueError(
            "a, b and clusters must be 1-D arrays of equal length (one entry per item)"
        )
    d = a_ - b_
    if len(d) < 2:
        raise ValueError("need at least 2 items")
    if not np.isfinite(d).all():
        raise ValueError("scores contain NaN or infinite values")
    labels, codes = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    if len(labels) < 2:
        raise ValueError("need at least 2 clusters")
    return d, codes.ravel()


def paired_cluster_robust(
    scores_a: np.ndarray, scores_b: np.ndarray, clusters: np.ndarray, alpha: float = 0.05
) -> MethodResult:
    """Paired difference in means with a cluster-robust standard error (Miller 2024, sec. 3-4).

    Both models are scored on the same items, so the quantity of interest is the mean of the
    per-item differences ``d_i = a_i - b_i`` (pairing removes the item-difficulty variance),
    and items that share a cluster (passage, task, source) are correlated, so the standard
    error comes from cluster totals:

    ``SE^2 = G/(G-1) * sum_c (sum_{i in c} (d_i - dbar))^2 / n^2``

    which is the plain paired SE ``var(d)/n`` plus the within-cluster cross-products
    ``(1/n^2) sum_c sum_i sum_{j != i} e_i e_j`` (Miller's ``SE_clustered``), times the CR1
    factor ``G/(G-1)``. Inference uses t(G - 1). With every item its own cluster this is
    exactly the paired t-test. ``estimate`` is ``mean(a) - mean(b)``.

    Few clusters (roughly under 20-30) leave the interval liberal; the diagnostics report G.
    """
    d, codes = _paired_prep(scores_a, scores_b, clusters)
    n, G = len(d), int(codes.max()) + 1
    dbar = float(d.mean())
    totals = np.bincount(codes, weights=d - dbar, minlength=G)
    se = float(np.sqrt(G / (G - 1) * (totals**2).sum()) / n)
    se_iid = float(d.std(ddof=1) / np.sqrt(n))
    if se == 0:
        raise ValueError("zero cluster-robust variance: t statistic undefined")
    df = float(G - 1)
    t = dbar / se
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return MethodResult(
        method="paired_cluster_robust",
        estimate=dbar,
        ci_low=dbar - crit * se,
        ci_high=dbar + crit * se,
        ci_level=1 - alpha,
        p_value=float(2 * stats.t.sf(abs(t), df)),
        statistic=float(t),
        df=df,
        n_effective=float(G),
        rationale=(
            f"Paired difference over {n} items with a cluster-robust (CR1) standard error "
            f"({G} clusters, t({G - 1}) inference): both models answer the same items, so the "
            "per-item differences are analysed, and items in a cluster are correlated, so the "
            "error bar counts clusters rather than items."
        ),
        diagnostics={
            "se": se,
            "se_iid": se_iid,
            "design_effect": (se / se_iid) ** 2 if se_iid > 0 else float("nan"),
            "n_clusters": G,
            "n_items": n,
        },
    )


def paired_cluster_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 9_999,
    alpha: float = 0.05,
    batch: int = 200,
) -> MethodResult:
    """Percentile bootstrap of the paired difference that resamples whole clusters.

    Memory is O(batch * G): the differences are reduced to per-cluster totals first.
    """
    d, codes = _paired_prep(scores_a, scores_b, clusters)
    G = int(codes.max()) + 1
    sums = np.bincount(codes, weights=d, minlength=G)
    counts = np.bincount(codes, minlength=G).astype(np.float64)
    est = float(d.mean())
    out = np.empty(n_resamples)
    step = max(1, min(batch, 2_000_000 // G))
    for start in range(0, n_resamples, step):
        stop = min(start + step, n_resamples)
        idx = rng.integers(0, G, size=(stop - start, G))
        out[start:stop] = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return MethodResult(
        method="paired_cluster_bootstrap",
        estimate=est,
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=1 - alpha,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(G),
        rationale=(
            f"Percentile bootstrap of the paired difference resampling whole clusters ({G} "
            "clusters); a cross-check of the cluster-robust interval."
        ),
        diagnostics={"se": float(out.std(ddof=1)), "n_resamples": n_resamples},
    )
