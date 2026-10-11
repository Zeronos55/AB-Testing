"""Parametric tests: Welch t, paired t, and standardised effect sizes.

All functions report ``estimate = mean(a) - mean(b)`` with a two-sided CI.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult
from isbetter.core.sufficient import SufficientStats


def _clean(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if len(arr) < 2:
        raise ValueError(f"{name} needs at least 2 observations")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with the pooled standard deviation."""
    a, b = _clean(a, "a"), _clean(b, "b")
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 == 0:
        raise ValueError("zero pooled variance: effect size undefined")
    return float((a.mean() - b.mean()) / np.sqrt(sp2))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g: Cohen's d with the small-sample correction J = 1 - 3/(4*df - 1)."""
    df = len(a) + len(b) - 2
    return float(cohens_d(a, b) * (1 - 3 / (4 * df - 1)))


def _welch_from_moments(
    na: int, ma: float, va_raw: float, nb: int, mb: float, vb_raw: float, alpha: float
) -> MethodResult:
    """Welch's t from per-arm n, mean and sample variance (shared by the array and sums paths)."""
    va, vb = va_raw / na, vb_raw / nb
    if va + vb == 0:
        raise ValueError("zero variance in both samples: t statistic undefined")
    diff = float(ma - mb)
    se = float(np.sqrt(va + vb))
    df = float((va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1)))
    t = diff / se
    p = float(2 * stats.t.sf(abs(t), df))
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    sp2 = ((na - 1) * va_raw + (nb - 1) * vb_raw) / (na + nb - 2)
    d = diff / float(np.sqrt(sp2)) if sp2 > 0 else float("nan")
    g = d * (1 - 3 / (4 * (na + nb - 2) - 1))
    return MethodResult(
        method="welch_t",
        estimate=diff,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        ci_level=1 - alpha,
        p_value=p,
        statistic=float(t),
        df=df,
        n_effective=float(na + nb),
        rationale=(
            "Welch t-test: independent units (dependence=iid); does not assume equal variances."
        ),
        diagnostics={"se": se, "cohens_d": d, "hedges_g": g},
    )


def welch_t(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> MethodResult:
    """Welch's unequal-variance t-test for independent samples (Welch, 1947)."""
    a, b = _clean(a, "a"), _clean(b, "b")
    return _welch_from_moments(
        len(a), float(a.mean()), float(a.var(ddof=1)), len(b), float(b.mean()),
        float(b.var(ddof=1)), alpha,
    )  # fmt: skip


def welch_t_stats(a: SufficientStats, b: SufficientStats, alpha: float = 0.05) -> MethodResult:
    """:func:`welch_t` from the aggregated statistics of the two arms."""
    for name, s in (("a", a), ("b", b)):
        if s.n < 2:
            raise ValueError(f"{name} needs at least 2 observations")
    return _welch_from_moments(a.n, a.mean, a.var, b.n, b.mean, b.var, alpha)


def paired_t(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> MethodResult:
    """Paired t-test on within-unit differences a_i - b_i."""
    a, b = _clean(a, "a"), _clean(b, "b")
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        raise ValueError("zero variance in paired differences: t statistic undefined")
    se = float(sd / np.sqrt(n))
    mean = float(d.mean())
    t = mean / se
    df = float(n - 1)
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return MethodResult(
        method="paired_t",
        estimate=mean,
        ci_low=mean - crit * se,
        ci_high=mean + crit * se,
        ci_level=1 - alpha,
        p_value=float(2 * stats.t.sf(abs(t), df)),
        statistic=float(t),
        df=df,
        n_effective=float(n),
        rationale=(
            "Paired t-test: both variants measured on the same units (dependence=paired); "
            "pairing removes shared unit-level noise."
        ),
        diagnostics={"se": se, "cohens_dz": float(mean / sd)},
    )
