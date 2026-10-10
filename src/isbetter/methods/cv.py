"""Tests for comparing models from cross-validation scores.

K-fold training sets overlap, so per-fold scores are positively correlated and a plain
t-test on them is anti-conservative. Nadeau & Bengio (2003) correct the variance.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult


def _diffs(d: np.ndarray, minimum: int = 2) -> np.ndarray:
    arr = np.asarray(d, dtype=np.float64).ravel()
    if len(arr) < minimum:
        raise ValueError(f"need at least {minimum} score differences")
    if not np.isfinite(arr).all():
        raise ValueError("score differences contain NaN or infinite values")
    return arr


def cv_variance_factor(n_scores: int, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau-Bengio factor ``1/J + n_test/n_train`` and the share due to the second term.

    The second term does not shrink with more repeats: when it dominates, more repeats
    cannot help and more data is needed.
    """
    if n_scores < 1 or n_train < 1 or n_test < 1:
        raise ValueError("n_scores, n_train and n_test must be >= 1")
    first, second = 1.0 / n_scores, n_test / n_train
    return first + second, second / (first + second)


def nadeau_bengio(
    diffs: np.ndarray, n_train: int, n_test: int, alpha: float = 0.05
) -> MethodResult:
    """Corrected resampled t-test (Nadeau & Bengio, 2003, Machine Learning 52).

    With J = r*k per-fold differences d_j:
    ``t = mean(d) / sqrt((1/J + n_test/n_train) * s^2_d)``, df = J - 1.
    Valid but conservative (Bengio & Grandvalet 2004: no unbiased estimator exists).
    """
    d = _diffs(diffs)
    J = len(d)
    s2 = float(d.var(ddof=1))
    if s2 == 0:
        raise ValueError("zero variance in score differences: t statistic undefined")
    factor, test_share = cv_variance_factor(J, n_train, n_test)
    se = float(np.sqrt(factor * s2))
    mean = float(d.mean())
    df = float(J - 1)
    t = mean / se
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return MethodResult(
        method="nadeau_bengio",
        estimate=mean,
        ci_low=mean - crit * se,
        ci_high=mean + crit * se,
        ci_level=1 - alpha,
        p_value=float(2 * stats.t.sf(abs(t), df)),
        statistic=float(t),
        df=df,
        n_effective=float(J),
        rationale=(
            "Nadeau-Bengio corrected resampled t-test: dependence=cv_overlap. Training sets "
            "overlap, so per-fold scores are correlated; the variance is inflated by "
            "(1/J + n_test/n_train) instead of 1/J. Valid but conservative."
        ),
        diagnostics={
            "se": se,
            "variance_factor": factor,
            "test_term_share": test_share,
            "n_train": n_train,
            "n_test": n_test,
        },
    )


def naive_kfold_t(diffs: np.ndarray, alpha: float = 0.05) -> MethodResult:
    """NAIVE paired t-test on per-fold differences. For the failure gallery only.

    Ignores training-set overlap (variance factor 1/J), so its false-positive rate
    exceeds the nominal level. Do not use it for real decisions.
    """
    d = _diffs(diffs)
    J = len(d)
    sd = float(d.std(ddof=1))
    if sd == 0:
        raise ValueError("zero variance in score differences: t statistic undefined")
    se = sd / np.sqrt(J)
    mean = float(d.mean())
    df = float(J - 1)
    t = mean / se
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return MethodResult(
        method="naive_kfold_t",
        estimate=mean,
        ci_low=mean - crit * se,
        ci_high=mean + crit * se,
        ci_level=1 - alpha,
        p_value=float(2 * stats.t.sf(abs(t), df)),
        statistic=float(t),
        df=df,
        n_effective=float(J),
        rationale=(
            "NAIVE paired t-test on per-fold scores: ignores training-set overlap, so it is "
            "too confident. Kept only to demonstrate the failure."
        ),
        diagnostics={"se": float(se), "naive": True},
    )


def five_by_two_cv(diffs: np.ndarray) -> MethodResult:
    """Dietterich (1998) 5x2cv paired t-test.

    ``diffs`` has shape (5, 2): score differences for each of 5 replications of 2-fold CV.
    ``t = d_11 / sqrt(mean_i s_i^2)`` with ``s_i^2 = (d_i1 - dbar_i)^2 + (d_i2 - dbar_i)^2``,
    compared with t(5). Test only: no confidence interval is reported.
    """
    d = np.asarray(diffs, dtype=np.float64)
    if d.shape != (5, 2):
        raise ValueError("diffs must have shape (5, 2)")
    if not np.isfinite(d).all():
        raise ValueError("score differences contain NaN or infinite values")
    s2 = ((d - d.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    denom = float(np.sqrt(s2.mean()))
    if denom == 0:
        raise ValueError("zero variance in score differences: t statistic undefined")
    t = float(d[0, 0] / denom)
    return MethodResult(
        method="five_by_two_cv_t",
        estimate=float(d.mean()),
        ci_low=None,
        ci_high=None,
        ci_level=None,
        p_value=float(2 * stats.t.sf(abs(t), 5)),
        statistic=t,
        df=5.0,
        n_effective=10.0,
        rationale=(
            "Dietterich 5x2cv paired t-test: five replications of 2-fold CV with a pooled "
            "variance estimate; low false-positive rate but needs 10 model fits per variant."
        ),
        diagnostics={"denominator": denom},
    )
