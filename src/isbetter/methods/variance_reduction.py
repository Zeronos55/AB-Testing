"""Variance reduction and ratio metrics for independent A/B arms.

* :func:`cuped`: CUPED (Deng et al., 2013), one common slope on a pre-experiment covariate.
* :func:`regression_adjustment`: arm-specific slopes (Lin, 2013).
* :func:`delta_method_ratio`: ratio-of-means metrics such as revenue per visit.

All take :class:`~isbetter.core.sufficient.SufficientStats` for the two arms, so they work on
14-million-row tables without loading them. ``estimate = A - B``. For the covariate methods
the stats' ``x`` moments are the covariate; for the ratio method they are the denominator.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult
from isbetter.core.sufficient import SufficientStats


def _need_x(a: SufficientStats, b: SufficientStats, what: str) -> None:
    if not (a.has_x and b.has_x):
        raise ValueError(f"{what} needs x moments (covariate or denominator) for both arms")
    for name, s in (("a", a), ("b", b)):
        if s.n < 4:
            raise ValueError(f"arm {name} needs at least 4 observations")


def _welch_df(se2_a: float, se2_b: float, na: int, nb: int) -> float:
    return float((se2_a + se2_b) ** 2 / (se2_a**2 / (na - 1) + se2_b**2 / (nb - 1)))


def _resid_var(s: SufficientStats, theta: float) -> float:
    """Sample variance of ``y - theta * x`` within one arm, from the moments."""
    return max(s.var - 2 * theta * s.cov_xy + theta * theta * s.var_x, 0.0)


def _within_sums(a: SufficientStats, b: SufficientStats) -> tuple[float, float, float]:
    """Pooled within-arm centred sums (Sxx, Sxy, Syy)."""
    sxx = sxy = syy = 0.0
    for s in (a, b):
        assert s.sum_x is not None and s.sum_x_sq is not None and s.sum_xy is not None
        sxx += s.sum_x_sq - s.sum_x**2 / s.n
        sxy += s.sum_xy - s.sum_x * s.sum / s.n
        syy += s.sum_sq - s.sum**2 / s.n
    return sxx, sxy, syy


def cuped(a: SufficientStats, b: SufficientStats, alpha: float = 0.05) -> MethodResult:
    """CUPED estimate of ``mean(A) - mean(B)`` using one pre-experiment covariate ``x``.

    ``theta = Sxy / Sxx`` with both sums centred within arm and pooled across arms, and
    ``estimate = (ybar_A - ybar_B) - theta * (xbar_A - xbar_B)``. This equals the coefficient
    on treatment in the OLS regression ``y ~ treatment + x``. Because randomisation makes
    ``xbar_A - xbar_B`` mean zero, the estimate is unbiased for any theta and the variance
    shrinks by about ``1 - rho^2``. The SE treats theta as fixed (``var(y - theta x)`` per arm);
    degrees of freedom are Welch's minus one for the estimated slope.
    """
    _need_x(a, b, "CUPED")
    sxx, sxy, syy = _within_sums(a, b)
    if sxx <= 0:
        raise ValueError("covariate has no variance: CUPED undefined")
    theta = sxy / sxx
    est = (a.mean - b.mean) - theta * (a.mean_x - b.mean_x)
    se2_a, se2_b = _resid_var(a, theta) / a.n, _resid_var(b, theta) / b.n
    if se2_a + se2_b == 0:
        raise ValueError("zero variance after adjustment: t statistic undefined")
    se = float(np.sqrt(se2_a + se2_b))
    df = max(_welch_df(se2_a, se2_b, a.n, b.n) - 1.0, 1.0)
    t = est / se
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    se_raw = float(np.sqrt(a.var / a.n + b.var / b.n))
    rho = sxy / float(np.sqrt(sxx * syy)) if syy > 0 else 0.0
    return MethodResult(
        method="cuped",
        estimate=float(est),
        ci_low=float(est - crit * se),
        ci_high=float(est + crit * se),
        ci_level=1 - alpha,
        p_value=float(2 * stats.t.sf(abs(t), df)),
        statistic=float(t),
        df=df,
        n_effective=float(a.n + b.n),
        rationale=(
            "CUPED: outcome adjusted with a pre-experiment covariate "
            f"(theta = {theta:.4g}, within-arm correlation rho = {rho:.3f}); randomisation keeps "
            "the estimate unbiased while the variance shrinks by about 1 - rho^2."
        ),
        diagnostics={
            "theta": float(theta),
            "rho": rho,
            "se": se,
            "se_unadjusted": se_raw,
            "variance_reduction": float(1 - se**2 / se_raw**2) if se_raw > 0 else 0.0,
            "predicted_variance_reduction": rho * rho,
        },
    )


def regression_adjustment(
    a: SufficientStats, b: SufficientStats, alpha: float = 0.05
) -> MethodResult:
    """Regression-adjusted difference with arm-specific slopes (Lin, 2013).

    Each arm's mean is adjusted to the pooled covariate mean,
    ``ybar_k - theta_k * (xbar_k - xbar)`` with ``theta_k = cov_k(x, y) / var_k(x)``. The
    estimate equals the treatment coefficient of ``y ~ T + xc + T:xc`` with ``xc`` centred at
    the overall mean. It never does worse than the unadjusted difference asymptotically, even
    when the covariate relates to the outcome differently in the two arms.
    """
    _need_x(a, b, "regression adjustment")
    if a.var_x <= 0 or b.var_x <= 0:
        raise ValueError("covariate has no variance in an arm: regression adjustment undefined")
    xbar = (a.sum_x + b.sum_x) / (a.n + b.n)  # type: ignore[operator]
    adj, se2 = [], []
    for s in (a, b):
        theta = s.cov_xy / s.var_x
        adj.append(s.mean - theta * (s.mean_x - xbar))
        se2.append(_resid_var(s, theta) / s.n)
    est = adj[0] - adj[1]
    if se2[0] + se2[1] == 0:
        raise ValueError("zero variance after adjustment: t statistic undefined")
    se = float(np.sqrt(se2[0] + se2[1]))
    df = max(_welch_df(se2[0], se2[1], a.n, b.n) - 2.0, 1.0)
    t = est / se
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    se_raw = float(np.sqrt(a.var / a.n + b.var / b.n))
    return MethodResult(
        method="regression_adjustment",
        estimate=float(est),
        ci_low=float(est - crit * se),
        ci_high=float(est + crit * se),
        ci_level=1 - alpha,
        p_value=float(2 * stats.t.sf(abs(t), df)),
        statistic=float(t),
        df=df,
        n_effective=float(a.n + b.n),
        rationale=(
            "Regression adjustment with arm-specific covariate slopes (Lin 2013): "
            "robust to different covariate relationships in the two arms."
        ),
        diagnostics={
            "se": se,
            "se_unadjusted": se_raw,
            "variance_reduction": float(1 - se**2 / se_raw**2) if se_raw > 0 else 0.0,
        },
    )


def _ratio_arm(s: SufficientStats, name: str) -> tuple[float, float]:
    """(ratio, delta-method variance of the ratio) for one arm; x moments are the denominator."""
    if s.n < 2:
        raise ValueError(f"arm {name} needs at least 2 observations")
    sx = s.sum_x
    assert sx is not None
    if sx == 0:
        raise ValueError(f"arm {name}: denominator sums to zero, ratio undefined")
    r = s.sum / sx
    zbar = s.mean_x
    var = (s.var - 2 * r * s.cov_xy + r * r * s.var_x) / (s.n * zbar * zbar)
    return r, max(var, 0.0)


def delta_method_ratio(a: SufficientStats, b: SufficientStats, alpha: float = 0.05) -> MethodResult:
    """Difference of ratios-of-means ``sum(y)/sum(x)`` between independent arms.

    First-order delta method: ``Var(R) ~ (var_y - 2 R cov_xy + R^2 var_x) / (n * xbar^2)`` per
    arm, ``SE = sqrt(Var_A + Var_B)``, normal inference. Appropriate when units (users) carry
    different numbers of denominator events (visits, sessions), so a plain per-row mean would
    treat correlated events as independent. Asymptotic: it needs a denominator whose mean is
    many standard errors from zero (reported as ``denominator_z``).
    """
    _need_x(a, b, "delta method")
    ra, va = _ratio_arm(a, "a")
    rb, vb = _ratio_arm(b, "b")
    if va + vb == 0:
        raise ValueError("zero delta-method variance: z statistic undefined")
    est = ra - rb
    se = float(np.sqrt(va + vb))
    z = est / se
    crit = float(stats.norm.ppf(1 - alpha / 2))
    den_z = min(
        s.mean_x * np.sqrt(s.n) / np.sqrt(s.var_x) if s.var_x > 0 else float("inf") for s in (a, b)
    )
    return MethodResult(
        method="delta_method",
        estimate=float(est),
        ci_low=float(est - crit * se),
        ci_high=float(est + crit * se),
        ci_level=1 - alpha,
        p_value=float(2 * stats.norm.sf(abs(z))),
        statistic=float(z),
        df=None,
        n_effective=float(a.n + b.n),
        rationale=(
            "Delta method for a ratio-of-means metric: per-unit numerators and denominators are "
            "correlated, so the variance of each ratio is linearised rather than treating "
            "the ratio as a plain mean."
        ),
        diagnostics={
            "ratio_a": float(ra),
            "ratio_b": float(rb),
            "se": se,
            "denominator_z": float(den_z),
        },
    )


def ratio_bootstrap(
    num_a: np.ndarray,
    den_a: np.ndarray,
    num_b: np.ndarray,
    den_b: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 9_999,
    alpha: float = 0.05,
    batch: int = 200,
) -> MethodResult:
    """Percentile bootstrap of ``sum(num)/sum(den)`` per arm, resampling units (cross-check)."""
    arrs = [np.asarray(v, dtype=np.float64) for v in (num_a, den_a, num_b, den_b)]
    if len(arrs[0]) != len(arrs[1]) or len(arrs[2]) != len(arrs[3]):
        raise ValueError("numerator and denominator must have equal length per arm")
    if min(len(arrs[0]), len(arrs[2])) < 2:
        raise ValueError("each arm needs at least 2 units")
    if not all(np.isfinite(v).all() for v in arrs):
        raise ValueError("values contain NaN or infinite values")
    ya, za, yb, zb = arrs
    if za.sum() == 0 or zb.sum() == 0:
        raise ValueError("denominator sums to zero, ratio undefined")
    est = float(np.sum(ya)) / float(np.sum(za)) - float(np.sum(yb)) / float(np.sum(zb))
    na, nb = len(ya), len(yb)
    out = np.empty(n_resamples)
    step = max(1, min(batch, 2_000_000 // max(na, nb)))
    for start in range(0, n_resamples, step):
        stop = min(start + step, n_resamples)
        ia = rng.integers(0, na, size=(stop - start, na))
        ib = rng.integers(0, nb, size=(stop - start, nb))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:stop] = ya[ia].sum(1) / za[ia].sum(1) - yb[ib].sum(1) / zb[ib].sum(1)
    ok = out[np.isfinite(out)]
    if len(ok) < 0.9 * n_resamples:
        raise ValueError("too many bootstrap resamples had a zero denominator")
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    return MethodResult(
        method="ratio_bootstrap",
        estimate=est,
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=1 - alpha,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(na + nb),
        rationale="Percentile bootstrap resampling units, recomputing each arm's ratio of sums.",
        diagnostics={"se": float(ok.std(ddof=1)), "n_resamples": n_resamples},
    )
