"""Comparing forecasts: the Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

Two forecasters are scored on the same targets. With per-observation losses ``L_A,t`` and
``L_B,t`` the loss differential is ``d_t = L_A,t - L_B,t`` and the null is ``E[d_t] = 0``.
Errors of an ``h``-step-ahead forecast made at consecutive origins overlap, so ``d_t`` is
autocorrelated (an MA(h - 1) process under the null) and the variance of its mean must
include autocovariances up to lag ``h - 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult


def _differential(loss_a: np.ndarray, loss_b: np.ndarray) -> np.ndarray:
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("loss series must be 1-D arrays of equal length")
    if len(a) < 3:
        raise ValueError("need at least 3 forecast origins")
    d: np.ndarray = a - b
    if not np.isfinite(d).all():
        raise ValueError("loss series contain NaN or infinite values")
    return d


def autocovariances(d: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocovariances ``gamma_0..gamma_max_lag`` of ``d``, each divided by ``T``.

    The divisor is ``T`` for every lag, not ``T - lag``; this is what makes the long-run
    variance estimate below non-negative for the Bartlett kernel and is the convention of
    R's ``acf(type = "covariance")`` used by ``forecast::dm.test``.
    """
    T = len(d)
    c = d - d.mean()
    return np.array([float(c[k:] @ c[: T - k]) / T for k in range(max_lag + 1)])


def long_run_variance(d: np.ndarray, lags: int, kernel: str = "rectangular") -> float:
    """Long-run variance ``gamma_0 + 2 * sum_k w_k gamma_k`` of ``d`` using ``lags`` lags.

    ``rectangular`` (``w_k = 1``) is Diebold & Mariano's original truncated estimator;
    ``bartlett`` (``w_k = 1 - k/(lags + 1)``, Newey-West) is never negative.
    """
    if kernel not in ("rectangular", "bartlett"):
        raise ValueError("kernel must be 'rectangular' or 'bartlett'")
    gamma = autocovariances(d, lags)
    k = np.arange(1, lags + 1)
    w = np.ones(lags) if kernel == "rectangular" else 1.0 - k / (lags + 1.0)
    return float(gamma[0] + 2.0 * (w * gamma[1:]).sum())


def hln_factor(T: int, horizon: int) -> float:
    """Harvey-Leybourne-Newbold small-sample factor ``sqrt((T + 1 - 2h + h(h-1)/T) / T)``."""
    inner = (T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T
    if inner <= 0:
        raise ValueError(f"horizon {horizon} is too large for {T} forecast origins")
    return float(np.sqrt(inner))


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    alpha: float = 0.05,
    hln: bool = True,
    hac_lags: int | None = None,
) -> MethodResult:
    """Diebold-Mariano test of equal predictive accuracy, with the HLN correction by default.

    Diebold & Mariano (1995, JBES 13); Harvey, Leybourne & Newbold (1997, Int. J. Forecasting
    13). ``loss_a`` and ``loss_b`` are per-origin losses of the two forecasters at forecast
    horizon ``horizon``, in time order; the estimate is ``mean(loss_a - loss_b)``.

    ``DM = mean(d) / sqrt(LRV / T)`` where ``LRV = gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k`` and
    ``gamma_k`` is divided by ``T`` (not ``T - k``). With ``hln`` the statistic is multiplied by
    ``sqrt((T + 1 - 2h + h(h-1)/T) / T)`` and compared with ``t(T - 1)`` (otherwise ``N(0, 1)``).
    The confidence interval inverts that statistic. If ``hac_lags`` is given, the variance
    uses that many lags with Bartlett weights instead (never negative; use it when the
    rectangular estimate is not positive, which can happen for ``h > 1`` and small ``T``).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if hac_lags is not None and hac_lags < 0:
        raise ValueError("hac_lags must be >= 0")
    d = _differential(loss_a, loss_b)
    T = len(d)
    lags = horizon - 1 if hac_lags is None else hac_lags
    if lags >= T:
        raise ValueError(f"{lags} autocovariance lags need more than {T} forecast origins")
    kernel = "rectangular" if hac_lags is None else "bartlett"
    lrv = long_run_variance(d, lags, kernel)
    if lrv <= 0:
        raise ValueError(
            "long-run variance of the loss differential is not positive "
            f"({lrv:.3g}); the {kernel} estimate with {lags} lags failed. Identical forecasts, "
            "or a small sample at horizon > 1; pass hac_lags for the Bartlett estimator"
        )
    mean = float(d.mean())
    se = float(np.sqrt(lrv / T))
    factor = hln_factor(T, horizon) if hln else 1.0
    dm = mean / se
    stat = factor * dm
    if hln:
        df: float | None = float(T - 1)
        p = float(2 * stats.t.sf(abs(stat), T - 1))
        crit = float(stats.t.ppf(1 - alpha / 2, T - 1))
    else:
        df = None
        p = float(2 * stats.norm.sf(abs(stat)))
        crit = float(stats.norm.ppf(1 - alpha / 2))
    half = crit * se / factor
    ref = f"t({T - 1})" if hln else "N(0, 1)"
    return MethodResult(
        method="diebold_mariano_hln" if hln else "diebold_mariano",
        estimate=mean,
        ci_low=mean - half,
        ci_high=mean + half,
        ci_level=1 - alpha,
        p_value=p,
        statistic=float(stat),
        df=df,
        n_effective=float(T),
        rationale=(
            f"Diebold-Mariano test of equal accuracy: dependence=timeseries, horizon {horizon}. "
            f"The {horizon}-step loss differential is autocorrelated, so the variance uses "
            f"{lags} autocovariance lag(s) ({kernel}) instead of assuming independent origins"
            + (f"; Harvey-Leybourne-Newbold small-sample correction, compared with {ref}." if hln
               else f"; no small-sample correction, compared with {ref}.")
        ),
        diagnostics={
            "se": se / factor,
            "dm_statistic": float(dm),
            "hln_factor": factor,
            "long_run_variance": lrv,
            "gamma0": float(autocovariances(d, 0)[0]),
            "lags": lags,
            "kernel": kernel,
            "horizon": horizon,
        },
    )  # fmt: skip


def naive_dm_t(loss_a: np.ndarray, loss_b: np.ndarray, alpha: float = 0.05) -> MethodResult:
    """NAIVE test: paired t-test on the loss differential. For the failure gallery only.

    Treats the forecast origins as independent (variance ``gamma_0 / T``), which is correct
    only for one-step forecasts. At horizon ``h > 1`` the differential is autocorrelated, the
    variance is understated and the false-positive rate exceeds the nominal level.
    """
    d = _differential(loss_a, loss_b)
    T = len(d)
    sd = float(d.std(ddof=1))
    if sd == 0:
        raise ValueError("zero variance in loss differential: t statistic undefined")
    se = sd / np.sqrt(T)
    mean = float(d.mean())
    t = mean / se
    crit = float(stats.t.ppf(1 - alpha / 2, T - 1))
    return MethodResult(
        method="naive_dm_t",
        estimate=mean,
        ci_low=mean - crit * se,
        ci_high=mean + crit * se,
        ci_level=1 - alpha,
        p_value=float(2 * stats.t.sf(abs(t), T - 1)),
        statistic=float(t),
        df=float(T - 1),
        n_effective=float(T),
        rationale=(
            "NAIVE paired t-test on the loss differential: ignores the autocorrelation of "
            "multi-step forecast errors, so it is too confident. Kept only to demonstrate "
            "the failure."
        ),
        diagnostics={"se": float(se), "naive": True},
    )


@dataclass(frozen=True)
class BlockLength:
    """Suggested block length for a block bootstrap of a (loss-differential) series."""

    stationary: float  # mean block length for the stationary bootstrap
    circular: float  # fixed block length for the circular / moving-block bootstrap
    source: str  # "politis_white" or "rule_of_thumb"

    def integer(self, kind: str, n: int) -> int:
        """Whole-number block length for ``kind``, kept within ``1..n``."""
        value = self.stationary if kind == "stationary" else self.circular
        return max(1, min(n, math.ceil(value)))


def suggest_block_length(d: np.ndarray) -> BlockLength:
    """Automatic block length of Politis & White (2004), with the Patton, Politis & White
    (2009) correction, via ``arch.bootstrap.optimal_block_length``.

    Without ``arch`` (the ``[forecast]`` extra) or when the estimator is undefined for the
    series (for example a constant), it falls back to the ``ceil(T**(1/3))`` rule of thumb and
    says so in ``source``; it never fails silently into a different method.
    """
    x = np.asarray(d, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("need a 1-D finite series of at least 3 observations")
    rule = float(math.ceil(len(x) ** (1 / 3)))
    try:
        from arch.bootstrap import optimal_block_length
    except ImportError:
        return BlockLength(rule, rule, "rule_of_thumb")
    if float(x.std()) == 0.0:
        return BlockLength(rule, rule, "rule_of_thumb")
    res = optimal_block_length(x)
    stationary, circular = float(res["stationary"].iloc[0]), float(res["circular"].iloc[0])
    if not (np.isfinite(stationary) and np.isfinite(circular)):
        return BlockLength(rule, rule, "rule_of_thumb")
    return BlockLength(stationary, circular, "politis_white")
