"""Normal-mixture confidence sequences: error bars you may look at after every observation.

A fixed-horizon test is valid only if you look once, at the planned sample size. Peeking after
every batch and stopping at the first ``p < 0.05`` inflates the false-positive rate far above 5%
(``validation/failure_gallery/04_peeking.py``). A *confidence sequence* is an interval for the
mean that holds at **all** sample sizes simultaneously, so it may be read at any time, including
adaptively chosen ones, with ``P(truth ever outside) <= alpha``.

Construction (Robbins 1970; Howard et al. 2021; Johari et al. 2022 for the always-valid p-value).
For observations with mean ``mu`` and standard deviation ``s``, let ``S_t = sum(x_i - mu)``.
Mixing the martingale ``exp(lam * S_t / s - lam^2 * t / 2)`` over ``lam ~ N(0, 1/rho)`` gives

    ``M_t = sqrt(rho / (rho + t)) * exp(S_t^2 / (2 s^2 (rho + t)))``,

which has expectation one under the truth, so by Ville's inequality ``P(sup_t M_t >= 1/alpha)
<= alpha``. Solving ``M_t < 1/alpha`` for the mean gives the confidence sequence

    ``xbar_t +/- (s / t) * sqrt((t + rho) * log((t + rho) / (rho * alpha^2)))``.

``rho`` (in units of observations) says where the sequence is tightest; it must be chosen
before seeing the data, which is why a function of the stream length is refused here
(:func:`optimal_rho` takes the planned ``n_target``). The always-valid p-value is
``min(1, 1 / max_{s <= t} M_s)``. With ``s`` known (or a sub-Gaussian bound such as 1/2 for scores
in [0, 1]) the guarantee is exact; with ``s`` estimated by the running sample standard deviation
it is asymptotic (Waudby-Smith et al. 2021) and needs a minimum number of observations
(``min_n``) before the first look: with 2-9 observations the plug-in variance is so noisy that
the false-positive rate exceeds alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from isbetter.core.result import MethodResult

DEFAULT_MIN_N = 20


def mixture_boundary(
    t: np.ndarray | float, rho: np.ndarray | float, alpha: float = 0.05
) -> np.ndarray:
    """``sqrt((t + rho) * log((t + rho) / (rho * alpha^2)))``: the radius in units of ``s / t``."""
    _check_rho_alpha(rho, alpha)
    tt, r = np.asarray(t, dtype=np.float64), np.asarray(rho, dtype=np.float64)
    return np.asarray(np.sqrt((tt + r) * np.log((tt + r) / (r * alpha**2))))


def mixture_martingale(
    t: np.ndarray | float, deviation: np.ndarray | float, s: np.ndarray | float, rho: float
) -> np.ndarray:
    """``M_t`` at sample size ``t``, for ``deviation = xbar_t - mu`` and std. dev. ``s``."""
    tt = np.asarray(t, dtype=np.float64)
    d = np.asarray(deviation, dtype=np.float64)
    return np.asarray(
        np.sqrt(rho / (rho + tt)) * np.exp(tt**2 * d**2 / (2 * np.asarray(s) ** 2 * (rho + tt)))
    )


def optimal_rho(n_target: float, alpha: float = 0.05) -> float:
    """The ``rho`` that makes the interval narrowest at sample size ``n_target``.

    Minimises ``(t + rho) * log((t + rho)/(rho alpha^2))`` over ``rho`` numerically. Howard et al.
    (2021) give the near-identical closed form ``t / (2 log(1/alpha) + log(1 + 2 log(1/alpha)))``.
    """
    if n_target <= 0:
        raise ValueError("n_target must be positive")
    _check_rho_alpha(1.0, alpha)

    def f(log_rho: float) -> float:
        return float(mixture_boundary(n_target, float(np.exp(log_rho)), alpha) ** 2)

    centre = np.log(n_target)
    res = optimize.minimize_scalar(
        f, bounds=(centre - 8, centre + 8), method="bounded", options={"xatol": 1e-10}
    )
    return float(np.exp(res.x))


def _check_rho_alpha(rho: np.ndarray | float, alpha: float) -> None:
    if np.any(np.asarray(rho) <= 0):
        raise ValueError("rho must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


@dataclass(frozen=True)
class ConfidenceSequence:
    """A confidence sequence evaluated at every look from ``n[0]`` on.

    ``lower`` / ``upper`` are intersected over time (``intersect=True``), which is also valid
    and never wider; ``radius`` is the pointwise half-width. ``p_value`` is the always-valid
    p-value (running minimum), so it may only fall.
    """

    n: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    radius: np.ndarray
    p_value: np.ndarray
    scale: np.ndarray  # s: standard deviation per observation (known or running estimate)
    alpha: float
    rho: float
    null: float
    known_sigma: bool

    def first_exclusion(self, value: float | None = None) -> int | None:
        """Smallest sample size at which the sequence excludes ``value`` (default: the null)."""
        v = self.null if value is None else value
        gone = np.flatnonzero((v < self.lower) | (v > self.upper))
        return int(self.n[gone[0]]) if len(gone) else None

    def final(self, name: str, rationale: str, n_effective: float) -> MethodResult:
        """The last look as a :class:`MethodResult` (estimate = mean, or A - B for two arms)."""
        est, lo, hi = float(self.estimate[-1]), float(self.lower[-1]), float(self.upper[-1])
        z = float(stats.norm.ppf(1 - self.alpha / 2))
        fixed = z * float(self.scale[-1]) / float(np.sqrt(self.n[-1]))  # one-look z interval
        return MethodResult(
            method=name,
            estimate=est,
            ci_low=lo,
            ci_high=hi,
            ci_level=1 - self.alpha,
            p_value=float(self.p_value[-1]),
            statistic=None,
            df=None,
            n_effective=n_effective,
            rationale=rationale,
            diagnostics={
                "rho": self.rho,
                "n_looks": len(self.n),
                "first_look": int(self.n[0]),
                "first_exclusion_n": self.first_exclusion(),
                "variance": "known" if self.known_sigma else "plug-in (asymptotic)",
                "fixed_horizon_ci": [est - fixed, est + fixed],
            },
        )


def _resolve_rho(rho: float | None, n_target: float | None, alpha: float) -> float:
    if (rho is None) == (n_target is None):
        raise ValueError(
            "give exactly one of rho or n_target. The sequence must be tuned before looking at "
            "the data; pass the sample size you plan to reach as n_target (not the length of "
            "the data you already have, which may depend on when you stopped)."
        )
    return float(rho) if rho is not None else optimal_rho(float(n_target or 0.0), alpha)


def _sequence(
    t: np.ndarray,
    mean: np.ndarray,
    s: np.ndarray,
    rho: float,
    alpha: float,
    null: float,
    known: bool,
    intersect: bool,
) -> ConfidenceSequence:
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(s > 0, s * mixture_boundary(t, rho, alpha) / t, np.inf)
        m = np.where(s > 0, mixture_martingale(t, mean - null, np.where(s > 0, s, 1.0), rho), 1.0)
    lower, upper = mean - radius, mean + radius
    if intersect:
        lower, upper = np.maximum.accumulate(lower), np.minimum.accumulate(upper)
    p = np.minimum.accumulate(np.minimum(1.0, 1.0 / m))
    return ConfidenceSequence(
        t.astype(int), mean, lower, upper, radius, p, s, alpha, rho, null, known
    )


def confidence_sequence_mean(
    x: np.ndarray,
    alpha: float = 0.05,
    *,
    rho: float | None = None,
    n_target: float | None = None,
    sigma: float | None = None,
    min_n: int = DEFAULT_MIN_N,
    null: float = 0.0,
    intersect: bool = True,
) -> ConfidenceSequence:
    """Confidence sequence for the mean of a stream, looked at after every observation.

    ``sigma`` known (or a sub-Gaussian bound, e.g. ``0.5`` for scores in [0, 1]) gives an exact
    guarantee; otherwise the running sample standard deviation is plugged in (asymptotic) and
    the first look is at ``min_n`` observations. For paired data pass the per-item differences.
    A look whose running standard deviation is zero has an unbounded interval.
    """
    rho_ = _resolve_rho(rho, n_target, alpha)
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("x must be a 1-D stream")
    if not np.isfinite(v).all():
        raise ValueError("x contains NaN or infinite values")
    start = max(int(min_n), 1 if sigma is not None else 2)
    if len(v) < start:
        raise ValueError(f"need at least {start} observations (min_n); got {len(v)}")
    if sigma is not None and sigma <= 0:
        raise ValueError("sigma must be positive")
    t = np.arange(1, len(v) + 1, dtype=np.float64)
    mean = np.cumsum(v) / t
    if sigma is not None:
        s = np.full_like(t, float(sigma))
    else:
        var = np.maximum(np.cumsum(v**2) - t * mean**2, 0.0) / np.maximum(t - 1, 1)
        s = np.sqrt(var)
    sl = slice(start - 1, None)
    return _sequence(t[sl], mean[sl], s[sl], rho_, alpha, null, sigma is not None, intersect)


def confidence_sequence_diff(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    *,
    rho: float | None = None,
    n_target: float | None = None,
    sigma: tuple[float, float] | None = None,
    min_n: int = DEFAULT_MIN_N,
    intersect: bool = True,
) -> ConfidenceSequence:
    """Confidence sequence for ``mean(a) - mean(b)`` of two independent arms, equal counts.

    Looks are at ``n`` observations per arm; the difference has variance ``(s_a^2 + s_b^2)/n``,
    so the stream behaves like one with ``s^2 = s_a^2 + s_b^2``. Arms of unequal length are
    refused rather than silently truncated: cut them to the same length on purpose.
    """
    rho_ = _resolve_rho(rho, n_target, alpha)
    xa, xb = (np.asarray(v, dtype=np.float64) for v in (a, b))
    if xa.ndim != 1 or xb.ndim != 1:
        raise ValueError("a and b must be 1-D streams")
    if len(xa) != len(xb):
        raise ValueError(
            f"arms have {len(xa)} and {len(xb)} observations; the sequence is evaluated at equal "
            "counts per arm, so truncate to the same length explicitly"
        )
    if not (np.isfinite(xa).all() and np.isfinite(xb).all()):
        raise ValueError("a and b contain NaN or infinite values")
    start = max(int(min_n), 1 if sigma is not None else 2)
    if len(xa) < start:
        raise ValueError(f"need at least {start} observations per arm (min_n); got {len(xa)}")
    t = np.arange(1, len(xa) + 1, dtype=np.float64)
    ma, mb = np.cumsum(xa) / t, np.cumsum(xb) / t
    if sigma is not None:
        if min(sigma) <= 0:
            raise ValueError("sigma must be positive")
        s = np.full_like(t, float(np.sqrt(sigma[0] ** 2 + sigma[1] ** 2)))
    else:
        den = np.maximum(t - 1, 1)
        va = np.maximum(np.cumsum(xa**2) - t * ma**2, 0.0) / den
        vb = np.maximum(np.cumsum(xb**2) - t * mb**2, 0.0) / den
        s = np.sqrt(va + vb)
    sl = slice(start - 1, None)
    return _sequence((t)[sl], (ma - mb)[sl], s[sl], rho_, alpha, 0.0, sigma is not None, intersect)


def sequential_mean_result(
    x: np.ndarray,
    alpha: float = 0.05,
    *,
    rho: float | None = None,
    n_target: float | None = None,
    sigma: float | None = None,
    min_n: int = DEFAULT_MIN_N,
) -> MethodResult:
    """The final look of :func:`confidence_sequence_mean` as a :class:`MethodResult`."""
    cs = confidence_sequence_mean(x, alpha, rho=rho, n_target=n_target, sigma=sigma, min_n=min_n)
    return cs.final(
        "confidence_sequence_normal_mixture",
        (
            f"Normal-mixture confidence sequence (rho = {cs.rho:.4g}) after {len(x)} observations: "
            "the interval is valid at every sample size at once, so it can be read after each "
            "observation without inflating the false-positive rate; the p-value is the "
            "always-valid one. It is wider than a fixed-horizon interval, which is the price of "
            "being allowed to look."
        ),
        float(len(x)),
    )
