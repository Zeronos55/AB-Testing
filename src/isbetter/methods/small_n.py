"""Intervals that stay honest at small N: LLM evals with 30-200 items (Bowyer et al. 2025).

The usual error bar ``mean +/- 1.96 * sd / sqrt(n)`` leans on the central limit theorem, which
under-covers when n is small and accuracy is near 0 or 1 (see
``validation/failure_gallery/03_clt_small_n.py``). This module provides the alternatives:

* per-arm intervals for a proportion: :func:`clopper_pearson_interval` (exact), Wilson and
  Jeffreys (in :mod:`isbetter.methods.proportions`), :func:`beta_posterior_interval`;
* the *paired* difference of two models scored right/wrong on the same items:
  :func:`paired_binary_score`, a score (Tango 1998) interval plus McNemar p-value, with
  :func:`paired_beta_posterior` as the Bayesian cross-check;
* :func:`clt_interval`, the interval being replaced, kept so it can be compared.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult
from isbetter.methods.classification import mcnemar
from isbetter.methods.proportions import _counts, jeffreys_interval, wilson_interval


def clt_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Normal-theory interval ``mean +/- z * sd / sqrt(n)`` (sd with ddof = 1)."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or len(v) < 2:
        raise ValueError("need a 1-D array with at least 2 values")
    half = float(stats.norm.ppf(1 - alpha / 2)) * float(v.std(ddof=1)) / np.sqrt(len(v))
    m = float(v.mean())
    return m - half, m + half


def clopper_pearson_interval(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact interval from Beta quantiles: ``B(a/2; x, n-x+1)`` and ``B(1-a/2; x+1, n-x)``.

    Guaranteed coverage of at least ``1 - alpha``, hence conservative.
    """
    x, n = _counts(x, n)
    lo = 0.0 if x == 0 else float(stats.beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lo, hi


def beta_posterior_interval(
    x: int, n: int, alpha: float = 0.05, prior: tuple[float, float] = (0.5, 0.5)
) -> tuple[float, float]:
    """Equal-tailed credible interval of Beta(prior_a + x, prior_b + n - x).

    The default prior is Jeffreys' Beta(1/2, 1/2); Beta(1, 1) is the uniform prior. Unlike the
    frequentist Jeffreys interval there is no endpoint modification at x = 0 or x = n.
    """
    x, n = _counts(x, n)
    pa, pb = prior
    if pa <= 0 or pb <= 0:
        raise ValueError("prior parameters must be positive")
    lo, hi = stats.beta.ppf([alpha / 2, 1 - alpha / 2], pa + x, pb + n - x)
    return float(lo), float(hi)


# -- paired difference of two right/wrong score vectors ---------------------------------------
def _binary_pair(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = []
    for name, v in (("a", correct_a), ("b", correct_b)):
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise ValueError(f"scores of {name} must be 1-D")
        if not np.isin(arr, (0, 1)).all():
            raise ValueError(f"scores of {name} are not binary (values other than 0 and 1)")
        out.append(arr.astype(bool))
    if out[0].shape != out[1].shape:
        raise ValueError("a and b must have the same number of items")
    if len(out[0]) < 1:
        raise ValueError("need at least 1 item")
    return out[0], out[1]


def _q_tilde(b: int, c: int, n: int, delta: float) -> float:
    """Restricted MLE of ``p_c`` when ``p_b - p_c = delta`` (maximises the multinomial).

    The stationarity condition ``b/(q+d) + c/q - 2s/(1-2q-d) = 0``, with ``s = n - b - c``
    concordant items, is the quadratic ``2n q^2 - B q - c d (1-d) = 0`` with
    ``B = (b+c)(1-d) - 2d(c+s)``. The log-likelihood is concave in ``q``, so clipping the root
    to the feasible range ``[max(0, -d), (1-d)/2]`` gives the maximiser at the edges too.
    """
    s = n - b - c
    big_b = (b + c) * (1 - delta) - 2 * delta * (c + s)
    disc = big_b * big_b + 8 * n * c * delta * (1 - delta)
    q = (big_b + np.sqrt(max(disc, 0.0))) / (4 * n)
    return float(min(max(q, max(0.0, -delta)), (1 - delta) / 2))


def _score_stat(b: int, c: int, n: int, delta: float) -> float:
    """Score statistic for H0: p_b - p_c = delta. Equals (b-c)/sqrt(b+c) at delta = 0."""
    var = n * (2 * _q_tilde(b, c, n, delta) + delta - delta * delta)
    num = b - c - n * delta
    if var <= 1e-300:
        return 0.0 if abs(num) < 1e-12 else float(np.copysign(np.inf, num))
    return float(num / np.sqrt(var))


def paired_score_interval(b: int, c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Score confidence interval for ``p_b - p_c`` of a paired 2x2 table (Tango 1998).

    ``b`` = items only A got right, ``c`` = only B got right, ``n`` = all items. It inverts the
    score test: the limits are the ``delta`` where ``|T(delta)| = z`` with
    ``T = (b - c - n delta) / sqrt(n (2 q~ + delta - delta^2))`` and ``q~`` the restricted MLE
    of ``p_c`` (:func:`_q_tilde`). Unlike the Wald interval it never has zero width at
    ``b = 0`` or ``c = 0`` and stays inside [-1, 1]; at ``delta = 0`` the squared statistic is
    McNemar's uncorrected chi-square.
    """
    n = int(n)
    if n < 1 or b < 0 or c < 0 or b + c > n:
        raise ValueError("need integers b, c >= 0 with b + c <= n and n >= 1")
    z = float(stats.norm.ppf(1 - alpha / 2))
    est = (b - c) / n

    def limit(lo: float, hi: float, target: float) -> float:
        # T is decreasing in delta; find where it crosses `target` on [lo, hi].
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if _score_stat(b, c, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lower = -1.0 if est == -1.0 else limit(-1.0, est, z)
    upper = 1.0 if est == 1.0 else limit(est, 1.0, -z)
    return lower, upper


def paired_binary_score(
    correct_a: np.ndarray, correct_b: np.ndarray, alpha: float = 0.05
) -> MethodResult:
    """Paired accuracy difference of two models on the same items, valid at small N.

    ``estimate = acc_A - acc_B = (b - c)/n``. The interval is the Tango score interval
    (:func:`paired_score_interval`); the p-value is McNemar's (exact binomial when fewer than
    25 items are discordant). Each arm's Wilson and Jeffreys intervals are reported as
    diagnostics. Only the discordant items carry information about the difference, and
    ``n_effective`` is their number.
    """
    a, bb = _binary_pair(correct_a, correct_b)
    n = len(a)
    b = int((a & ~bb).sum())
    c = int((~a & bb).sum())
    lo, hi = paired_score_interval(b, c, n, alpha)
    mc = mcnemar(a, bb, alpha)
    xa, xb = int(a.sum()), int(bb.sum())
    return MethodResult(
        method="paired_binary_score",
        estimate=(b - c) / n,
        ci_low=lo,
        ci_high=hi,
        ci_level=1 - alpha,
        p_value=mc.p_value,
        statistic=mc.statistic,
        df=mc.df,
        n_effective=float(b + c),
        rationale=(
            f"Paired right/wrong scores on {n} items: Tango score interval for the accuracy "
            "difference (valid when N is small, where the normal-theory interval undercovers) "
            f"with the {'exact' if mc.method == 'mcnemar_exact' else 'chi-square'} McNemar "
            f"p-value; only the {b + c} discordant items (b={b}, c={c}) are informative."
        ),
        diagnostics={
            "b": b,
            "c": c,
            "n": n,
            "acc_a": xa / n,
            "acc_b": xb / n,
            "wilson_a": list(wilson_interval(xa, n, alpha)),
            "wilson_b": list(wilson_interval(xb, n, alpha)),
            "jeffreys_a": list(jeffreys_interval(xa, n, alpha)),
            "jeffreys_b": list(jeffreys_interval(xb, n, alpha)),
            "mcnemar": mc.method,
        },
    )


def paired_beta_posterior(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.05,
    draws: int = 200_000,
    prior: float = 0.5,
) -> MethodResult:
    """Bayesian cross-check: Dirichlet posterior over the 2x2 table, then ``p_b - p_c``.

    The four cells (both right, only A, only B, both wrong) get a Dirichlet(prior) prior (1/2 =
    Jeffreys-type) and counts add to it. The estimate is the posterior mean
    ``(b - c)/(n + 4 prior)``, the interval is equal-tailed, and ``prob_a_better`` is
    ``P(p_b > p_c)``. Independent Beta posteriors per arm would ignore the pairing.
    """
    a, bb = _binary_pair(correct_a, correct_b)
    if prior <= 0:
        raise ValueError("prior must be positive")
    if not 1_000 <= draws <= 5_000_000:
        raise ValueError("draws must be between 1,000 and 5,000,000")
    counts = np.array(
        [(a & bb).sum(), (a & ~bb).sum(), (~a & bb).sum(), (~a & ~bb).sum()], dtype=np.float64
    )
    p = rng.dirichlet(counts + prior, size=draws)
    delta = p[:, 1] - p[:, 2]
    lo, hi = np.quantile(delta, [alpha / 2, 1 - alpha / 2])
    return MethodResult(
        method="paired_beta_posterior",
        estimate=float(delta.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=1 - alpha,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(counts[1] + counts[2]),
        rationale=(
            f"Bayesian cross-check: Dirichlet({prior}) prior on the paired 2x2 table, "
            f"{draws:,} posterior draws; equal-tailed credible interval for the accuracy "
            "difference."
        ),
        diagnostics={
            "prob_a_better": float((delta > 0).mean()),
            "prior": prior,
            "draws": draws,
            "se": float(delta.std(ddof=1)),
        },
    )
