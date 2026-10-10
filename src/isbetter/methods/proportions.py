"""Comparing two proportions (conversion rates): z-test, Newcombe CI, Wilson, Jeffreys, Fisher.

All functions report ``estimate = p_a - p_b`` for independent arms (dependence=iid).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult
from isbetter.core.sufficient import SufficientStats

# Below this many successes (or failures) in an arm the Wald interval is unreliable.
SMALL_CELL = 10


def _counts(x: int, n: int, name: str = "arm") -> tuple[int, int]:
    if int(n) != n or int(x) != x:
        raise ValueError(f"{name}: counts must be integers")
    x, n = int(x), int(n)
    if n < 1:
        raise ValueError(f"{name}: n must be >= 1")
    if not 0 <= x <= n:
        raise ValueError(f"{name}: successes must satisfy 0 <= x <= n (got x={x}, n={n})")
    return x, n


def binary_counts(s: SufficientStats, name: str = "arm") -> tuple[int, int]:
    """(successes, n) from the sums of a 0/1 metric; error if the data are not binary."""
    if s.sum_sq != s.sum or not 0 <= s.sum <= s.n:
        raise ValueError(f"{name}: metric is not binary (values other than 0 and 1 present)")
    return int(round(s.sum)), s.n


def wilson_interval(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval: ``(p + z^2/2n +/- z*sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)``."""
    x, n = _counts(x, n)
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = x / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return float((centre - half) / denom), float((centre + half) / denom)


def jeffreys_interval(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Jeffreys interval: Beta(x + 1/2, n - x + 1/2) quantiles.

    Uses the modification of Brown, Cai and DasGupta (2001): the lower limit is 0 when x = 0
    and the upper limit is 1 when x = n, so the interval always contains the sample proportion.
    """
    x, n = _counts(x, n)
    lo = 0.0 if x == 0 else float(stats.beta.ppf(alpha / 2, x + 0.5, n - x + 0.5))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - alpha / 2, x + 0.5, n - x + 0.5))
    return lo, hi


def newcombe_interval(
    x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Newcombe's hybrid score interval for ``p1 - p2`` (method 10 of Newcombe, 1998).

    Built from the two Wilson intervals ``(l_i, u_i)``:
    ``L = d - sqrt((p1 - l1)^2 + (u2 - p2)^2)``, ``U = d + sqrt((u1 - p1)^2 + (p2 - l2)^2)``.
    Keeps good coverage when counts are small, where the Wald interval fails.
    """
    x1, n1 = _counts(x1, n1, "arm a")
    x2, n2 = _counts(x2, n2, "arm b")
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson_interval(x1, n1, alpha)
    l2, u2 = wilson_interval(x2, n2, alpha)
    d = p1 - p2
    return (
        float(d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),
        float(d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)),
    )


def two_proportion_z(
    x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05, ci: str = "auto"
) -> MethodResult:
    """Two-proportion z-test with a Wald or Newcombe confidence interval.

    The test statistic uses the pooled proportion (the variance under H0):
    ``z = (p1 - p2) / sqrt(p(1 - p)(1/n1 + 1/n2))``. The interval is the unpooled Wald
    interval, or Newcombe's when any arm has fewer than 10 successes or failures
    (``ci="auto"``); force one with ``ci="wald"`` or ``ci="newcombe"``.
    """
    x1, n1 = _counts(x1, n1, "arm a")
    x2, n2 = _counts(x2, n2, "arm b")
    if ci not in ("auto", "wald", "newcombe"):
        raise ValueError("ci must be 'auto', 'wald' or 'newcombe'")
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    if pooled in (0.0, 1.0):
        raise ValueError("no successes (or only successes) in both arms: z statistic undefined")
    diff = p1 - p2
    se0 = float(np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)))
    z = diff / se0
    small = min(x1, n1 - x1, x2, n2 - x2) < SMALL_CELL
    use_newcombe = ci == "newcombe" or (ci == "auto" and small)
    se = float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
    if use_newcombe:
        lo, hi = newcombe_interval(x1, n1, x2, n2, alpha)
        why = (
            "Newcombe hybrid-score interval (a cell has fewer than "
            f"{SMALL_CELL} successes or failures, where the Wald interval undercovers)"
            if ci == "auto"
            else "Newcombe hybrid-score interval (requested)"
        )
    else:
        crit = float(stats.norm.ppf(1 - alpha / 2))
        lo, hi = diff - crit * se, diff + crit * se
        why = "Wald interval with unpooled variance"
    return MethodResult(
        method="two_proportion_z",
        estimate=float(diff),
        ci_low=lo,
        ci_high=hi,
        ci_level=1 - alpha,
        p_value=float(2 * stats.norm.sf(abs(z))),
        statistic=float(z),
        df=None,
        n_effective=float(n1 + n2),
        rationale=(
            "Two-proportion z-test (binary metric, independent units): the p-value uses the "
            f"pooled variance under H0; the CI is a {why}."
        ),
        diagnostics={
            "p_a": p1,
            "p_b": p2,
            "se": se,
            "ci_method": "newcombe" if use_newcombe else "wald",
            "wilson_a": wilson_interval(x1, n1, alpha),
            "wilson_b": wilson_interval(x2, n2, alpha),
        },
    )


def fisher_exact(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> MethodResult:
    """Fisher's exact test (two-sided) on the 2x2 table; conditional on the margins.

    Gives a p-value but no interval, so it serves as a cross-check of the z-test. ``alpha`` is
    accepted for interface symmetry with the other methods.
    """
    x1, n1 = _counts(x1, n1, "arm a")
    x2, n2 = _counts(x2, n2, "arm b")
    odds, p = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]], alternative="two-sided")
    return MethodResult(
        method="fisher_exact",
        estimate=float(x1 / n1 - x2 / n2),
        ci_low=None,
        ci_high=None,
        ci_level=None,
        p_value=float(p),
        statistic=float(odds),
        df=None,
        n_effective=float(n1 + n2),
        rationale="Fisher's exact test: exact p-value conditional on the table margins "
        "(cross-check; valid for any counts).",
        diagnostics={"odds_ratio": float(odds)},
    )


def two_proportion_z_stats(
    a: SufficientStats, b: SufficientStats, alpha: float = 0.05, ci: str = "auto"
) -> MethodResult:
    """:func:`two_proportion_z` from the aggregated statistics of two 0/1 metrics."""
    (x1, n1), (x2, n2) = binary_counts(a, "arm a"), binary_counts(b, "arm b")
    return two_proportion_z(x1, n1, x2, n2, alpha, ci)


def fisher_exact_stats(a: SufficientStats, b: SufficientStats, alpha: float = 0.05) -> MethodResult:
    """:func:`fisher_exact` from the aggregated statistics of two 0/1 metrics."""
    (x1, n1), (x2, n2) = binary_counts(a, "arm a"), binary_counts(b, "arm b")
    return fisher_exact(x1, n1, x2, n2, alpha)
