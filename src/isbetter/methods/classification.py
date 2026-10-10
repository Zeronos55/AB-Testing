"""Paired comparison of two classifiers on the same test items: McNemar's test."""

from __future__ import annotations

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult

_EXACT_BELOW = 25


def mcnemar(
    correct_a: np.ndarray, correct_b: np.ndarray, alpha: float = 0.05, exact: bool | None = None
) -> MethodResult:
    """McNemar's test on per-item correctness (A right/wrong vs B right/wrong).

    b = A right & B wrong, c = A wrong & B right. Exact two-sided binomial test
    ``p = min(1, 2 * BinomCDF(min(b, c); b + c, 0.5))`` when b + c < 25 (or ``exact=True``),
    otherwise the continuity-corrected chi-square ``(|b - c| - 1)^2 / (b + c)`` on 1 df.
    The estimate is the accuracy difference ``(b - c) / n`` with a Wald interval.
    """
    a = np.asarray(correct_a, dtype=bool)
    bb = np.asarray(correct_b, dtype=bool)
    if a.shape != bb.shape or a.ndim != 1:
        raise ValueError("correct_a and correct_b must be 1-D arrays of equal length")
    n = len(a)
    if n < 1:
        raise ValueError("need at least 1 item")
    b = int((a & ~bb).sum())
    c = int((~a & bb).sum())
    disc = b + c
    use_exact = disc < _EXACT_BELOW if exact is None else exact
    if disc == 0:
        p, stat = 1.0, None
    elif use_exact:
        p = float(min(1.0, 2 * stats.binom.cdf(min(b, c), disc, 0.5)))
        stat = float(min(b, c))
    else:
        stat = (abs(b - c) - 1) ** 2 / disc
        p = float(stats.chi2.sf(stat, 1))
    est = (b - c) / n
    se = float(np.sqrt(max(disc - (b - c) ** 2 / n, 0.0)) / n)
    z = float(stats.norm.ppf(1 - alpha / 2))
    return MethodResult(
        method="mcnemar_exact" if use_exact else "mcnemar_chi2",
        estimate=est,
        ci_low=est - z * se,
        ci_high=est + z * se,
        ci_level=1 - alpha,
        p_value=p,
        statistic=stat,
        df=None if use_exact else 1.0,
        n_effective=float(disc),
        rationale=(
            "McNemar's test on items both models classified: only the discordant pairs "
            f"(b={b}, c={c}) carry information. "
            + (
                "Exact binomial because few discordant pairs."
                if use_exact
                else "Continuity-corrected chi-square."
            )
        ),
        diagnostics={"b": b, "c": c, "n": n},
    )
