"""Sample-size helpers.

* :func:`n_needed` / :func:`cv_n_needed`: how much more data would make an inconclusive CI
  decisive (used by the verdict).
* The planner (:func:`sample_size_means`, :func:`sample_size_proportions` and their power / MDE
  companions): how many units a *future* A/B test needs, for ``isbetter plan-size``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from isbetter.methods.cv import cv_variance_factor

TEST_SHARE_LIMIT = 0.8


def target_half_width(center: float, margin: float) -> float:
    """Largest CI half-width that would still give a decisive verdict.

    Candidates: separate from 0 (|c|), clear the margin (|c| - margin) or fit inside the
    equivalence band (margin - |c|). The largest positive one is the cheapest decisive outcome.
    """
    return max(abs(center), abs(center) - margin, margin - abs(center))


def n_needed(ci_low: float, ci_high: float, margin: float, n: float) -> tuple[float | None, str]:
    """Total n for a decisive CI, assuming half-width shrinks like 1/sqrt(n)."""
    h = (ci_high - ci_low) / 2
    best = target_half_width((ci_high + ci_low) / 2, margin)
    if best <= 0 or h <= 0:
        return None, "point estimate sits exactly on a decision boundary; no finite n suffices"
    return float(n * (h / best) ** 2), (
        "assumes CI half-width shrinks like 1/sqrt(n) at the current estimate"
    )


def cv_n_needed(
    ci_low: float, ci_high: float, margin: float, n_scores: int, n_train: int, n_test: int
) -> tuple[float | None, str | None]:
    """Number of CV scores (repeats x folds) needed, or a "get more data" note.

    Under Nadeau-Bengio the half-width scales like sqrt(1/J + n_test/n_train). The second
    term does not shrink with more repeats, so even infinitely many repeats leave a floor.
    """
    factor, share = cv_variance_factor(n_scores, n_train, n_test)
    h = (ci_high - ci_low) / 2
    best = target_half_width((ci_high + ci_low) / 2, margin)
    floor_ratio = n_test / n_train
    note: str | None = None
    if share > TEST_SHARE_LIMIT:
        note = (
            f"more CV repeats won't help — you need more data "
            f"({share:.0%} of the variance factor is the n_test/n_train term)"
        )
    if best <= 0 or h <= 0:
        return None, note or "point estimate sits exactly on a decision boundary"
    target_factor = factor * (best / h) ** 2
    if target_factor <= floor_ratio:
        return None, note or (
            "even infinitely many CV repeats cannot reach a decisive CI — you need more data "
            "(a larger dataset or a smaller test fraction)"
        )
    needed = 1.0 / (target_factor - floor_ratio)
    return float(max(needed, n_scores)), note


# ------------------------------------------------------------------------ planner

_NORMAL_ABOVE_DF = 10_000  # noncentral t and the normal agree far beyond rounding here


@dataclass(frozen=True)
class SizePlan:
    """Result of a sample-size calculation (``ratio = n_treatment / n_control``)."""

    kind: str
    n_control: int
    n_treatment: int
    n_control_exact: float
    power_achieved: float
    alpha: float
    power_target: float
    ratio: float
    two_sided: bool
    delta: float
    method: str
    note: str

    @property
    def n_total(self) -> int:
        return self.n_control + self.n_treatment


def _check_plan(alpha: float, power: float, ratio: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if ratio <= 0:
        raise ValueError("ratio (n_treatment / n_control) must be positive")


def _zcrit(alpha: float, two_sided: bool) -> float:
    return float(stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha))


def _rho_factor(rho: float) -> float:
    if not 0 <= rho < 1:
        raise ValueError("rho (covariate correlation) must be in [0, 1)")
    return 1.0 - rho * rho


def power_means(
    n_control: float, delta: float, sd: float, alpha: float = 0.05, ratio: float = 1.0,
    two_sided: bool = True,
) -> float:  # fmt: skip
    """Power of the two-sample t-test for a mean difference ``delta`` with common SD ``sd``.

    Exact noncentral-t power (the same quantity as ``statsmodels.stats.power.TTestIndPower``),
    switching to the normal limit when the degrees of freedom exceed 10,000.
    """
    _check_plan(alpha, 0.5, ratio)
    if sd <= 0:
        raise ValueError("sd must be positive")
    n1 = float(n_control)
    n2 = n1 * ratio
    df = n1 + n2 - 2
    if df <= 0:
        raise ValueError("sample too small: need more than 2 observations in total")
    nc = abs(delta) / sd * np.sqrt(1.0 / (1.0 / n1 + 1.0 / n2))
    if df <= _NORMAL_ABOVE_DF:
        if two_sided:
            tc = stats.t.ppf(1 - alpha / 2, df)
            exact = float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))
        else:
            exact = float(stats.nct.sf(stats.t.ppf(1 - alpha, df), df, nc))
        if np.isfinite(exact):  # scipy's noncentral t returns NaN for very large nc
            return exact
    zc = _zcrit(alpha, two_sided)
    up = stats.norm.sf(zc - nc)
    return float(up + (stats.norm.cdf(-zc - nc) if two_sided else 0.0))


def _solve_n(power_fn: Callable[[float], float], target: float, lo: float = 2.0) -> float:
    """Smallest control-arm n (continuous) whose power reaches ``target``."""
    hi = lo * 2
    while power_fn(hi) < target:
        hi *= 2
        if hi > 1e12:
            raise ValueError("effect is too small to detect with any realistic sample size")
    if power_fn(lo) >= target:
        return lo
    return float(optimize.brentq(lambda n: power_fn(n) - target, lo, hi, xtol=1e-9, rtol=1e-12))


def sample_size_means(
    delta: float, sd: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0,
    two_sided: bool = True, rho: float = 0.0,
) -> SizePlan:  # fmt: skip
    """Units needed to detect a mean difference ``delta`` (absolute, in metric units).

    ``rho`` is the correlation between the outcome and a CUPED covariate: the outcome
    variance shrinks by ``1 - rho^2``, so the SD used is ``sd * sqrt(1 - rho^2)``.
    """
    _check_plan(alpha, power, ratio)
    if delta == 0:
        raise ValueError("delta must be non-zero")
    sd_eff = sd * float(np.sqrt(_rho_factor(rho)))
    exact = _solve_n(lambda n: power_means(n, delta, sd_eff, alpha, ratio, two_sided), power)
    n1 = int(np.ceil(exact))
    n2 = int(np.ceil(exact * ratio))
    got = power_means(n1, delta, sd_eff, alpha, n2 / n1, two_sided)
    return SizePlan(
        "means", n1, n2, exact, got, alpha, power, ratio, two_sided, delta, "noncentral_t",
        f"sd={sd:g}" + (f", CUPED rho={rho:g} (effective sd {sd_eff:.4g})" if rho else ""),
    )  # fmt: skip


def _solve_up(f: Callable[[float], float], start: float) -> float:
    """Root of an increasing ``f`` (negative at ``start``): grow the bracket, then bisect."""
    hi = start
    while f(hi) < 0:
        hi *= 2
        if hi > 1e12:
            raise ValueError("no effect size reaches the target power")
    lo = hi / 2 if hi > start else 1e-12 * start
    return float(optimize.brentq(f, lo, hi, xtol=1e-14, rtol=1e-12))


def mde_means(
    n_control: float, sd: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0,
    two_sided: bool = True, rho: float = 0.0,
) -> float:  # fmt: skip
    """Smallest absolute mean difference detectable with ``n_control`` units per control arm."""
    _check_plan(alpha, power, ratio)
    sd_eff = sd * float(np.sqrt(_rho_factor(rho)))
    return _solve_up(
        lambda d: power_means(n_control, d, sd_eff, alpha, ratio, two_sided) - power,
        0.01 * sd_eff,
    )


def _prop_params(p_control: float, delta: float) -> tuple[float, float]:
    if not 0 < p_control < 1:
        raise ValueError("p_control must be in (0, 1)")
    p_treat = p_control + delta
    if not 0 < p_treat < 1:
        raise ValueError(f"p_control + delta = {p_treat:g} must be in (0, 1)")
    return p_control, p_treat


def power_proportions(
    n_control: float, p_control: float, delta: float, alpha: float = 0.05, ratio: float = 1.0,
    two_sided: bool = True, method: str = "arcsine",
) -> float:  # fmt: skip
    """Power of a two-proportion z-test.

    ``method="arcsine"``: Cohen's h, the variance-stabilised effect
    ``h = 2 asin(sqrt(p2)) - 2 asin(sqrt(p1))`` (matches ``statsmodels`` ``NormalIndPower``).
    ``method="pooled"``: the textbook (Fleiss) formula with the pooled variance under H0 and
    the unpooled variance under H1; closer to what the Wald/pooled z-test actually does for
    rare events.
    """
    _check_plan(alpha, 0.5, ratio)
    p1, p2 = _prop_params(p_control, delta)
    n1 = float(n_control)
    n2 = n1 * ratio
    zc = _zcrit(alpha, two_sided)
    if method == "arcsine":
        h = abs(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))
        nc = h * np.sqrt(1.0 / (1.0 / n1 + 1.0 / n2))
        low = stats.norm.cdf(-zc - nc) if two_sided else 0.0
        return float(stats.norm.sf(zc - nc) + low)
    if method == "pooled":
        pbar = (n1 * p1 + n2 * p2) / (n1 + n2)
        se0 = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
        se1 = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        d = abs(p2 - p1)
        low = stats.norm.cdf((-d - zc * se0) / se1) if two_sided else 0.0
        return float(stats.norm.sf((zc * se0 - d) / se1) + low)
    raise ValueError("method must be 'arcsine' or 'pooled'")


def sample_size_proportions(
    p_control: float, delta: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0,
    two_sided: bool = True, method: str = "arcsine", rho: float = 0.0,
) -> SizePlan:  # fmt: skip
    """Units needed to detect an absolute lift ``delta`` over the baseline rate ``p_control``.

    ``rho`` approximates CUPED on a binary outcome by scaling the required n by ``1 - rho^2``.
    """
    _check_plan(alpha, power, ratio)
    if delta == 0:
        raise ValueError("delta must be non-zero")
    f = _rho_factor(rho)
    exact = (
        _solve_n(
            lambda n: power_proportions(n, p_control, delta, alpha, ratio, two_sided, method), power
        )
        * f
    )
    exact = max(exact, 1.0)
    n1 = int(np.ceil(exact))
    n2 = int(np.ceil(exact * ratio))
    got = power_proportions(n1, p_control, delta, alpha, n2 / n1, two_sided, method)
    note = f"baseline {p_control:g} -> {p_control + delta:g}"
    if rho:
        note += f"; CUPED rho={rho:g} scales n by {f:.3g} (approximation for binary outcomes)"
    return SizePlan(
        "proportions", n1, n2, exact, got, alpha, power, ratio, two_sided, delta, method, note
    )  # fmt: skip


def mde_proportions(
    n_control: float, p_control: float, alpha: float = 0.05, power: float = 0.8,
    ratio: float = 1.0, two_sided: bool = True, method: str = "arcsine", rho: float = 0.0,
    direction: float = 1.0,
) -> float:  # fmt: skip
    """Smallest absolute lift detectable with ``n_control`` units per control arm.

    ``direction`` is +1 to look for an increase and -1 for a decrease.
    """
    _check_plan(alpha, power, ratio)
    n_eff = n_control / _rho_factor(rho)
    upper = (1 - p_control - 1e-9) if direction > 0 else (p_control - 1e-9)

    def f(d: float) -> float:
        return (
            power_proportions(n_eff, p_control, direction * d, alpha, ratio, two_sided, method)
            - power
        )

    if f(upper) < 0:
        raise ValueError("no lift within (0, 1) is detectable with this sample size")
    return float(optimize.brentq(f, 1e-9, upper, xtol=1e-12))
