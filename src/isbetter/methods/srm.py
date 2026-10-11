"""Sample-ratio-mismatch (SRM) check: did randomisation deliver the planned split?

A chi-square goodness-of-fit test of the observed arm counts against the planned allocation.
A tiny p-value means assignment, logging or filtering is broken, so the arm comparison cannot
be trusted no matter what the effect test says. The threshold is deliberately strict
(p < 0.001) because the check runs on every experiment and a false alarm is costly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats

SRM_ALPHA = 0.001
MIN_EXPECTED = 5.0


@dataclass(frozen=True)
class SRMResult:
    flagged: bool
    p_value: float
    statistic: float
    df: int
    alpha: float
    observed: dict[str, int]
    expected: dict[str, float]
    expected_share: dict[str, float]
    approx_ok: bool
    rationale: str


def srm_check(
    counts: Mapping[str, int],
    expected_share: Mapping[str, float] | None = None,
    alpha: float = SRM_ALPHA,
) -> SRMResult:
    """Chi-square test of ``counts`` against ``expected_share`` (equal split when omitted).

    ``chi2 = sum((observed - expected)^2 / expected)`` on ``k - 1`` degrees of freedom, with
    ``expected = total * share``. ``approx_ok`` is False when an expected count is below 5,
    where the chi-square approximation is unreliable.
    """
    if len(counts) < 2:
        raise ValueError("SRM check needs at least 2 arms")
    arms = list(counts)
    obs = np.array([counts[a] for a in arms], dtype=np.float64)
    if (obs < 0).any() or (obs != np.floor(obs)).any():
        raise ValueError("arm counts must be non-negative integers")
    total = float(obs.sum())
    if total <= 0:
        raise ValueError("no units in any arm")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if expected_share is None:
        share = np.full(len(arms), 1.0 / len(arms))
    else:
        if set(expected_share) != set(arms):
            raise ValueError(
                f"expected_share arms {sorted(expected_share)} "
                f"do not match data arms {sorted(arms)}"
            )
        share = np.array([expected_share[a] for a in arms], dtype=np.float64)
        if (share <= 0).any():
            raise ValueError("expected shares must be positive")
        if abs(share.sum() - 1.0) > 1e-6:
            raise ValueError(f"expected shares must sum to 1 (got {share.sum():.6g})")
    exp = total * share
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    df = len(arms) - 1
    p = float(stats.chi2.sf(chi2, df))
    flagged = p < alpha
    approx_ok = bool((exp >= MIN_EXPECTED).all())
    shares_txt = ", ".join(f"{a}: {s:.1%}" for a, s in zip(arms, share, strict=True))
    seen_txt = ", ".join(f"{a}: {int(o):,}" for a, o in zip(arms, obs, strict=True))
    verdict = (
        "SRM DETECTED: the observed split is incompatible with the planned allocation; "
        "assignment or logging is broken and the comparison is not trustworthy."
        if flagged
        else "no sample-ratio mismatch detected."
    )
    return SRMResult(
        flagged=flagged,
        p_value=p,
        statistic=chi2,
        df=df,
        alpha=alpha,
        observed={a: int(o) for a, o in zip(arms, obs, strict=True)},
        expected={a: float(e) for a, e in zip(arms, exp, strict=True)},
        expected_share={a: float(s) for a, s in zip(arms, share, strict=True)},
        approx_ok=approx_ok,
        rationale=(
            f"Chi-square goodness of fit of arm counts ({seen_txt}) against the planned split "
            f"({shares_txt}): chi2 = {chi2:.4g}, df = {df}, p = {p:.3g} vs threshold {alpha:g}; "
            f"{verdict}"
        ),
    )
