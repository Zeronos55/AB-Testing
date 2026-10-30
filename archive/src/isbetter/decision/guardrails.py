"""Ship / no-ship decision rule for an A/B test with success metrics and guardrails.

Follows the structure of Spotify's decision rules for experiments:

* **Success metrics** are tested for superiority. Multiplicity correction is applied across the
  success hypotheses only (all candidates x success metrics), because those are the tests that
  could each justify shipping.
* **Guardrail metrics** are tested for non-inferiority at the unadjusted alpha: a guardrail
  passes only if we can rule out a regression larger than its margin. Guardrails are
  deliberately not corrected, since adjusting them would make it easier to miss a regression.
* **Ship** a candidate only if at least one success metric wins and no guardrail fails.
  A sample-ratio mismatch overrides everything: results from broken randomisation are not
  evidence.

Everything here is pure (numbers in, :class:`Decision` out).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

WIN_LABELS = ("A_better_practical", "A_better_maybe_trivial")
WORSE_LABELS = ("B_better_practical", "B_better_maybe_trivial")


@dataclass(frozen=True)
class MetricCheck:
    """Outcome of one metric for one candidate.

    ``status`` for a success metric is ``win``, ``trivial_win`` (significant but may be below
    the minimum effect), ``no_win`` or ``worse``; for a guardrail it is ``pass``,
    ``fail_harm`` (a regression beyond the margin is demonstrated) or ``fail_inconclusive``
    (a regression beyond the margin cannot be ruled out). Estimates are oriented so that
    positive means the candidate is better.
    """

    metric: str
    role: str
    status: str
    oriented_estimate: float
    oriented_ci_low: float
    oriented_ci_high: float
    detail: str
    adjusted_p: float | None = None
    margin: float | None = None


@dataclass(frozen=True)
class Decision:
    candidate: str
    ship: bool
    reason: str
    srm_failed: bool = False
    checks: list[MetricCheck] = field(default_factory=list)


def success_check(
    metric: str,
    label: str,
    oriented_estimate: float,
    oriented_ci_low: float,
    oriented_ci_high: float,
    adjusted_p: float | None = None,
) -> MetricCheck:
    """Classify a success metric from its verdict label (CI at the corrected alpha)."""
    if label == "A_better_practical":
        status, why = "win", "improvement is statistically and practically significant"
    elif label == "A_better_maybe_trivial":
        status, why = (
            "trivial_win",
            "improvement is significant but may be below the minimum effect",
        )
    elif label in WORSE_LABELS:
        status, why = "worse", "metric is significantly worse than baseline"
    else:
        status, why = "no_win", "no significant improvement"
    return MetricCheck(
        metric,
        "success",
        status,
        oriented_estimate,
        oriented_ci_low,
        oriented_ci_high,
        why,
        adjusted_p=adjusted_p,
    )


def guardrail_check(
    metric: str,
    oriented_estimate: float,
    oriented_ci_low: float,
    oriented_ci_high: float,
    margin: float,
) -> MetricCheck:
    """Non-inferiority: pass iff the lower confidence limit is above ``-margin``.

    The interval must be the two-sided (1 - 2*alpha) interval, whose lower end is the
    one-sided (1 - alpha) limit, so the test is a one-sided test at the unadjusted alpha.
    """
    if margin <= 0:
        raise ValueError("a guardrail needs a positive non-inferiority margin")
    if oriented_ci_low > -margin:
        status, why = "pass", f"non-inferior: regression larger than {margin:g} is ruled out"
    elif oriented_ci_high < -margin:
        status, why = "fail_harm", f"regression beyond the {margin:g} margin is demonstrated"
    else:
        status, why = (
            "fail_inconclusive",
            f"cannot rule out a regression beyond the {margin:g} margin (too little data)",
        )
    return MetricCheck(
        metric,
        "guardrail",
        status,
        oriented_estimate,
        oriented_ci_low,
        oriented_ci_high,
        why,
        margin=margin,
    )


def decide_ship(
    candidate: str,
    checks: Sequence[MetricCheck],
    srm_failed: bool = False,
    require_practical: bool = False,
) -> Decision:
    """Combine metric checks into the ship decision for one candidate."""
    success = [c for c in checks if c.role == "success"]
    guards = [c for c in checks if c.role == "guardrail"]
    if not success:
        raise ValueError("at least one success metric is required to make a ship decision")
    listed = list(checks)
    if srm_failed:
        return Decision(
            candidate,
            False,
            "sample-ratio mismatch: randomisation did not deliver the planned split, "
            "so the metric comparisons cannot be trusted",
            True,
            listed,
        )
    failed = [c for c in guards if c.status != "pass"]
    if failed:
        names = ", ".join(f"{c.metric} ({c.status.removeprefix('fail_')})" for c in failed)
        return Decision(candidate, False, f"guardrail failed: {names}", False, listed)
    winning = [
        c
        for c in success
        if c.status == "win" or (c.status == "trivial_win" and not require_practical)
    ]
    if not winning:
        trivial = [c.metric for c in success if c.status == "trivial_win"]
        extra = (
            f" ({', '.join(trivial)} improved by a possibly trivial amount; "
            "ship_requires_practical is on)"
            if trivial
            else ""
        )
        return Decision(
            candidate,
            False,
            "no success metric shows a significant improvement" + extra,
            False,
            listed,
        )
    names = ", ".join(
        c.metric + (" (possibly trivial)" if c.status == "trivial_win" else "") for c in winning
    )
    guard_txt = f"; all {len(guards)} guardrail(s) passed" if guards else ""
    return Decision(candidate, True, f"success metric(s) won: {names}{guard_txt}", False, listed)
