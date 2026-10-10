"""Table-driven tests for the ship / no-ship rule (ticket 2.5)."""

import pytest

from isbetter.decision.guardrails import (
    MetricCheck,
    decide_ship,
    guardrail_check,
    success_check,
)

# ---------------------------------------------------------------- success checks


@pytest.mark.parametrize(
    "label, status",
    [
        ("A_better_practical", "win"),
        ("A_better_maybe_trivial", "trivial_win"),
        ("equivalent", "no_win"),
        ("inconclusive", "no_win"),
        ("B_better_maybe_trivial", "worse"),
        ("B_better_practical", "worse"),
    ],
)
def test_success_status_from_label(label, status):
    assert success_check("m", label, 0.1, 0.0, 0.2).status == status


# ------------------------------------------------------------- guardrail checks


@pytest.mark.parametrize(
    "lo, hi, status",
    [
        (-0.05, 0.30, "pass"),  # lower limit above -0.1: non-inferior
        (0.02, 0.30, "pass"),
        (-0.30, 0.20, "fail_inconclusive"),  # cannot rule out a drop beyond 0.1
        (-0.50, -0.20, "fail_harm"),  # whole interval below -0.1
        (-0.10, 0.20, "fail_inconclusive"),  # exactly on the boundary is not enough
    ],
)
def test_guardrail_non_inferiority(lo, hi, status):
    c = guardrail_check("spend", (lo + hi) / 2, lo, hi, margin=0.1)
    assert c.status == status and c.role == "guardrail" and c.margin == 0.1


def test_guardrail_requires_positive_margin():
    with pytest.raises(ValueError, match="positive"):
        guardrail_check("g", 0.0, -1.0, 1.0, margin=0.0)


# --------------------------------------------------------------- ship decisions


def win(m="conv"):
    return success_check(m, "A_better_practical", 0.1, 0.05, 0.15)


def trivial(m="conv"):
    return success_check(m, "A_better_maybe_trivial", 0.01, 0.001, 0.02)


def flat(m="conv"):
    return success_check(m, "inconclusive", 0.0, -0.1, 0.1)


def worse(m="conv"):
    return success_check(m, "B_better_practical", -0.1, -0.2, -0.05)


def ok_guard(m="spend"):
    return guardrail_check(m, 0.0, -0.05, 0.05, margin=0.1)


def harm_guard(m="spend"):
    return guardrail_check(m, -0.5, -0.6, -0.4, margin=0.1)


def vague_guard(m="spend"):
    return guardrail_check(m, 0.0, -0.4, 0.4, margin=0.1)


CASES = [
    # name, checks, srm_failed, require_practical, ship, reason fragment
    ("win + guardrail ok", [win(), ok_guard()], False, False, True, "won: conv"),
    ("win, no guardrails", [win()], False, False, True, "won: conv"),
    ("win but guardrail harmed", [win(), harm_guard()], False, False, False, "guardrail failed"),
    ("win but guardrail vague", [win(), vague_guard()], False, False, False, "inconclusive"),
    ("no win", [flat(), ok_guard()], False, False, False, "no success metric"),
    ("success worse", [worse(), ok_guard()], False, False, False, "no success metric"),
    ("one of two wins", [flat("a"), win("b"), ok_guard()], False, False, True, "won: b"),
    ("trivial win ships by default", [trivial(), ok_guard()], False, False, True, "won: conv"),
    ("trivial win blocked", [trivial(), ok_guard()], False, True, False, "possibly trivial"),
    (
        "win + one bad of two guards",
        [win(), ok_guard("g1"), harm_guard("g2")],
        False,
        False,
        False,
        "g2",
    ),
    ("SRM overrides a clean win", [win(), ok_guard()], True, False, False, "sample-ratio"),
    (
        "SRM + failed guardrail reports SRM",
        [win(), harm_guard()],
        True,
        False,
        False,
        "sample-ratio",
    ),
]


@pytest.mark.parametrize(
    "name, checks, srm, practical, ship, fragment", CASES, ids=[c[0] for c in CASES]
)
def test_ship_decision_table(name, checks, srm, practical, ship, fragment):
    d = decide_ship("treatment", checks, srm_failed=srm, require_practical=practical)
    assert d.ship is ship, d.reason
    assert fragment in d.reason
    assert d.srm_failed is srm
    assert d.candidate == "treatment" and len(d.checks) == len(checks)


def test_requires_a_success_metric():
    with pytest.raises(ValueError, match="success metric"):
        decide_ship("t", [ok_guard()])


def test_checks_are_returned_unchanged():
    checks = [win(), ok_guard()]
    assert decide_ship("t", checks).checks == checks
    assert all(isinstance(c, MetricCheck) for c in decide_ship("t", checks).checks)
