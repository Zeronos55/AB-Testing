import pytest

from isbetter.decision.verdict import CVSpec, decide
from isbetter.methods.cv import cv_variance_factor
from isbetter.methods.power import cv_n_needed, n_needed, target_half_width


def test_half_width_ratio():
    # c=0.1, h=0.3, margin=0 -> target 0.1 -> n * 9
    assert n_needed(-0.2, 0.4, 0.0, 100)[0] == pytest.approx(900.0)
    assert target_half_width(0.1, 0.05) == pytest.approx(0.1)


def test_10x10_cv_says_more_repeats_wont_help():
    _, note = cv_n_needed(-0.01, 0.03, 0.0, 100, 900, 100)
    assert note and "more CV repeats won't help" in note


def test_cv_floor_means_no_repeat_count_suffices():
    # huge h relative to |c|: need factor below the n_test/n_train floor
    needed, note = cv_n_needed(-1.0, 1.2, 0.0, 100, 900, 100)
    assert needed is None
    assert "you need more data" in note


def test_cv_repeats_needed_when_feasible():
    # 5 scores, tiny test fraction: repeats dominate the factor
    L, U = -0.02, 0.06
    needed, note = cv_n_needed(L, U, 0.0, 5, 10_000, 100)
    assert note is None
    factor, _ = cv_variance_factor(5, 10_000, 100)
    h, best = (U - L) / 2, 0.02
    target = factor * (best / h) ** 2
    assert needed == pytest.approx(max(1 / (target - 0.01), 5))
    assert needed > 5


def test_decide_with_cv_spec():
    v = decide(-0.01, 0.03, 0.0, cv=CVSpec(100, 900, 100))
    assert v.label == "inconclusive"
    assert "more CV repeats won't help" in v.n_needed_note
