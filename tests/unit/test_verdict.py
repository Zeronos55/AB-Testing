import pytest

from isbetter.decision.verdict import decide, mirror_label
from isbetter.methods.power import n_needed

CASES = [
    # (L, U, margin, label)
    (0.12, 0.30, 0.10, "A_better_practical"),
    (0.02, 0.30, 0.10, "A_better_maybe_trivial"),
    (-0.05, 0.05, 0.10, "equivalent"),
    (-0.30, -0.12, 0.10, "B_better_practical"),
    (-0.30, -0.02, 0.10, "B_better_maybe_trivial"),
    (-0.20, 0.20, 0.10, "inconclusive"),
    (-0.01, 0.30, 0.10, "inconclusive"),
]


@pytest.mark.parametrize("L,U,m,label", CASES)
def test_all_cells(L, U, m, label):
    assert decide(L, U, m).label == label


@pytest.mark.parametrize("L,U,m,label", CASES)
def test_swap_ab_mirrors(L, U, m, label):
    assert decide(-U, -L, m).label == mirror_label(label)


def test_zero_margin_has_no_equivalence():
    assert decide(-0.1, 0.1, 0.0).label == "inconclusive"
    assert decide(0.01, 0.1, 0.0).label == "A_better_practical"


def test_inconclusive_has_n_needed():
    v = decide(-0.05, 0.25, 0.10, n=100)
    assert v.label == "inconclusive"
    # c=0.10, h=0.15; best target = max(0.10, 0.0, 0.0)=0.10 -> n = 100*(1.5)^2
    assert v.n_needed == pytest.approx(225.0)


def test_n_needed_on_boundary():
    assert n_needed(-0.1, 0.1, 0.0, 50)[0] is None


def test_bad_inputs():
    with pytest.raises(ValueError):
        decide(1, 0, 0.1)
    with pytest.raises(ValueError):
        decide(0, 1, -0.1)
