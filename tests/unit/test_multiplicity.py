import numpy as np
import pytest

from isbetter.methods.classification import mcnemar
from isbetter.methods.multiplicity import adjust, holm


def test_holm_hand_example():
    # sorted p: .01,.03,.04 -> min(1,3*.01)=.03, 2*.03=.06, 1*.04=.04->monotone .06
    assert np.allclose(holm(np.array([0.04, 0.01, 0.03])), [0.06, 0.03, 0.06])


def test_bad_inputs():
    with pytest.raises(ValueError):
        adjust(np.array([1.2]), "holm")
    with pytest.raises(ValueError, match="unknown"):
        adjust(np.array([0.1]), "tukey")
    with pytest.raises(ValueError):
        adjust(np.array([]), "none")
    assert adjust(np.array([0.2]), "none")[0] == 0.2


def test_mcnemar_validation():
    with pytest.raises(ValueError):
        mcnemar(np.array([True]), np.array([True, False]))
