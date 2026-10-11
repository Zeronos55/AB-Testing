import numpy as np
import pytest

from isbetter.methods.superior_predictive import model_confidence_set, spa_test, stepm

RNG = np.random.default_rng(0)
BENCH = RNG.standard_normal(60) ** 2
MODELS = {"a": RNG.standard_normal(60) ** 2, "b": RNG.standard_normal(60) ** 2}


def rng():
    return np.random.default_rng(1)


def test_result_shape_and_rationale():
    r = spa_test(BENCH, MODELS, rng(), reps=200)
    assert r.method == "hansen_spa" and 0 <= r.p_value <= 1 and r.n_effective == 60
    assert "Superior Predictive Ability" in r.rationale and "block bootstrap" in r.rationale
    assert set(r.diagnostics["pvalues"]) == {"lower", "consistent", "upper"}
    s = stepm(BENCH, MODELS, rng(), reps=200)
    assert s.method == "romano_wolf_stepm" and "StepM" in s.rationale
    m = model_confidence_set({"x": BENCH, **MODELS}, rng(), reps=200)
    assert m.method == "model_confidence_set" and "Model Confidence Set" in m.rationale
    assert m.estimate == len(m.diagnostics["included"])


def test_circular_bootstrap_and_options_are_accepted():
    r = spa_test(BENCH, MODELS, rng(), reps=200, bootstrap="circular", studentize=False)
    assert r.diagnostics["bootstrap"] == "circular" and r.diagnostics["studentize"] is False
    m = model_confidence_set({"x": BENCH, **MODELS}, rng(), reps=200, method="max")
    assert m.diagnostics["method"] == "max"


@pytest.mark.parametrize("fn", [spa_test, stepm])
def test_benchmark_validation(fn):
    with pytest.raises(ValueError, match="at least 10"):
        fn(BENCH[:5], {"a": BENCH[:5]}, rng())
    with pytest.raises(ValueError, match="benchmark's length"):
        fn(BENCH, {"a": BENCH[:-1]}, rng())
    with pytest.raises(ValueError, match="at least one model"):
        fn(BENCH, {}, rng())
    with pytest.raises(ValueError, match="NaN"):
        fn(BENCH, {"a": np.where(np.arange(60) == 3, np.nan, BENCH)}, rng())
    with pytest.raises(ValueError, match="bootstrap"):
        fn(BENCH, MODELS, rng(), bootstrap="moving")
    with pytest.raises(ValueError, match="reps"):
        fn(BENCH, MODELS, rng(), reps=10)
    with pytest.raises(ValueError, match="block_len"):
        fn(BENCH, MODELS, rng(), reps=200, block_len=0)


def test_mcs_validation():
    with pytest.raises(ValueError, match="at least two"):
        model_confidence_set({"x": BENCH}, rng())
    with pytest.raises(ValueError, match="method"):
        model_confidence_set({"x": BENCH, "y": BENCH[::-1]}, rng(), method="Q")
    with pytest.raises(ValueError, match="at least 10"):
        model_confidence_set({"x": BENCH[:5], "y": BENCH[:5]}, rng())
    with pytest.raises(ValueError, match="size"):
        model_confidence_set({"x": BENCH, "y": BENCH[::-1]}, rng(), size=1.5)
