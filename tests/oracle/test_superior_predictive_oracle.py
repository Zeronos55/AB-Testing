"""Wrapper tests: our SPA / StepM / MCS must reproduce arch's own results for the same data and
the same random stream, and recover planted structure. arch is the reference implementation
(Hansen 2005; Romano & Wolf 2005; Hansen, Lunde & Nason 2011), so the wrappers only need to get
the plumbing (loss sign, model names, seed, block length) right."""

import numpy as np
import pandas as pd
from arch.bootstrap import MCS, SPA, StepM

from isbetter.methods.superior_predictive import model_confidence_set, spa_test, stepm

T, REPS, BLOCK = 400, 400, 8


def make_losses(seed=0):
    """Squared-error-like losses: 'good' clearly beats the benchmark, 'same' ties, 'bad' loses."""
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(T) ** 2 * 0.3  # common component: dependent across models
    bench = shared + rng.standard_normal(T) ** 2
    models = {
        "good": shared + 0.6 * rng.standard_normal(T) ** 2,
        "same": shared + rng.standard_normal(T) ** 2,
        "bad": shared + 1.5 * rng.standard_normal(T) ** 2,
    }
    return bench, models


def test_spa_matches_arch_for_the_same_stream():
    bench, models = make_losses()
    ours = spa_test(bench, models, np.random.default_rng(5), reps=REPS, block_len=BLOCK)
    ref = SPA(
        pd.Series(bench), pd.DataFrame(models), block_size=BLOCK, reps=REPS,
        seed=np.random.default_rng(5),
    )  # fmt: skip
    ref.compute()
    assert ours.diagnostics["pvalues"] == {k: float(v) for k, v in ref.pvalues.items()}
    assert ours.p_value == float(ref.pvalues["consistent"])
    assert ours.diagnostics["better_models"] == [
        list(models)[i] for i in ref.better_models(0.05, "consistent")
    ]


def test_stepm_matches_arch_for_the_same_stream():
    bench, models = make_losses(1)
    ours = stepm(bench, models, np.random.default_rng(6), size=0.1, reps=REPS, block_len=BLOCK)
    ref = StepM(
        pd.Series(bench), pd.DataFrame(models), size=0.1, block_size=BLOCK, reps=REPS,
        seed=np.random.default_rng(6),
    )  # fmt: skip
    ref.compute()
    assert ours.diagnostics["superior_models"] == [str(m) for m in ref.superior_models]


def test_mcs_matches_arch_for_the_same_stream():
    bench, models = make_losses(2)
    losses = {"benchmark": bench, **models}
    ours = model_confidence_set(losses, np.random.default_rng(7), 0.1, REPS, block_len=BLOCK)
    ref = MCS(pd.DataFrame(losses), 0.1, reps=REPS, block_size=BLOCK, seed=np.random.default_rng(7))
    ref.compute()
    assert ours.diagnostics["included"] == [str(m) for m in ref.included]
    assert ours.diagnostics["excluded"] == [str(m) for m in ref.excluded]
    assert ours.diagnostics["pvalues"] == {
        str(k): float(v) for k, v in ref.pvalues["Pvalue"].items()
    }
    assert ours.estimate == len(ref.included)


def test_planted_structure_is_recovered():
    bench, models = make_losses(3)
    spa = spa_test(bench, models, np.random.default_rng(8), reps=REPS)
    assert spa.p_value < 0.01 and "good" in spa.diagnostics["better_models"]
    assert spa.diagnostics["best_model"] == "good" and spa.estimate < 0
    sm = stepm(bench, models, np.random.default_rng(8), reps=REPS)
    assert sm.diagnostics["superior_models"] == ["good"]
    mcs = model_confidence_set({"benchmark": bench, **models}, np.random.default_rng(8), reps=REPS)
    assert "good" in mcs.diagnostics["included"] and "bad" in mcs.diagnostics["excluded"]


def test_null_world_does_not_find_a_superior_model():
    rng = np.random.default_rng(9)
    bench = rng.standard_normal(T) ** 2
    models = {f"m{i}": rng.standard_normal(T) ** 2 for i in range(5)}
    assert spa_test(bench, models, np.random.default_rng(1), reps=REPS).p_value > 0.05
    assert (
        stepm(bench, models, np.random.default_rng(1), reps=REPS).diagnostics["superior_models"]
        == []
    )
    mcs = model_confidence_set({"benchmark": bench, **models}, np.random.default_rng(1), reps=REPS)
    assert len(mcs.diagnostics["included"]) == 6  # nobody can be eliminated


def test_reproducible_given_the_generator():
    bench, models = make_losses(4)
    a = spa_test(bench, models, np.random.default_rng(11), reps=REPS)
    b = spa_test(bench, models, np.random.default_rng(11), reps=REPS)
    assert a.diagnostics["pvalues"] == b.diagnostics["pvalues"]


def test_default_block_length_comes_from_politis_white():
    bench, models = make_losses(5)
    r = spa_test(bench, models, np.random.default_rng(12), reps=REPS)
    assert r.diagnostics["block_len_source"] == "politis_white"
    assert 1 <= r.diagnostics["block_len"] <= T
    assert (
        spa_test(bench, models, np.random.default_rng(12), reps=REPS, block_len=5).diagnostics[
            "block_len_source"
        ]
        == "user"
    )
