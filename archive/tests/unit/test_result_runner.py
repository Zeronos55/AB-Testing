import numpy as np

from isbetter.core.config import parse_config
from isbetter.core.plan import build_plan
from isbetter.core.result import Result
from isbetter.core.runner import run
from isbetter.core.schema import build_table


def make(dep, higher=True):
    return parse_config(
        {
            "experiment": "e",
            "data": "d.csv",
            "variants": {"baseline": "a", "candidates": ["b"]},
            "metric": {"name": "m", "higher_is_better": higher},
            "dependence": {"type": dep},
            "decision": {"min_effect": 0.1},
            "seed": 3,
        }
    )


def paired_table(n=80, shift=0.5, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    ids = np.arange(n)
    return build_table(
        {
            "unit_id": np.r_[ids, ids],
            "pair_id": np.r_[ids, ids],
            "variant": ["a"] * n + ["b"] * n,
            "metric": ["m"] * (2 * n),
            "value": np.r_[base, base + shift + rng.normal(0, 0.3, n)],
        }
    )


def test_paired_run_detects_effect_and_roundtrips():
    from isbetter.core.dependence import Paired

    plan = build_plan(make("paired"))
    res = run(plan, paired_table())
    c = res.comparisons[0]
    assert c.verdict.label == "A_better_practical"
    assert len(c.cross_checks) == 2
    again = Result.from_json(res.to_json())
    assert again == res
    assert isinstance(again.plan.dependence, Paired)


def test_lower_is_better_flips_orientation():
    plan = build_plan(make("paired", higher=False))
    c = run(plan, paired_table()).comparisons[0]
    assert c.oriented_estimate < 0
    assert c.verdict.label == "B_better_practical"


def test_iid_run():
    rng = np.random.default_rng(1)
    t = build_table(
        {
            "unit_id": np.arange(400),
            "variant": ["a"] * 200 + ["b"] * 200,
            "metric": ["m"] * 400,
            "value": np.r_[rng.normal(0, 1, 200), rng.normal(0.8, 1, 200)],
        }
    )
    c = run(build_plan(make("iid")), t).comparisons[0]
    assert c.primary.method == "welch_t"
    assert c.verdict.label.startswith("A_better")
