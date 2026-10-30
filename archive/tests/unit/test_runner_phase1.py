import numpy as np
import pytest

from isbetter.core.config import parse_config
from isbetter.core.plan import build_plan
from isbetter.core.result import Result
from isbetter.core.runner import run
from isbetter.core.schema import SchemaError, build_table


def cfg(dep, cands=("b",), correction="none", min_effect=0.0):
    return parse_config(
        {
            "experiment": "e",
            "data": "d.csv",
            "variants": {"baseline": "a", "candidates": list(cands)},
            "metric": {"name": "m"},
            "dependence": dep,
            "decision": {"min_effect": min_effect, "correction": correction},
            "seed": 1,
        }
    )


def cv_table(rng, J=100, shifts=(("b", 0.0),)):
    ids = [f"r{i // 10}f{i % 10}" for i in range(J)]
    base = rng.normal(0.5, 0.02, J)
    cols = {"unit_id": [], "pair_id": [], "variant": [], "metric": [], "value": []}
    for v, vals in (("a", base), *((n, base + s + rng.normal(0, 0.01, J)) for n, s in shifts)):
        cols["unit_id"] += ids
        cols["pair_id"] += ids
        cols["variant"] += [v] * J
        cols["metric"] += ["m"] * J
        cols["value"] += list(vals)
    return build_table(cols)


CV = {"type": "cv_overlap", "n_train": 900, "n_test": 100, "repeats": 10, "folds": 10}


def test_cv_overlap_run_uses_nadeau_bengio_and_roundtrips():
    t = cv_table(np.random.default_rng(0), shifts=(("b", 0.002),))
    res = run(build_plan(cfg(CV)), t)
    c = res.comparisons[0]
    assert c.primary.method == "nadeau_bengio"
    assert c.n_pairs == 100
    assert Result.from_json(res.to_json()) == res


def test_cv_wrong_score_count():
    t = cv_table(np.random.default_rng(0), J=50)
    with pytest.raises(SchemaError, match="repeats\\*folds"):
        run(build_plan(cfg(CV)), t)


def test_inconclusive_cv_gets_data_advice():
    t = cv_table(np.random.default_rng(3), shifts=(("b", 0.0005),))
    c = run(build_plan(cfg(CV)), t).comparisons[0]
    assert c.verdict.label == "inconclusive"
    assert "more CV repeats won't help" in c.verdict.n_needed_note


def test_holm_across_candidates():
    rng = np.random.default_rng(5)
    t = cv_table(rng, shifts=(("b", 0.05), ("c", 0.0), ("d", 0.0)))
    res = run(build_plan(cfg(CV, cands=("b", "c", "d"), correction="holm")), t)
    ps = [c.primary.p_value for c in res.comparisons]
    adj = [c.adjusted_p for c in res.comparisons]
    assert all(a >= p for a, p in zip(adj, ps, strict=True))
    assert res.comparisons[0].alpha_used == pytest.approx(0.05 / 3)
    assert res.comparisons[0].verdict.label.startswith("A_better")


def test_clustered_run():
    rng = np.random.default_rng(2)
    G, per = 30, 6
    cl = np.repeat(np.arange(G), per)
    eff = rng.normal(0, 1, G)[cl]
    cols = {
        "unit_id": np.arange(2 * G * per),
        "variant": ["a"] * (G * per) + ["b"] * (G * per),
        "metric": ["m"] * (2 * G * per),
        "cluster_id": np.r_[cl, cl],
        "value": np.r_[eff + rng.normal(0, 1, G * per), eff + 0.3 + rng.normal(0, 1, G * per)],
    }
    c = run(build_plan(cfg({"type": "clustered"})), build_table(cols)).comparisons[0]
    assert c.primary.method == "cluster_robust_diff"
    assert c.primary.df == G - 1
    assert len(c.cross_checks) == 1
