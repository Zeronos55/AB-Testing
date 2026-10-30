import numpy as np
import pytest

from isbetter.adapters.llm_eval import ItemScore, item_scores_table
from isbetter.core.config import parse_config
from isbetter.core.dependence import Clustered, dependence_from_dict, dependence_to_dict
from isbetter.core.plan import UnsupportedPlan, build_plan, route
from isbetter.core.result import Result
from isbetter.core.runner import run
from isbetter.core.schema import SchemaError


def cfg(dep, kind="binary", min_effect=0.0):
    return parse_config(
        {
            "experiment": "llm",
            "data": "d.jsonl",
            "adapter": "llm_eval",
            "variants": {"baseline": "base", "candidates": ["cand"]},
            "metric": {"name": "acc", "kind": kind},
            "dependence": dep,
            "decision": {"min_effect": min_effect},
            "seed": 3,
        }
    )


def scores(rng, n=60, per=3, p_base=0.6, p_gain=0.25, continuous=False):
    recs = []
    for i in range(n):
        passage = f"p{i // per}"
        if continuous:
            base = float(rng.normal(0.5, 0.2))
            cand = base + 0.1 + float(rng.normal(0, 0.1))
        else:
            base = float(rng.random() < p_base)
            cand = float(base or rng.random() < p_gain)
        recs.append(ItemScore("base", f"q{i}", base, passage))
        recs.append(ItemScore("cand", f"q{i}", cand, passage))
    return recs


def test_paired_binary_routes_to_score_interval_with_bayesian_cross_check():
    plan = build_plan(cfg({"type": "paired"}))
    assert plan.primary == "paired_binary_score"
    assert plan.cross_checks == ["paired_beta_posterior"]
    table = item_scores_table(scores(np.random.default_rng(0)), "acc", plan.dependence)
    res = run(plan, table)
    c = res.comparisons[0]
    assert c.primary.method == "paired_binary_score"
    assert [m.method for m in c.cross_checks] == ["paired_beta_posterior"]
    assert c.oriented_ci_low <= c.oriented_estimate <= c.oriented_ci_high
    assert Result.from_json(res.to_json()) == res


def test_paired_continuous_is_unchanged():
    plan = build_plan(cfg({"type": "paired"}, kind="continuous"))
    assert plan.primary == "paired_t"


def test_paired_clustered_route_runs_and_roundtrips_through_json():
    plan = build_plan(cfg({"type": "clustered", "paired": True}, kind="continuous"))
    assert plan.primary == "paired_cluster_robust"
    assert plan.cross_checks == ["paired_cluster_bootstrap"]
    table = item_scores_table(
        scores(np.random.default_rng(1), n=90, continuous=True), "acc", plan.dependence
    )
    res = run(plan, table)
    c = res.comparisons[0]
    assert c.primary.method == "paired_cluster_robust" and c.n_pairs == 90
    assert c.primary.df == 29  # 30 passages
    assert c.oriented_ci_low > 0  # candidate gains ~0.1 with sd 0.1
    assert Result.from_json(res.to_json()) == res
    assert "paired': True" in plan.pretty()


def test_unpaired_clustered_route_is_unchanged_and_old_plans_load():
    assert route(Clustered())[0] == "cluster_robust_diff"
    assert dependence_from_dict({"type": "clustered", "cluster_col": "g"}) == Clustered("g")
    assert dependence_to_dict(Clustered("g", True))["paired"] is True


def test_cv_overlap_binary_still_refused():
    with pytest.raises(UnsupportedPlan, match="cv_overlap"):
        build_plan(
            cfg(
                {"type": "cv_overlap", "n_train": 90, "n_test": 10, "repeats": 1, "folds": 10},
            )
        )


def test_paired_clustered_schema_checks():
    dep = Clustered(paired=True)
    recs = scores(np.random.default_rng(2))
    item_scores_table(recs, "acc", dep)  # valid
    moved = [r for r in recs]
    moved[0] = ItemScore("base", "q0", moved[0].score, "other")  # q0 in two different clusters
    with pytest.raises(SchemaError, match="more than one cluster"):
        item_scores_table(moved, "acc", dep)
    with pytest.raises(SchemaError, match="missing"):
        item_scores_table(recs[:-1], "acc", dep)


def test_binary_check_rejects_non_binary_scores():
    plan = build_plan(cfg({"type": "paired"}))
    table = item_scores_table(
        scores(np.random.default_rng(0), continuous=True), "acc", plan.dependence
    )
    with pytest.raises(ValueError, match="not binary"):
        run(plan, table)
