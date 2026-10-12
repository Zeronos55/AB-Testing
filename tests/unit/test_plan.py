import pytest

from isbetter.core.config import parse_config
from isbetter.core.plan import AnalysisPlan, UnsupportedPlan, build_plan, route


def cfg(dep):
    return parse_config(
        {
            "experiment": "e",
            "data": "d.csv",
            "variants": {"baseline": "a", "candidates": ["b"]},
            "metric": {"name": "m"},
            "dependence": dep,
        }
    )


def test_paired_routes_to_paired_t():
    plan = build_plan(cfg({"type": "paired"}))
    assert plan.primary == "paired_t"
    assert "sign_flip_permutation" in plan.cross_checks
    assert "paired_t" in plan.pretty()


def test_iid_routes_to_welch():
    assert build_plan(cfg({"type": "iid"})).primary == "welch_t"


def test_unknown_dependence_never_falls_back():
    class Weird:
        kind = "weird"

    with pytest.raises(UnsupportedPlan, match="not implemented"):
        route(Weird())  # type: ignore[arg-type]


def test_timeseries_routes_to_dm_with_block_bootstrap_cross_check():
    plan = build_plan(cfg({"type": "timeseries", "horizon": 3}))
    assert plan.primary == "diebold_mariano_hln"
    assert plan.cross_checks == ["stationary_block_bootstrap"]
    assert "horizon': 3" in plan.pretty()


@pytest.mark.parametrize("kind", ["binary", "ratio"])
def test_timeseries_rejects_non_loss_metric_kinds(kind):
    c = parse_config(
        {
            "experiment": "x",
            "data": "d.csv",
            "variants": {"baseline": "a", "candidates": ["b"]},
            "metric": {
                "name": "m",
                "kind": kind,
                **({"denominator": "n"} if kind == "ratio" else {}),
            },
            "dependence": {"type": "timeseries"},
        }
    )
    with pytest.raises(UnsupportedPlan, match="not routable|neither ratio"):
        build_plan(c)


def test_plan_dict_roundtrip():
    plan = build_plan(cfg({"type": "paired"}))
    assert AnalysisPlan.from_dict(plan.to_dict()) == plan


def test_cv_and_cluster_routes():
    cv = build_plan(
        cfg({"type": "cv_overlap", "n_train": 9, "n_test": 1, "repeats": 1, "folds": 5})
    )
    assert cv.primary == "nadeau_bengio"
    assert build_plan(cfg({"type": "clustered"})).primary == "cluster_robust_diff"
