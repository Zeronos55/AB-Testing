import pytest

from isbetter.core.config import parse_config
from isbetter.core.plan import AnalysisPlan, UnsupportedPlan, build_plan


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


@pytest.mark.parametrize(
    "dep",
    [
        {"type": "clustered"},
        {"type": "cv_overlap", "n_train": 9, "n_test": 1, "repeats": 1, "folds": 5},
        {"type": "timeseries"},
    ],
)
def test_unsupported_never_falls_back(dep):
    with pytest.raises(UnsupportedPlan, match="not implemented"):
        build_plan(cfg(dep))


def test_plan_dict_roundtrip():
    plan = build_plan(cfg({"type": "paired"}))
    assert AnalysisPlan.from_dict(plan.to_dict()) == plan
