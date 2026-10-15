"""Sequential monitoring through the router and runner (confidence sequences)."""

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from isbetter.cli import app
from isbetter.core.config import ConfigError, parse_config
from isbetter.core.dependence import Clustered, Iid, Paired
from isbetter.core.plan import UnsupportedPlan, build_plan, route
from isbetter.core.result import Result
from isbetter.core.runner import run
from isbetter.core.schema import SchemaError, build_table
from isbetter.methods.sequential import confidence_sequence_diff, confidence_sequence_mean

EX = Path(__file__).resolve().parents[2] / "examples"


def cfg_dict(dep="iid", **over):
    d = {
        "experiment": "stream",
        "data": "d.csv",
        "variants": {"baseline": "b", "candidates": ["c"]},
        "metric": {"name": "m"},
        "dependence": {"type": dep},
        "decision": {"min_effect": 0.0},
        "sequential": {"n_target": 300},
        "seed": 1,
    }
    d.update(over)
    return d


def iid_table(a, b, shuffle=True, seed=0):
    """Rows of two arms with arrival times 0..n-1, optionally shuffled in the file."""
    rows = [("c", f"c{i}", "m", v, i) for i, v in enumerate(a)]
    rows += [("b", f"b{i}", "m", v, i) for i, v in enumerate(b)]
    if shuffle:
        np.random.default_rng(seed).shuffle(rows)
    return build_table(
        {
            "variant": [r[0] for r in rows],
            "unit_id": [r[1] for r in rows],
            "metric": [r[2] for r in rows],
            "value": [r[3] for r in rows],
            "time": [r[4] for r in rows],
        },
        Iid(),
    )


def paired_table(a, b, shuffle=True):
    rows = [(v, i, "c") for i, v in enumerate(a)] + [(v, i, "b") for i, v in enumerate(b)]
    if shuffle:
        np.random.default_rng(1).shuffle(rows)
    return build_table(
        {
            "variant": [r[2] for r in rows],
            "unit_id": [f"{r[2]}{r[1]}" for r in rows],
            "pair_id": [f"p{r[1]}" for r in rows],
            "metric": ["m"] * len(rows),
            "value": [r[0] for r in rows],
            "time": [r[1] for r in rows],
        },
        Paired(),
    )


def test_routes_for_iid_and_paired():
    assert route(Iid(), "continuous", "none", sequential=True) == (
        "confidence_sequence_normal_mixture",
        ("welch_t",),
    )
    assert route(Paired(), "binary", "none", sequential=True)[1] == ("paired_t",)
    plan = build_plan(parse_config(cfg_dict()))
    assert plan.primary == "confidence_sequence_normal_mixture"
    assert "valid at every look" in plan.pretty()


@pytest.mark.parametrize(
    ("dep", "kind", "vr"),
    [(Clustered(), "continuous", "none"), (Iid(), "ratio", "none"), (Iid(), "continuous", "cuped")],
)
def test_router_refuses_what_it_cannot_monitor(dep, kind, vr):
    with pytest.raises(UnsupportedPlan):
        route(dep, kind, vr, sequential=True)


@pytest.mark.parametrize(
    ("override", "msg"),
    [
        ({"sequential": {}}, "exactly one of `n_target`"),
        ({"sequential": {"n_target": 100, "rho": 5}}, "exactly one of `n_target`"),
        ({"dependence": {"type": "timeseries"}}, "iid and paired only"),
        ({"metric": {"name": "m", "kind": "ratio", "denominator": "d"}}, "continuous and binary"),
        ({"adapter": "ab_table", "table": {"variant_col": "arm"}}, "not A/B experiments"),
    ],
)
def test_config_refuses_unsound_sequential(override, msg):
    with pytest.raises(ConfigError, match=msg):
        parse_config(cfg_dict(**override))


def test_iid_runner_matches_the_method_on_arrival_ordered_data():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.4, 1, 250), rng.normal(0, 1, 250)
    plan = build_plan(parse_config(cfg_dict()))
    res = run(plan, iid_table(a, b))  # rows are shuffled in the table; time restores the order
    cs = confidence_sequence_diff(a, b, 0.05, n_target=300)
    p = res.comparisons[0].primary
    assert (p.estimate, p.ci_low, p.ci_high, p.p_value) == (
        cs.estimate[-1],
        cs.lower[-1],
        cs.upper[-1],
        cs.p_value[-1],
    )
    assert p.diagnostics["rho"] == cs.rho
    assert res.comparisons[0].cross_checks[0].method == "welch_t"
    assert res.comparisons[0].n_pairs == 500


def test_paired_runner_uses_the_difference_stream():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 200)
    cand = base + rng.normal(0.3, 0.5, 200)
    plan = build_plan(parse_config(cfg_dict("paired")))
    res = run(plan, paired_table(cand, base))
    cs = confidence_sequence_mean(cand - base, 0.05, n_target=300)
    p = res.comparisons[0].primary
    assert (p.estimate, p.ci_low, p.ci_high) == (cs.estimate[-1], cs.lower[-1], cs.upper[-1])
    assert res.comparisons[0].cross_checks[0].method == "paired_t"


def test_unequal_arms_use_the_common_prefix_and_say_so():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0.2, 1, 180), rng.normal(0, 1, 150)
    res = run(build_plan(parse_config(cfg_dict())), iid_table(a, b))
    p = res.comparisons[0].primary
    assert p.diagnostics["unused_trailing_observations"] == 30
    assert "30 unmatched trailing" in p.rationale
    cs = confidence_sequence_diff(a[:150], b, 0.05, n_target=300)
    assert p.estimate == cs.estimate[-1]


def test_missing_time_column_is_refused():
    table = build_table(
        {
            "variant": ["c", "b"] * 30,
            "unit_id": range(60),
            "metric": ["m"] * 60,
            "value": np.random.default_rng(0).normal(size=60),
        },
        Iid(),
    )
    with pytest.raises(SchemaError, match="'time' column"):
        run(build_plan(parse_config(cfg_dict())), table)


def test_warns_when_the_fixed_test_would_declare_a_win_the_sequence_does_not():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.25, 1, 300), rng.normal(0, 1, 300)  # seed chosen: p_fixed < 0.05 <= p_cs
    res = run(build_plan(parse_config(cfg_dict())), iid_table(a, b))
    c = res.comparisons[0]
    assert c.cross_checks[0].p_value < 0.05 <= c.primary.p_value
    assert any("fixed-horizon test" in w for w in c.warnings)


def test_n_needed_is_reported_for_an_inconclusive_stream():
    rng = np.random.default_rng(8)
    res = run(
        build_plan(parse_config(cfg_dict(decision={"min_effect": 0.3}))),
        iid_table(rng.normal(0.05, 1, 120), rng.normal(0, 1, 120)),
    )
    v = res.comparisons[0].verdict
    assert v.label == "inconclusive" and v.n_needed and v.n_needed > 240


def test_correction_across_candidates_still_applies():
    rng = np.random.default_rng(6)
    cols = {k: [] for k in ("variant", "unit_id", "metric", "value", "time")}
    for name, shift in (("b", 0.0), ("c1", 0.5), ("c2", 0.0)):
        for i, v in enumerate(rng.normal(shift, 1, 200)):
            for k, x in zip(cols, (name, f"{name}{i}", "m", v, i), strict=True):
                cols[k].append(x)
    cfg = cfg_dict(
        variants={"baseline": "b", "candidates": ["c1", "c2"]}, decision={"correction": "holm"}
    )
    res = run(build_plan(parse_config(cfg)), build_table(cols, Iid()))
    assert all(c.adjusted_p is not None for c in res.comparisons)
    assert res.comparisons[0].alpha_used == pytest.approx(0.05 / 2)


def test_result_round_trips_through_json():
    rng = np.random.default_rng(0)
    res = run(
        build_plan(parse_config(cfg_dict())),
        iid_table(rng.normal(0.3, 1, 150), rng.normal(0, 1, 150)),
    )
    assert Result.from_json(res.to_json()) == res
    assert json.loads(res.to_json())["plan"]["sequential"]["n_target"] == 300


def test_false_positive_rate_at_the_final_look_is_within_alpha():
    plan = build_plan(parse_config(cfg_dict()))
    rng = np.random.default_rng(10)
    hits = 0
    reps = 200
    for _ in range(reps):
        res = run(plan, iid_table(rng.normal(size=150), rng.normal(size=150), shuffle=False))
        p = res.comparisons[0].primary
        hits += int(p.ci_low > 0 or p.ci_high < 0)
    assert hits / reps <= 0.05


def test_cli_end_to_end_on_the_example(tmp_path):
    out = tmp_path / "REPORT.md"
    res = CliRunner().invoke(app, ["run", str(EX / "toy_stream.yaml"), "--out", str(out)])
    assert res.exit_code == 0, res.output
    text = out.read_text()
    for needle in ("confidence_sequence_normal_mixture", "welch_t", "peeking foil", "every sample"):
        assert needle in text
