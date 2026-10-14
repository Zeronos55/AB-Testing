"""Sharpe-ratio comparisons through the router and runner: DSR, PSR, MinTRL and PBO."""

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from isbetter.adapters.load import load_deflation
from isbetter.adapters.strategy_returns import MissingTrialsError, TrialRegistry, returns_table
from isbetter.cli import app
from isbetter.core.config import ConfigError, parse_config
from isbetter.core.dependence import Iid
from isbetter.core.plan import UnsupportedPlan, build_plan, route
from isbetter.core.result import Result
from isbetter.core.runner import run
from isbetter.core.schema import SchemaError
from isbetter.methods.finance import (
    deflated_sharpe,
    min_track_record,
    probabilistic_sharpe,
    sharpe_block_bootstrap,
)
from isbetter.methods.pbo import pbo_cscv

EX = Path(__file__).resolve().parents[2] / "examples"


class Deflation:
    def __init__(self, n_trials=4, sr_variance=0.0004, source="test"):
        self.n_trials, self.sr_variance, self.source = n_trials, sr_variance, source


def cfg_dict(candidates=("a", "b", "c"), **over):
    d = {
        "experiment": "edge",
        "data": "d.csv",
        "adapter": "strategy_returns",
        "variants": {"baseline": "bh", "candidates": list(candidates)},
        "metric": {"name": "return", "kind": "returns"},
        "dependence": {"type": "timeseries"},
        "decision": {"min_effect": 0.0},
        "finance": {"n_trials": 4, "sr_variance": 0.0004},
        "seed": 3,
    }
    d.update(over)
    return d


def series(seed=0, n=480, edge=0.0):
    rng = np.random.default_rng(seed)
    bh = rng.normal(0.0002, 0.01, n)
    return {
        "bh": bh,
        "a": rng.normal(0.0, 0.01, n),
        "b": rng.normal(0.0, 0.01, n) + edge,
        "c": rng.normal(0.0, 0.01, n),
    }


def test_returns_kind_routes_to_the_sharpe_bootstrap_with_guards():
    plan = build_plan(parse_config(cfg_dict()))
    assert plan.primary == "sharpe_block_bootstrap"
    assert plan.cross_checks == [
        "probabilistic_sharpe",
        "deflated_sharpe",
        "min_track_record_length",
    ]
    assert "DSR from explicit n_trials" in plan.pretty() and "not annualised" in plan.pretty()


@pytest.mark.parametrize("dep", [Iid()])
def test_returns_kind_refuses_other_dependence(dep):
    with pytest.raises(UnsupportedPlan, match="timeseries"):
        route(dep, "returns")


@pytest.mark.parametrize(
    ("override", "msg"),
    [
        ({"finance": None}, "needs a `finance` block"),
        ({"finance": {"sr_variance": 0.1}}, "needs `registry`"),
        ({"finance": {"n_trials": 5}}, "also needs `sr_variance`"),
        ({"finance": {"n_trials": 5, "sr_variance": 0.1, "pbo_blocks": 7}}, "must be even"),
        ({"dependence": {"type": "paired"}}, "needs `dependence: {type: timeseries}`"),
        ({"decision": {"correction": "holm"}}, "must be 'none'"),
    ],
)
def test_config_refuses_incomplete_or_inconsistent_finance(override, msg):
    with pytest.raises(ConfigError, match=msg):
        parse_config(cfg_dict(**override))


def test_finance_block_is_only_for_returns_metrics():
    d = cfg_dict(metric={"name": "m"}, dependence={"type": "timeseries"})
    with pytest.raises(ConfigError, match="only valid with metric kind 'returns'"):
        parse_config(d)


def test_runner_matches_the_methods_called_directly():
    s = series(edge=0.0008)
    plan = build_plan(parse_config(cfg_dict()))
    res = run(plan, returns_table(s, metric="return"), deflation=Deflation())
    b = next(c for c in res.comparisons if c.candidate == "b")
    psr, dsr, mintrl = b.cross_checks
    want_psr = probabilistic_sharpe(s["b"], 0.0, 0.05)
    want_dsr = deflated_sharpe(s["b"], n_trials=4, sr_variance=0.0004, alpha=0.05)
    assert (psr.estimate, psr.p_value) == (want_psr.estimate, want_psr.p_value)
    assert dsr.diagnostics["dsr"] == want_dsr.diagnostics["dsr"]
    assert dsr.diagnostics["source"] == "test"
    assert mintrl.estimate == min_track_record(s["b"], 0.0, 0.05).estimate
    # the primary result is the Sharpe difference versus the baseline, not a DM test
    est = sharpe_block_bootstrap(s["b"], np.random.default_rng(0), benchmark=s["bh"]).estimate
    assert b.primary.estimate == pytest.approx(est)
    assert b.primary.method.endswith("block_bootstrap_sharpe")


def test_pbo_is_computed_over_the_candidates_and_matches_the_method():
    s = series()
    plan = build_plan(parse_config(cfg_dict()))
    res = run(plan, returns_table(s, metric="return"), deflation=Deflation())
    (pbo,) = res.family_checks
    want = pbo_cscv(np.column_stack([s["a"], s["b"], s["c"]]), 16, "sharpe", names=["a", "b", "c"])
    assert pbo.estimate == want.estimate and pbo.diagnostics["n_configs"] == 3


def test_pbo_needs_two_candidates():
    plan = build_plan(parse_config(cfg_dict(candidates=("a",))))
    res = run(plan, returns_table(series(), metric="return"), deflation=Deflation())
    assert res.family_checks == []


def test_unskilled_best_of_many_gets_a_dsr_warning_and_the_verdict_is_untouched():
    s = series(seed=5)
    plan = build_plan(parse_config(cfg_dict()))
    with_guards = run(plan, returns_table(s, metric="return"), deflation=Deflation(50, 0.0009))
    for c in with_guards.comparisons:
        assert c.cross_checks[1].diagnostics["dsr"] < 0.95
        assert any("Deflated Sharpe Ratio" in w for w in c.warnings)
    few = run(plan, returns_table(s, metric="return"), deflation=Deflation(2, 0.0001))
    # the guards change warnings only: the same data and seed give the same verdicts
    assert [c.verdict.label for c in with_guards.comparisons] == [
        c.verdict.label for c in few.comparisons
    ]


def test_pbo_warning_when_selection_does_not_carry_over():
    plan = build_plan(parse_config(cfg_dict()))
    res = run(plan, returns_table(series(seed=2), metric="return"), deflation=Deflation())
    pbo = res.family_checks[0].estimate
    assert bool(res.family_warnings) == (pbo > 0.5)


def test_missing_deflation_inputs_refuse_to_run():
    plan = build_plan(parse_config(cfg_dict()))
    with pytest.raises(ValueError, match="number of variants tried"):
        run(plan, returns_table(series(), metric="return"))


def test_misaligned_periods_are_refused():
    s = series()
    table = returns_table({"bh": s["bh"], "a": s["a"]}, metric="return")
    other = returns_table({"c": s["c"]}, times=np.arange(len(s["c"])) + 5, metric="return")
    from isbetter.core.schema import CanonicalTable

    merged = CanonicalTable(
        {k: np.concatenate([table.columns[k], other.columns[k]]) for k in table.columns}
    )
    plan = build_plan(parse_config(cfg_dict(candidates=("a", "c"))))
    with pytest.raises(SchemaError, match="different periods"):
        run(plan, merged, deflation=Deflation())


def test_result_with_guards_survives_a_json_round_trip():
    plan = build_plan(parse_config(cfg_dict()))
    res = run(plan, returns_table(series(), metric="return"), deflation=Deflation(30, 0.0009))
    again = Result.from_json(res.to_json())
    assert again == res and again.family_checks and again.comparisons[0].warnings


def test_older_result_json_without_the_new_fields_still_loads():
    plan = build_plan(parse_config(cfg_dict()))
    d = json.loads(
        run(plan, returns_table(series(), metric="return"), deflation=Deflation()).to_json()
    )
    d.pop("family_checks"), d.pop("family_warnings")
    for c in d["comparisons"]:
        c.pop("warnings")
    d["plan"].pop("finance")
    old = Result.from_json(json.dumps(d))
    assert old.family_checks == [] and old.plan.finance is None


def test_registry_supplies_the_trial_count_and_variance(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    rng = np.random.default_rng(1)
    for i in range(6):
        reg.log(f"v{i}", rng.normal(0, 0.01, 200), {"i": i})
    cfg = parse_config(cfg_dict(finance={"registry": "trials.jsonl"}))
    got = load_deflation(cfg, tmp_path)
    assert got is not None and got.n_trials == 6 and got.sr_variance == reg.sharpe_variance()
    with pytest.raises(ValueError, match="does not exist"):
        load_deflation(cfg, tmp_path / "elsewhere")
    reg_one = TrialRegistry(tmp_path / "one.jsonl")
    reg_one.log("only", rng.normal(0, 0.01, 200))
    one = parse_config(cfg_dict(finance={"registry": "one.jsonl"}))
    with pytest.raises(MissingTrialsError, match="at least 2"):
        load_deflation(one, tmp_path)


def test_cli_end_to_end_on_the_example(tmp_path):
    out, js = tmp_path / "REPORT.md", tmp_path / "r.json"
    res = CliRunner().invoke(
        app, ["run", str(EX / "toy_strategies.yaml"), "--out", str(out), "--json-out", str(js)]
    )
    assert res.exit_code == 0, res.output
    text = out.read_text()
    for needle in ("Overfitting guards", "deflated_sharpe", "pbo_cscv", "min_track_record_length"):
        assert needle in text
    assert Result.from_json(js.read_text()).family_checks[0].method == "pbo_cscv"


def test_cli_without_trial_information_exits_2(tmp_path):
    cfg = (
        (EX / "toy_strategies.yaml")
        .read_text()
        .replace(
            "finance: {n_trials: 3, sr_variance: 0.0016, pbo_blocks: 8}", "finance: {pbo_blocks: 8}"
        )
    )
    (tmp_path / "c.yaml").write_text(cfg)
    (tmp_path / "toy_strategies.csv").write_text((EX / "toy_strategies.csv").read_text())
    res = CliRunner().invoke(
        app, ["run", str(tmp_path / "c.yaml"), "--out", str(tmp_path / "r.md")]
    )
    assert res.exit_code == 2 and "registry" in res.output


def test_iso_dates_in_a_returns_csv_are_read_as_ordered_times(tmp_path):
    table = returns_table(
        {"x": [0.01, -0.02, 0.0], "y": [0.0, 0.01, 0.02]},
        ["2024-01-02", "2024-01-03", "2024-01-04"],
    )
    assert (np.diff(table.select("x", "return")["time"]) > 0).all()
    with pytest.raises(SchemaError, match="numbers or ISO dates"):
        returns_table({"x": [0.01, -0.02, 0.03]}, ["first", "second", "third"])
