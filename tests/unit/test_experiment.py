"""A/B experiment layer: routing, SRM, multiplicity family, guardrails, both data paths."""

import numpy as np
import pytest
from typer.testing import CliRunner

from isbetter.adapters.load import load_data
from isbetter.cli import app
from isbetter.core.config import ConfigError, parse_config
from isbetter.core.dependence import Clustered, CVOverlap, Iid, Paired
from isbetter.core.experiment import (
    ExperimentResult,
    build_experiment_plan,
    run_experiment,
)
from isbetter.core.plan import UnsupportedPlan, build_plan, route
from isbetter.methods.multiplicity import holm
from isbetter.report.markdown import render_experiment_markdown
from isbetter.simulate.ab_data import hillstrom_like, write_csv

ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]


def make_cfg(
    tmp_path, mode="unit", metrics=None, counts=(2500, 2500, 2500), design=None, seed=3, **kw
):
    data = hillstrom_like(np.random.default_rng(seed), counts=counts)
    write_csv(tmp_path / "h.csv", data)
    raw = {
        "experiment": "t",
        "adapter": "ab_table",
        "data": "h.csv",
        "table": {"variant_col": "segment", "mode": mode},
        "variants": {"baseline": ARMS[0], "candidates": ARMS[1:]},
        "metrics": metrics
        or [
            {"name": "visit", "kind": "binary", "min_effect": 0.01},
            {"name": "conversion", "kind": "binary", "min_effect": 0.002},
            {
                "name": "spend",
                "role": "guardrail",
                "min_effect": 0.5,
                "variance_reduction": "cuped",
                "covariate": "history",
            },
        ],
        "dependence": {"type": "iid"},
        "decision": {"alpha": 0.05, "correction": "holm"},
        "seed": 5,
    }
    if design:
        raw["design"] = design
    raw.update(kw)
    cfg = parse_config(raw)
    eplan = build_experiment_plan(cfg)
    return cfg, eplan, load_data(cfg, eplan.plans, tmp_path)


# --------------------------------------------------------------------- routing


@pytest.mark.parametrize(
    "kind, vr, primary, checks",
    [
        ("continuous", "none", "welch_t", ()),
        ("continuous", "cuped", "cuped", ("welch_t",)),
        ("continuous", "regression", "regression_adjustment", ("welch_t",)),
        ("binary", "none", "two_proportion_z", ("fisher_exact",)),
        ("binary", "cuped", "cuped", ("two_proportion_z",)),
        ("ratio", "none", "delta_method", ("ratio_bootstrap",)),
    ],
)
def test_iid_routing(kind, vr, primary, checks):
    assert route(Iid(), kind, vr) == (primary, checks)


@pytest.mark.parametrize(
    "dep, kind, vr",
    [
        (Paired(), "continuous", "cuped"),
        (Clustered(), "ratio", "none"),
        (CVOverlap(10, 2, 2, 2), "continuous", "regression"),
        (Paired(), "binary", "none"),
        (Iid(), "ratio", "cuped"),
    ],
)
def test_unsupported_combinations_refuse_instead_of_degrading(dep, kind, vr):
    with pytest.raises(UnsupportedPlan):
        route(dep, kind, vr)


def test_clustered_binary_is_routable():
    assert route(Clustered(), "binary", "none")[0] == "cluster_robust_diff"


# ---------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "metric, msg",
    [
        ({"name": "m", "kind": "ratio"}, "denominator"),
        ({"name": "m", "denominator": "z"}, "only valid for kind: ratio"),
        ({"name": "m", "variance_reduction": "cuped"}, "needs `covariate`"),
        ({"name": "m", "covariate": "x"}, "variance_reduction is none"),
        (
            {
                "name": "m",
                "kind": "ratio",
                "denominator": "z",
                "variance_reduction": "cuped",
                "covariate": "x",
            },
            "not supported for ratio",
        ),
        ({"name": "m", "role": "guardrail"}, "non-inferiority margin"),
    ],
)
def test_metric_config_errors_are_readable(metric, msg):
    raw = {
        "experiment": "x",
        "data": "d.csv",
        "variants": {"baseline": "a", "candidates": ["b"]},
        "dependence": {"type": "iid"},
        "metric": metric,
    }
    with pytest.raises(ConfigError, match=msg):
        parse_config(raw)


def test_config_needs_exactly_one_of_metric_or_metrics():
    base = {
        "experiment": "x",
        "data": "d.csv",
        "variants": {"baseline": "a", "candidates": ["b"]},
        "dependence": {"type": "iid"},
    }
    with pytest.raises(ConfigError, match="exactly one"):
        parse_config(base)
    with pytest.raises(ConfigError, match="exactly one"):
        parse_config({**base, "metric": {"name": "m"}, "metrics": [{"name": "n"}]})
    with pytest.raises(ConfigError, match="unique"):
        parse_config({**base, "metrics": [{"name": "m"}, {"name": "m"}]})


def test_ab_table_config_rules_and_numeric_arm_labels():
    base = {
        "experiment": "x",
        "adapter": "ab_table",
        "data": "d.csv",
        "variants": {"baseline": 0, "candidates": [1]},
        "metric": {"name": "m"},
        "dependence": {"type": "iid"},
    }
    with pytest.raises(ConfigError, match="variant_col"):
        parse_config(base)
    cfg = parse_config({**base, "table": {"variant_col": "t"}})
    assert cfg.variants.baseline == "0" and cfg.variants.candidates == ["1"]
    with pytest.raises(ConfigError, match="iid only"):
        parse_config(
            {
                **base,
                "table": {"variant_col": "t", "mode": "stats"},
                "dependence": {"type": "clustered"},
            }
        )
    with pytest.raises(ConfigError, match="expected_split"):
        parse_config(
            {**base, "table": {"variant_col": "t"}, "design": {"expected_split": {"0": 1.0}}}
        )


def test_build_plan_refuses_multi_metric_configs(tmp_path):
    cfg, _, _ = make_cfg(tmp_path)
    with pytest.raises(ValueError, match="3 metrics"):
        build_plan(cfg)


# ------------------------------------------------------------------ experiments


def test_unit_and_stats_paths_agree_exactly(tmp_path):
    out = {}
    for mode in ("unit", "stats"):
        _, eplan, data = make_cfg(tmp_path, mode=mode)
        out[mode] = run_experiment(eplan, data)
    for ru, rs in zip(out["unit"].results, out["stats"].results, strict=True):
        for cu, cs in zip(ru.comparisons, rs.comparisons, strict=True):
            for f in ("estimate", "ci_low", "ci_high", "p_value"):
                assert getattr(cu.primary, f) == pytest.approx(getattr(cs.primary, f), rel=1e-9)
            assert cu.verdict.label == cs.verdict.label
    assert [d.ship for d in out["unit"].decisions] == [d.ship for d in out["stats"].decisions]


def test_srm_runs_automatically_and_blocks_shipping(tmp_path):
    _, eplan, data = make_cfg(tmp_path, counts=(4000, 2000, 2000))  # planted 50/25/25
    res = run_experiment(eplan, data)
    assert res.srm is not None and res.srm.flagged
    assert all(not d.ship and d.srm_failed for d in res.decisions)
    assert "sample-ratio" in res.decisions[0].reason


def test_srm_passes_on_balanced_and_respects_planned_split(tmp_path):
    _, eplan, data = make_cfg(tmp_path)
    assert not run_experiment(eplan, data).srm.flagged
    split = {ARMS[0]: 0.5, ARMS[1]: 0.25, ARMS[2]: 0.25}
    _, eplan2, data2 = make_cfg(
        tmp_path, counts=(4000, 2000, 2000), design={"expected_split": split}
    )
    assert not run_experiment(eplan2, data2).srm.flagged  # same data, correct plan


def test_multiplicity_family_spans_success_metrics_only(tmp_path):
    _, eplan, data = make_cfg(tmp_path)
    res = run_experiment(eplan, data)
    vis, conv, spend = res.results
    raw = np.array([c.primary.p_value for r in (vis, conv) for c in r.comparisons])
    adj = holm(raw)
    got = [c.adjusted_p for r in (vis, conv) for c in r.comparisons]
    assert got == pytest.approx(list(adj))
    assert all(c.family_size == 4 for r in (vis, conv) for c in r.comparisons)
    assert vis.comparisons[0].alpha_used == pytest.approx(0.05 / 4)
    # guardrail: unadjusted, one-sided via the (1 - 2 alpha) interval, family of one
    assert all(c.adjusted_p is None and c.family_size == 1 for c in spend.comparisons)
    assert spend.comparisons[0].alpha_used == pytest.approx(0.10)
    assert spend.comparisons[0].primary.ci_level == pytest.approx(0.90)


def test_cuped_guardrail_reports_unadjusted_comparison(tmp_path):
    _, eplan, data = make_cfg(tmp_path)
    spend = run_experiment(eplan, data).results[2].comparisons[0]
    assert spend.primary.method == "cuped"
    assert spend.cross_checks[0].method == "welch_t"
    assert spend.cross_checks[0].rationale.startswith("Unadjusted comparison")
    assert spend.primary.diagnostics["variance_reduction"] > 0


def test_ratio_metric_and_bootstrap_cross_check_only_on_unit_rows(tmp_path):
    metrics = [{"name": "spend", "kind": "ratio", "denominator": "visit", "min_effect": 1.0}]
    _, ep_u, d_u = make_cfg(tmp_path, mode="unit", metrics=metrics)
    _, ep_s, d_s = make_cfg(tmp_path, mode="stats", metrics=metrics)
    ru, rs = run_experiment(ep_u, d_u), run_experiment(ep_s, d_s)
    cu, cs = ru.results[0].comparisons[0], rs.results[0].comparisons[0]
    assert cu.primary.method == cs.primary.method == "delta_method"
    assert cu.primary.estimate == pytest.approx(cs.primary.estimate, rel=1e-9)
    assert [m.method for m in cu.cross_checks] == ["ratio_bootstrap"]
    assert cs.cross_checks == [] and "ratio_bootstrap" not in rs.results[0].plan.cross_checks


def test_stats_path_missing_covariate_moments_fails_loudly(tmp_path):
    from isbetter.core.schema import SchemaError

    _, eplan, data = make_cfg(tmp_path, mode="stats")
    plain = type(data)(
        {k: type(v)(v.n, v.sum, v.sum_sq) for k, v in data.stats.items()}, data.units
    )
    with pytest.raises(SchemaError, match="lacks covariate"):
        run_experiment(eplan, plain)


def test_experiment_needs_a_success_metric(tmp_path):
    metrics = [{"name": "spend", "role": "guardrail", "min_effect": 0.5}]
    with pytest.raises(ValueError, match="success metric"):
        make_cfg(tmp_path, metrics=metrics)


def test_json_round_trip_and_report(tmp_path):
    _, eplan, data = make_cfg(tmp_path)
    res = run_experiment(eplan, data)
    again = ExperimentResult.from_json(res.to_json())
    assert again.to_dict() == res.to_dict()
    text = render_experiment_markdown(res)
    for needle in (
        "Sample-ratio check",
        "## Decision",
        "SHIP",
        "guardrail",
        "holm across 4 hypotheses",
        "Results by metric",
        "cuped",
    ):
        assert needle in text


def test_clustered_ab_counts_clusters_for_srm(tmp_path):
    rng = np.random.default_rng(0)
    rows = ["arm,user,y"]
    for arm, base in (("a", 0.0), ("b", 0.0)):
        for u in range(40):
            for _ in range(3):
                rows.append(f"{arm},{arm}{u},{base + rng.normal():.4f}")
    (tmp_path / "c.csv").write_text("\n".join(rows) + "\n")
    cfg = parse_config(
        {
            "experiment": "c",
            "adapter": "ab_table",
            "data": "c.csv",
            "table": {"variant_col": "arm", "cluster_col": "user"},
            "variants": {"baseline": "a", "candidates": ["b"]},
            "metrics": [{"name": "y"}],
            "dependence": {"type": "clustered"},
        }
    )
    eplan = build_experiment_plan(cfg)
    res = run_experiment(eplan, load_data(cfg, eplan.plans, tmp_path))
    assert res.srm.observed == {"a": 40, "b": 40}  # users, not the 120 rows per arm
    assert res.results[0].comparisons[0].primary.method == "cluster_robust_diff"


# -------------------------------------------------------------------------- CLI


def write_cfg(tmp_path, mode):
    make_cfg(tmp_path, mode=mode)
    (tmp_path / "c.yaml").write_text(f"""
experiment: cli_{mode}
adapter: ab_table
data: h.csv
table: {{variant_col: segment, mode: {mode}}}
variants: {{baseline: "No E-Mail", candidates: ["Mens E-Mail", "Womens E-Mail"]}}
metrics:
  - {{name: visit, kind: binary, min_effect: 0.01}}
  - {{name: spend, role: guardrail, min_effect: 0.5, variance_reduction: cuped, covariate: history}}
dependence: {{type: iid}}
decision: {{alpha: 0.05, correction: holm}}
seed: 1
""")
    return tmp_path / "c.yaml"


@pytest.mark.parametrize("mode", ["unit", "stats"])
def test_cli_runs_ab_experiment(tmp_path, mode):
    cfg = write_cfg(tmp_path, mode)
    out, js = tmp_path / "R.md", tmp_path / "R.json"
    res = CliRunner().invoke(app, ["run", str(cfg), "--out", str(out), "--json-out", str(js)])
    assert res.exit_code == 0, res.output
    assert "Experiment plan" in res.output and "SRM check: passed" in res.output
    assert "SHIP" in res.output
    assert "Sample-ratio check" in out.read_text()
    assert ExperimentResult.from_json(js.read_text()).srm is not None


def test_cli_bad_arm_labels_exit_2(tmp_path):
    cfg = write_cfg(tmp_path, "stats")
    cfg.write_text(cfg.read_text().replace("Womens E-Mail", "Ghost"))
    res = CliRunner().invoke(app, ["run", str(cfg), "--out", str(tmp_path / "r.md")])
    assert res.exit_code == 2
    assert "not named in the config" in res.output


def test_underflowed_p_values_are_not_printed_as_zero():
    from isbetter.report.markdown import _pval

    assert _pval(0.0) == "<1e-300"
    assert _pval(3.5e-17) == "3.5e-17"
    assert _pval(0.04321) == "0.04321"
    assert _pval(None) == "n/a"
