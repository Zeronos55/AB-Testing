"""Helpers of case_studies/05_llm_eval/run.py: the synthetic eval, rows, and the --data path."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from isbetter.adapters.llm_eval import ItemScore, read_item_scores, write_item_scores
from isbetter.core.config import load_config
from isbetter.core.plan import build_plan
from isbetter.methods.cluster import paired_cluster_robust

DIR = Path(__file__).resolve().parents[2] / "case_studies" / "05_llm_eval"
spec = importlib.util.spec_from_file_location("llm_eval_case_study", DIR / "run.py")
cs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cs
spec.loader.exec_module(cs)


def test_config_routes_to_the_paired_cluster_robust_method():
    plan = build_plan(load_config(DIR / "config.yaml"))
    assert plan.primary == "paired_cluster_robust"
    assert plan.cross_checks == ["paired_cluster_bootstrap"]


def test_synthetic_eval_shape_and_expected_gap_is_zero_without_a_gap():
    a, b, passage = cs.synthetic_eval(1, 0.0)
    assert len(a) == len(b) == len(passage) == cs.N_PASSAGES * cs.PER_PASSAGE
    assert set(np.unique(a)) <= {0.0, 1.0} and len(set(passage)) == cs.N_PASSAGES
    diffs = [np.subtract(*cs.synthetic_eval(s, 0.0)[1::-1]).mean() for s in range(300)]
    assert abs(np.mean(diffs)) < 0.01  # no systematic difference by construction
    gains = [np.subtract(*cs.synthetic_eval(s, 0.06)[1::-1]).mean() for s in range(300)]
    assert np.mean(gains) == pytest.approx(0.06, abs=0.01)


def test_synthetic_eval_is_reproducible_per_seed():
    first, again = cs.synthetic_eval(5, 0.06), cs.synthetic_eval(5, 0.06)
    assert all(np.array_equal(x, y) for x, y in zip(first, again, strict=True))


def test_comparison_rows_match_the_methods(tmp_path):
    a, b, passage = cs.synthetic_eval(2, 0.06)
    rows = {r.name: r for r in cs.comparison_rows(a, b, passage, np.random.default_rng(0))}
    cr1 = paired_cluster_robust(b, a, passage)
    r = rows["paired + clustered (CR1)"]
    assert (r.estimate, r.low, r.high, r.p) == (cr1.estimate, cr1.ci_low, cr1.ci_high, cr1.p_value)
    assert rows["unpaired normal (Welch)"].estimate == pytest.approx(b.mean() - a.mean())


def test_clustering_widens_the_paired_interval():
    a, b, passage = cs.synthetic_eval(3, 0.0)
    rows = {r.name: r for r in cs.comparison_rows(a, b, passage, np.random.default_rng(0))}
    width = lambda r: r.high - r.low  # noqa: E731
    assert width(rows["paired + clustered (CR1)"]) > 0.95 * width(rows["paired normal (paired t)"])


def test_first_fooled_seed_is_a_world_with_no_difference_that_the_paired_t_flags():
    from isbetter.methods.parametric import paired_t

    seed, scanned = cs.first_fooled_seed(20261010)
    a, b, _ = cs.synthetic_eval(seed, 0.0)
    assert paired_t(b, a).p_value < 0.05 and b.mean() > a.mean() and scanned >= 1


def test_records_roundtrip_through_the_adapter(tmp_path):
    a, b, passage = cs.synthetic_eval(4, 0.0)
    path = tmp_path / "items.jsonl"
    write_item_scores(path, cs.records(a, b, passage))
    plan = build_plan(load_config(DIR / "config.yaml"))
    table = read_item_scores(path, plan.dependence, metric=plan.metric)
    a2, b2, p2 = cs.arrays(table, cs.BASE, cs.CAND, plan.metric)
    assert sorted(a2) == sorted(a) and sorted(b2) == sorted(b) and len(set(p2)) == cs.N_PASSAGES


def test_data_path_runs_on_a_tiny_file(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    recs = []
    for i in range(24):
        cluster = f"p{i // 4}"
        base = float(rng.random() < 0.6)
        recs.append(ItemScore("m1", f"q{i}", base, cluster))
        recs.append(ItemScore("m2", f"q{i}", float(base or rng.random() < 0.3), cluster))
    path = tmp_path / "tiny.jsonl"
    write_item_scores(path, recs)
    monkeypatch.setattr(cs, "HERE", tmp_path)
    (tmp_path / "config.yaml").write_text((DIR / "config.yaml").read_text())
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--data", str(path), "--baseline", "m1", "--candidate", "m2"]
    )
    cs.main()
    text = (tmp_path / "REPORT.md").read_text()
    assert "Real data were used." in text and "paired_cluster_robust" in text
    assert "Your data" in text
