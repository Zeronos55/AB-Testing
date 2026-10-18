"""Saved-result viewer: loading, row shaping, the Streamlit page (via AppTest) and the CLI."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from isbetter.adapters.load import load_data
from isbetter.cli import app
from isbetter.core.config import load_config, parse_config
from isbetter.core.experiment import build_experiment_plan, run_experiment
from isbetter.core.plan import build_plan
from isbetter.core.runner import run
from isbetter.simulate.ab_data import hillstrom_like, write_csv
from isbetter.viewer import model
from isbetter.viewer.model import ViewerError, load_saved

EX = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="module")
def paired_json() -> str:
    cfg = load_config(EX / "toy_paired.yaml")
    plan = build_plan(cfg)
    return run(plan, load_data(cfg, [plan], EX)).to_json()


@pytest.fixture(scope="module")
def experiment_json(tmp_path_factory) -> str:
    tmp = tmp_path_factory.mktemp("exp")
    arms = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
    write_csv(tmp / "h.csv", hillstrom_like(np.random.default_rng(3), counts=(2500,) * 3))
    cfg = parse_config(
        {
            "experiment": "email_test",
            "adapter": "ab_table",
            "data": "h.csv",
            "table": {"variant_col": "segment", "mode": "unit"},
            "variants": {"baseline": arms[0], "candidates": arms[1:]},
            "metrics": [
                {"name": "visit", "kind": "binary", "min_effect": 0.01},
                {"name": "spend", "role": "guardrail", "min_effect": 0.5},
            ],
            "dependence": {"type": "iid"},
            "decision": {"alpha": 0.05, "correction": "holm"},
            "seed": 5,
        }
    )
    eplan = build_experiment_plan(cfg)
    return run_experiment(eplan, load_data(cfg, eplan.plans, tmp)).to_json()


# ---------------------------------------------------------------- model


def test_loads_both_kinds_and_round_trips(paired_json, experiment_json):
    single, exp = load_saved(paired_json), load_saved(experiment_json)
    assert single.to_json() == paired_json
    assert exp.to_json() == experiment_json


def test_summary_has_one_row_per_comparison_with_plain_english_verdicts(paired_json):
    rows = model.summary_rows(load_saved(paired_json))
    assert len(rows) == 1 and rows[0]["verdict"] == "candidate is better (practically)"
    assert rows[0]["comparison"] == "candidate vs baseline — score"


def test_experiment_entries_cover_every_metric_and_flag_guardrails(experiment_json):
    exp = load_saved(experiment_json)
    items = model.entries(exp)
    assert len(items) == 4  # 2 candidates x 2 metrics
    assert {e.role for e in items} == {"success", "guardrail"}
    assert any("(guardrail)" in e.label for e in items)
    assert [r["decision"] for r in model.decision_rows(exp)] and model.srm_rows(exp)
    assert model.check_rows(exp, "Mens E-Mail") and model.check_rows(exp, "nobody") == []


def test_method_rows_list_primary_first(paired_json):
    rows = model.method_rows(model.entries(load_saved(paired_json))[0].comparison)
    assert rows[0]["method"] == "paired_t" and rows[0]["role"] == "primary"
    assert all(r["role"] == "cross-check" for r in rows[1:]) and len(rows) == 3


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('{"hello": 1}', "does not look like"),
        ('{"comparisons": []}', "incomplete"),
    ],
)
def test_bad_files_give_a_readable_error(text, match):
    with pytest.raises(ViewerError, match=match):
        load_saved(text)


def test_model_never_loads_data_adapters():
    code = (
        "import sys, isbetter.viewer.model;"
        "bad = [m for m in sys.modules if m.startswith('isbetter.adapters')"
        " or m in ('polars', 'duckdb', 'sklearn')];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


# ---------------------------------------------------------------- the page

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _page(path: str) -> AppTest:
    def script(path: str) -> None:
        from isbetter.viewer.app import main

        main([path])

    return AppTest.from_function(script, args=(path,), default_timeout=30).run()


def test_page_shows_a_single_result(tmp_path, paired_json):
    f = tmp_path / "r.json"
    f.write_text(paired_json)
    at = _page(str(f))
    assert not at.exception
    assert at.title[0].value == "toy_paired"
    assert any("candidate is better (practically)" in s.value for s in at.success)
    assert len(at.metric) == 3 and at.selectbox[0].label == "Open a comparison"


def test_page_shows_an_experiment_with_decisions_and_srm(tmp_path, experiment_json):
    f = tmp_path / "e.json"
    f.write_text(experiment_json)
    at = _page(str(f))
    assert not at.exception
    assert [s.label for s in at.selectbox][0].startswith("Metrics behind the decision")
    assert any("Sample-ratio check passed" in s.value for s in at.success)
    at.selectbox[1].select_index(3).run()  # a guardrail comparison
    assert not at.exception
    assert any("non-inferiority margin" in c.value for c in at.caption)


def test_page_reports_a_bad_or_missing_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": 1}))
    assert "Cannot open this file" in _page(str(bad)).error[0].value
    assert "Cannot read" in _page(str(tmp_path / "missing.json")).error[0].value


def test_page_without_a_file_asks_for_one():
    def script() -> None:
        from isbetter.viewer.app import main

        main([])

    at = AppTest.from_function(script, default_timeout=30).run()
    assert not at.exception and "Upload a result" in at.info[0].value


# ---------------------------------------------------------------- the CLI


def test_view_command_launches_streamlit_on_the_saved_file(tmp_path, paired_json, monkeypatch):
    f = tmp_path / "r.json"
    f.write_text(paired_json)
    seen = {}
    monkeypatch.setattr(subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 0)
    res = CliRunner().invoke(app, ["view", str(f), "--port", "8600"])
    assert res.exit_code == 0, res.output
    cmd = seen["cmd"]
    assert cmd[1:4] == ["-m", "streamlit", "run"] and cmd[4].endswith("viewer/app.py")
    assert cmd[-2:] == ["--", str(f)] and "8600" in cmd


def test_view_command_explains_a_missing_extra(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    res = CliRunner().invoke(app, ["view"])
    assert res.exit_code == 2 and "isbetter[app]" in res.output
