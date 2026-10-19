"""docs/own-data.md: every recipe runs on a file with the columns it describes.

The runbook's configs are read out of the page itself (the YAML block after each
``<!-- recipe: NAME -->`` marker), run through the CLI on a small generated file, and the messages
and "wrong dependence" claims it makes are checked against what the code does.
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from isbetter.cli import app
from isbetter.core.config import parse_config

PAGE = Path(__file__).resolve().parents[2] / "docs" / "own-data.md"
TEXT = PAGE.read_text()
RECIPES = dict(re.findall(r"<!-- recipe: (\w+) -->\n```yaml\n(.*?)```", TEXT, re.S))


def _csv(header: list[str], rows: list[list[object]]) -> str:
    return "\n".join([",".join(header), *(",".join(str(v) for v in r) for r in rows)]) + "\n"


def _long(rows: list[tuple]) -> str:
    return _csv(["unit_id", "pair_id", "variant", "metric", "value"], [list(r) for r in rows])


def paired() -> dict[str, str]:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(80):
        u = rng.normal(0.7, 0.15)
        rows.append((f"q{i}", f"q{i}", "baseline", "accuracy", round(u + rng.normal(0, 0.05), 4)))
        rows.append(
            (f"q{i}", f"q{i}", "candidate", "accuracy", round(u + 0.03 + rng.normal(0, 0.05), 4))
        )
    return {"paired_scores.csv": _long(rows)}


def ab() -> dict[str, str]:
    rng = np.random.default_rng(2)
    rows = []
    for i in range(800):
        arm = "treatment" if i % 2 else "control"
        hist = rng.gamma(2, 10)
        conv = int(rng.random() < 0.10 + (0.03 if arm == "treatment" else 0))
        spend = conv * rng.gamma(2, 15) + 0.1 * hist
        sessions = 1 + rng.poisson(3)
        rows.append([f"u{i}", arm, conv, round(spend, 3), round(hist, 3), sessions])
    header = ["user_id", "arm", "converted", "revenue", "revenue_pre", "sessions"]
    return {"experiment.csv": _csv(header, rows)}


def clustered(sessions_per_user: int = 8) -> dict[str, str]:
    rng = np.random.default_rng(3)
    rows = []
    for u in range(120):
        arm = "treatment" if u % 2 else "control"
        user_effect = rng.normal(0, 1)
        for s in range(sessions_per_user):
            y = user_effect + (0.15 if arm == "treatment" else 0) + rng.normal(0, 1)
            rows.append([f"u{u}_s{s}", f"u{u}", arm, round(y, 4)])
    return {"sessions.csv": _csv(["session_id", "user_id", "arm", "time_on_page"], rows)}


def cv() -> dict[str, str]:
    rng = np.random.default_rng(4)
    rows = []
    for r in range(5):
        for f in range(5):
            base = 0.5 + rng.normal(0, 0.02)
            for v, d in (("logreg", 0.0), ("lightgbm", 0.03)):
                value = round(base + d + rng.normal(0, 0.01), 4)
                rows.append([f"r{r}f{f}", f"r{r}f{f}", v, "pr_auc", value, r, f])
    header = ["unit_id", "pair_id", "variant", "metric", "value", "repeat", "fold"]
    return {"cv_scores.csv": _csv(header, rows)}


def forecast() -> dict[str, str]:
    rng = np.random.default_rng(5)
    rows, e = [], 0.0
    for t in range(150):
        e = 0.6 * e + rng.normal()
        for v, d in (("seasonal_naive", 1.0), ("arima", 0.9)):
            value = round(abs(1 + e + rng.normal(0, 0.2)) + d, 4)
            rows.append((f"o{t}", f"o{t}", v, "mae_h3", value, t))
    header = ["unit_id", "pair_id", "variant", "metric", "value", "time"]
    return {"forecast_losses.csv": _csv(header, [list(r) for r in rows])}


def returns() -> dict[str, str]:
    rng = np.random.default_rng(6)
    start = date(2023, 1, 2)
    rows = [
        [(start + timedelta(days=i)).isoformat()]
        + [round(float(rng.normal(m, 0.01)), 6) for m in (0.0004, 0.0008, 0.0)]
        for i in range(400)
    ]
    return {"returns.csv": _csv(["date", "buy_and_hold", "trend_fast", "trend_slow"], rows)}


def llm() -> dict[str, str]:
    rng = np.random.default_rng(7)
    lines = []
    for p in range(20):
        pe = rng.normal(0, 0.8)
        for q in range(4):
            for m, d in (("model_a", 0.0), ("model_b", 0.2)):
                score = int(rng.random() < 1 / (1 + np.exp(-(0.4 + pe + d))))
                rec = {"model": m, "item_id": f"p{p}q{q}", "cluster_id": f"p{p}", "score": score}
                lines.append(json.dumps(rec))
    return {"llm_items.jsonl": "\n".join(lines) + "\n"}


def stream() -> dict[str, str]:
    rng = np.random.default_rng(8)
    rows = []
    for i in range(300):
        for v, d in (("control", 0.0), ("treatment", 0.1)):
            rows.append((f"{v[0]}{i}", v, "revenue", round(rng.exponential(5) + d, 4), i))
    header = ["unit_id", "variant", "metric", "value", "time"]
    return {"stream.csv": _csv(header, [list(r) for r in rows])}


# recipe -> (files it reads, primary method its plan must name)
CASES = {
    "paired": (paired, "paired_t"),
    "ab": (ab, "two_proportion_z"),
    "clustered": (clustered, "cluster_robust_diff"),
    "stats": (ab, "two_proportion_z"),
    "cv": (cv, "nadeau_bengio"),
    "forecast": (forecast, "diebold_mariano_hln"),
    "returns": (returns, "sharpe_block_bootstrap"),
    "llm": (llm, "paired_cluster_robust"),
    "stream": (stream, "confidence_sequence_normal_mixture"),
}


def run(tmp_path: Path, config: str, files: dict[str, str]):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(config)
    out = tmp_path / "result.json"
    res = CliRunner().invoke(app, ["run", str(cfg), "--out", str(tmp_path / "r.md"),
                                   "--json-out", str(out)])  # fmt: skip
    return res, out


def ci_width(tmp_path: Path, config: str, files: dict[str, str]) -> float:
    res, out = run(tmp_path, config, files)
    assert res.exit_code == 0, res.output
    d = json.loads(out.read_text())
    comp = d["comparisons"][0] if "comparisons" in d else d["results"][0]["comparisons"][0]
    return comp["oriented_ci_high"] - comp["oriented_ci_low"]


def test_page_has_a_recipe_for_every_case():
    assert set(RECIPES) == set(CASES)


def test_every_yaml_block_is_a_valid_config():
    blocks = re.findall(r"```yaml\n(.*?)```", TEXT, re.S)
    assert len(blocks) >= len(CASES)
    for block in blocks:
        parse_config(yaml.safe_load(block))


@pytest.mark.parametrize("name", sorted(CASES))
def test_recipe_runs_on_a_file_with_the_documented_columns(tmp_path, name):
    make, primary = CASES[name]
    res, out = run(tmp_path, RECIPES[name], make())
    assert res.exit_code == 0, res.output
    assert primary in res.output  # the plan names the method the page says it routes to
    assert out.exists()


def _drop_line(n: int):
    return lambda t: "".join(ln for i, ln in enumerate(t.splitlines(True)) if i != n)


def _iso_dates(t: str) -> str:
    return re.sub(r",(\d+)\n", r",2024-01-\1\n", t)


def _reverse_rows(t: str) -> str:
    head, *rows = t.splitlines()
    return "\n".join([head, *reversed(rows)]) + "\n"


def _blank_value(t: str) -> str:
    return re.sub(r"[0-9.]+\n", "\n", t, count=1)


NO_EDIT: list[tuple[str, str]] = []

# message fragment listed on the page -> (recipe, config edits, edit to the recipe's data file)
ERRORS = {
    "missing required column(s)": ("paired", NO_EDIT, lambda t: t.replace(",value", ",score", 1)),
    "partner(s), e.g. pair_id=": ("paired", NO_EDIT, _drop_line(3)),
    "duplicate (variant, metric, pair_id) rows": (
        "paired", NO_EDIT, lambda t: t + t.splitlines()[1] + "\n"),
    "no rows for variant": ("paired", [("[candidate]", "[cand]")], None),
    "must be numeric": ("paired", NO_EDIT, _blank_value),
    "convert dates to numbers first": ("forecast", NO_EDIT, _iso_dates),
    "must be strictly increasing within variant": ("forecast", NO_EDIT, _reverse_rows),
    "scores per variant, found": ("cv", [("repeats: 5, folds: 5", "repeats: 10, folds: 10")], None),
    "Extra inputs are not permitted": ("paired", [("decision:", "decison:")], None),
    "contains arm(s) not named in the config": (
        "stats", [("baseline: control", "baseline: ctrl"), ("{control: 0.5", "{ctrl: 0.5")], None),
    "needs a `finance` block": (
        "returns", [("finance: {n_trials: 2, sr_variance: 0.0016}\n", "")], None),
    "Refusing to": ("forecast", [("higher_is_better: false", "kind: binary")], None),
}  # fmt: skip


@pytest.mark.parametrize("fragment", sorted(ERRORS))
def test_listed_message_is_what_the_code_says(tmp_path, fragment):
    recipe, cfg_edits, data_edit = ERRORS[fragment]
    assert fragment in TEXT, "the page does not list this message"
    files = CASES[recipe][0]()
    if data_edit is not None:
        first = next(iter(files))
        files[first] = data_edit(files[first])
    config = RECIPES[recipe]
    for old, new in cfg_edits:
        assert old in config, old
        config = config.replace(old, new)
    res, _ = run(tmp_path, config, files)
    assert res.exit_code == 2, res.output
    assert fragment in res.output


def test_every_message_in_the_table_is_tested():
    table = TEXT.split("## 5.")[1].split("## 6.")[0]
    listed = re.findall(r"^\| ``?([^|]+?)``? \| .+ \|$", table, re.M)
    assert sorted(m for m in listed if m != "Message contains") == sorted(ERRORS)


def test_ignoring_clustering_gives_a_narrower_interval(tmp_path):
    files = clustered()
    right = ci_width(tmp_path, RECIPES["clustered"], files)
    iid = RECIPES["clustered"].replace(
        "dependence: {type: clustered, cluster_col: user_id}", "dependence: {type: iid}"
    )
    wrong = ci_width(tmp_path, iid, files)
    assert wrong < 0.8 * right  # the page says ignoring the cluster makes the interval too narrow


def test_treating_cv_folds_as_paired_gives_a_narrower_interval(tmp_path):
    files = cv()
    right = ci_width(tmp_path, RECIPES["cv"], files)
    paired_cfg = re.sub(r"dependence: \{type: cv_overlap.*\}", "dependence: {type: paired}",
                        RECIPES["cv"])  # fmt: skip
    wrong = ci_width(tmp_path, paired_cfg, files)
    assert wrong < 0.5 * right  # "well under half its honest width"
