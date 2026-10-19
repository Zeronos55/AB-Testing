"""HTML report: same facts as the Markdown report, self-contained, well-formed, escaped."""

import math
import shutil
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from isbetter.adapters.load import load_data
from isbetter.cli import app
from isbetter.core.config import parse_config
from isbetter.core.experiment import build_experiment_plan, run_experiment
from isbetter.report.html import interval_svg, render_experiment_html
from isbetter.simulate.ab_data import hillstrom_like, write_csv

EX = Path(__file__).resolve().parents[2] / "examples"
VOID = {"meta", "br", "hr", "img", "input", "link", "line", "rect", "circle"}


class _Balance(HTMLParser):
    """Checks that every non-void tag is closed in order."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.tags: set[str] = set()
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.add(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with stack {self.stack[-3:]}")
        else:
            self.stack.pop()


def assert_well_formed(text: str) -> _Balance:
    p = _Balance()
    p.feed(text)
    assert not p.errors, p.errors
    assert not p.stack, f"unclosed: {p.stack}"
    assert text.lstrip().startswith("<!doctype html>")
    return p


def assert_self_contained(text: str) -> None:
    for needle in ("<script", "http://", "https://", "<link", "@import", "src="):
        assert needle not in text, needle


def stage(tmp_path: Path, name: str, fmt: str) -> Path:
    """Copy an example config (and its data) into tmp_path with a chosen report format."""
    cfg = tmp_path / f"{name}.yaml"
    text = (EX / f"{name}.yaml").read_text()
    text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("report:"))
    cfg.write_text(text + f"\nreport: {{format: {fmt}}}\n")
    shutil.copy(EX / f"{name}.csv", tmp_path / f"{name}.csv")
    return cfg


def test_cli_writes_both_reports_with_the_same_numbers(tmp_path):
    out = tmp_path / "REPORT.md"
    res = CliRunner().invoke(app, ["run", str(EX / "toy_paired.yaml"), "--out", str(out)])
    assert res.exit_code == 0, res.output
    page = out.with_suffix(".html")
    assert str(page) in res.output
    md, html = out.read_text(), page.read_text()
    assert_well_formed(html)
    assert_self_contained(html)
    for needle in ("Why this method", "paired_t", "95.0% CI", "Analysis plan"):
        assert needle in html
    # every number in the Markdown table appears in the HTML table
    rows = [ln for ln in md.splitlines() if ln.startswith("| paired_t")]
    assert rows
    for cell in rows[0].strip("|").split("|"):
        cell = cell.strip()
        if cell and cell != "paired_t":
            assert cell.replace("[", "").replace("]", "").split(",")[0].strip() in html


def test_html_only_format_does_not_write_markdown(tmp_path):
    cfg = stage(tmp_path, "toy_paired", "[html]")
    out = tmp_path / "R.md"
    res = CliRunner().invoke(
        app, ["run", str(cfg), "--out", str(out), "--html-out", str(tmp_path / "x.html")]
    )
    assert res.exit_code == 0, res.output
    assert not out.exists()
    assert_well_formed((tmp_path / "x.html").read_text())


def test_default_format_stays_markdown_only(tmp_path):
    cfg = stage(tmp_path, "toy_paired", "[markdown]")
    out = tmp_path / "R.md"
    assert CliRunner().invoke(app, ["run", str(cfg), "--out", str(out)]).exit_code == 0
    assert out.exists() and not out.with_suffix(".html").exists()


@pytest.mark.parametrize("name", ["toy_paired", "toy_forecast", "toy_strategies", "toy_stream"])
def test_every_example_renders_to_well_formed_html(tmp_path, name):
    cfg = stage(tmp_path, name, "[html]")
    out = tmp_path / "R.md"
    res = CliRunner().invoke(app, ["run", str(cfg), "--out", str(out)])
    assert res.exit_code == 0, res.output
    text = out.with_suffix(".html").read_text()
    p = assert_well_formed(text)
    assert "svg" in p.tags
    assert_self_contained(text)


def test_sections_specific_to_each_route_appear(tmp_path):
    for name, needle in (
        ("toy_strategies", "Overfitting guards"),
        ("toy_stream", "Reading this result while the data are still coming in"),
        ("toy_forecast", "Block length"),
    ):
        cfg = stage(tmp_path, name, "[html]")
        out = tmp_path / f"{name}.md"
        assert CliRunner().invoke(app, ["run", str(cfg), "--out", str(out)]).exit_code == 0
        assert needle in out.with_suffix(".html").read_text(), name


def test_experiment_report_has_srm_decision_and_every_metric(tmp_path):
    arms = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
    write_csv(tmp_path / "h.csv", hillstrom_like(np.random.default_rng(3), counts=(2500,) * 3))
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
    result = run_experiment(eplan, load_data(cfg, eplan.plans, tmp_path))
    text = render_experiment_html(result)
    assert_well_formed(text)
    assert_self_contained(text)
    for needle in ("Sample-ratio check", "Decision", "Results by metric", "visit", "spend"):
        assert needle in text
    assert ("SHIP" in text) and text.count("<svg") == len(result.results[0].comparisons) * 2


def test_user_text_is_escaped(tmp_path):
    cfg = stage(tmp_path, "toy_paired", "[html]")
    cfg.write_text(cfg.read_text().replace("experiment: toy_paired", "experiment: '<b>x</b> & y'"))
    out = tmp_path / "R.md"
    assert CliRunner().invoke(app, ["run", str(cfg), "--out", str(out)]).exit_code == 0
    text = out.with_suffix(".html").read_text()
    assert "<b>x</b>" not in text and "&lt;b&gt;x&lt;/b&gt; &amp; y" in text


def test_interval_svg_geometry():
    svg = interval_svg(0.5, 0.2, 0.8, 0.1)
    assert svg.startswith("<svg") and svg.count('class="ci"') == 3 and 'class="band"' in svg
    assert 'class="band"' not in interval_svg(0.5, 0.2, 0.8, 0.0)  # no band without a delta


def test_interval_svg_handles_infinite_and_nan_limits():
    open_ended = interval_svg(0.5, 0.2, math.inf, 0.1)
    drawing = lambda svg: svg.split(">", 1)[1]  # noqa: E731  (skip the aria-label text)
    assert "inf" not in drawing(open_ended) and "nan" not in drawing(open_ended)
    assert interval_svg(math.nan, 0.0, 1.0, 0.1) == ""
    assert "nan" not in drawing(interval_svg(0.5, math.nan, math.nan, 0.1))


def test_standalone_svg_carries_its_own_namespace_and_style():
    svg = interval_svg(0.5, 0.2, 0.8, 0.1, standalone=True)
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg and "<style>" in svg
    assert "<style>" not in interval_svg(0.5, 0.2, 0.8, 0.1)  # the report's CSS styles it instead
