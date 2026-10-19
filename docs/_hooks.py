"""MkDocs hooks: pull the evidence pages from the repository instead of copying them.

The failure gallery page is assembled from each script's docstring, its saved results table and
its figure; the case-study pages are the committed REPORT.md files. Nothing is retyped, so the
site cannot drift from the scripts and reports that produced the numbers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from mkdocs.structure.files import File, Files

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "validation" / "failure_gallery"
CASES = ROOT / "case_studies"
REPO = "https://github.com/Zeronos55/AB-Testing/blob/main"

CASE_TITLES = {
    "01_fraud_classifiers": "1. Fraud classifiers: model comparison under CV overlap",
    "02_ab_hillstrom_criteo": "2. A/B test: email campaign with CUPED, SRM and guardrails",
    "03_forecasting_m5": "3. Forecasting: Diebold-Mariano and the Model Confidence Set",
    "04_trading_edge": "4. Trading: edge or luck?",
    "05_llm_eval": "5. LLM eval: clustered, paired and small-N",
}


def _docstring_parts(script: Path) -> tuple[str, str]:
    """(title, body) of a gallery script: first line without its 'Failure gallery #n:' prefix,
    and the text before the 'Run:' line with single newlines joined into paragraphs."""
    doc = ast.get_docstring(ast.parse(script.read_text())) or script.stem
    first, _, rest = doc.partition("\n")
    title = re.sub(r"^Failure gallery #\w+:\s*", "", first).rstrip(".")
    body = rest.split("Run:")[0].strip()
    paragraphs = [" ".join(p.split("\n")) for p in body.split("\n\n")]
    return title[:1].upper() + title[1:], "\n\n".join(paragraphs)


def gallery_page() -> str:
    out = [
        "# Failure gallery\n",
        "Each entry is a case where a common method gives a wrong answer, shown on simulated data "
        "where the truth is known, next to the method `isbetter` uses instead. The text, tables "
        "and figures below are the output of the scripts in "
        f"[`validation/failure_gallery/`]({REPO}/validation/failure_gallery); "
        "nothing is retyped.\n",
    ]
    for script in sorted(GALLERY.glob("0*.py")):
        title, body = _docstring_parts(script)
        number = script.stem.split("_")[0]
        out.append(f"\n## {number}. {title}\n\n{body}\n")
        results = script.with_name(script.stem + "_results.md")
        if results.exists():
            out.append(f"\n{results.read_text().strip()}\n")
        fig = GALLERY / "figures" / f"{script.stem}.png"
        if fig.exists():
            out.append(f"\n![{title}](gallery/figures/{fig.name})\n")
        out.append(
            f"\nSource: [`{script.name}`]({REPO}/validation/failure_gallery/{script.name})\n"
        )
    return "".join(out)


def case_studies_page() -> str:
    out = [
        "# Case studies\n",
        "Five worked comparisons, each written to the same template: question, data and "
        "dependence structure, the naive analysis and why it misleads, the correct analysis, "
        "the verdict with effect size and `n_needed`, and what would change the conclusion.\n",
        '!!! warning "Synthetic stand-ins"\n    Studies 1 to 5 run on synthetic data generated to '
        "have the structure of the real datasets (credit-card fraud, Hillstrom and Criteo, M5, "
        "daily ETF returns, an LLM eval). Each REPORT says so, and each `run.py` has a path for "
        "the real data that is tested only on a tiny file. The numbers show how the methods "
        "behave, not what the real datasets contain.\n",
    ]
    for name, title in CASE_TITLES.items():
        out.append(f"\n- [{title}](case-studies/{name}.md) ([source]({REPO}/case_studies/{name}))")
    return "".join(out) + "\n"


def _generated(config: Any, uri: str, content: str | None = None, src: Path | None = None) -> File:
    if content is not None:
        return File.generated(config, uri, content=content)
    return File.generated(config, uri, abs_src_path=str(src))


def on_files(files: Files, config: Any) -> Files:
    files.append(_generated(config, "gallery.md", gallery_page()))
    files.append(_generated(config, "case-studies.md", case_studies_page()))
    for fig in sorted((GALLERY / "figures").glob("*.png")):
        files.append(_generated(config, f"gallery/figures/{fig.name}", src=fig))
    for name in CASE_TITLES:
        report = (CASES / name / "REPORT.md").read_text()
        banner = (
            f"[Case studies](../case-studies.md) · [source code and README]({REPO}/case_studies/"
            f"{name}) · synthetic stand-in data\n\n"
        )
        files.append(_generated(config, f"case-studies/{name}.md", banner + report))
    return files
