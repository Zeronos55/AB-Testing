"""The docs site: every method card has its sections, every page and link resolves."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CARDS = sorted(p for p in (DOCS / "methods").glob("*.md") if p.name != "index.md")
SECTIONS = ["When to use it", "What it assumes", "How it fails", "Use it", "Checked against",
            "References"]  # fmt: skip

# pages that the build hook generates from the repository
GENERATED = {"gallery.md", "case-studies.md"} | {
    f"case-studies/{p.name}.md" for p in (ROOT / "case_studies").iterdir() if p.is_dir()
}


def nav_files(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [f for n in node for f in nav_files(n)]
    if isinstance(node, dict):
        return [f for v in node.values() for f in nav_files(v)]
    return []


def exists(rel: str) -> bool:
    return rel in GENERATED or (DOCS / rel).exists()


@pytest.fixture(scope="module")
def mkdocs_cfg():
    text = (ROOT / "mkdocs.yml").read_text()
    return yaml.safe_load(text)


def test_every_card_has_all_sections_in_order():
    assert len(CARDS) >= 20
    for card in CARDS:
        text = card.read_text()
        assert text.startswith("# "), card.name
        positions = [text.find(f"## {s}\n") for s in SECTIONS]
        assert all(p >= 0 for p in positions), f"{card.name} lacks a section: {positions}"
        assert positions == sorted(positions), f"{card.name} sections out of order"
        assert "**Dependence:**" in text or "**Applies to:**" in text, card.name


def test_every_card_is_in_the_nav_and_the_index(mkdocs_cfg):
    nav = set(nav_files(mkdocs_cfg["nav"]))
    index = (DOCS / "methods" / "index.md").read_text()
    for card in CARDS:
        assert f"methods/{card.name}" in nav, f"{card.name} is not in the nav"
        assert f"]({card.name})" in index, f"{card.name} is not in the methods index"


def test_every_nav_entry_exists(mkdocs_cfg):
    for rel in nav_files(mkdocs_cfg["nav"]):
        assert exists(rel), f"nav entry {rel} has no page"


def test_internal_links_resolve():
    link = re.compile(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")
    bad = []
    for page in DOCS.rglob("*.md"):
        for target in link.findall(page.read_text()):
            if target.startswith("http"):
                continue
            resolved = (page.parent / target).resolve()
            rel = str(resolved.relative_to(DOCS)) if DOCS in resolved.parents else target
            if not exists(rel):
                bad.append(f"{page.relative_to(DOCS)} -> {target}")
    assert not bad, bad


def resolve(name: str) -> bool:
    """Import the longest module prefix of a dotted name, then getattr the rest."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for attr in parts[i:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


def test_cards_only_cite_code_that_exists():
    """Dotted names such as `isbetter.methods.forecasting.diebold_mariano` must resolve."""
    cite = re.compile(r"`(isbetter\.[A-Za-z_.]+)`")
    for card in CARDS:
        for name in set(cite.findall(card.read_text())):
            assert resolve(name), f"{card.name} cites {name}, which does not exist"


def test_build_hook_assembles_the_gallery_and_case_studies():
    pytest.importorskip("mkdocs")
    spec = importlib.util.spec_from_file_location("docs_hooks", DOCS / "_hooks.py")
    hooks = importlib.util.module_from_spec(spec)
    sys.modules["docs_hooks"] = hooks
    spec.loader.exec_module(hooks)
    page = hooks.gallery_page()
    scripts = sorted((ROOT / "validation" / "failure_gallery").glob("0*.py"))
    assert len(scripts) == 6
    for s in scripts:
        assert f"## {s.stem.split('_')[0]}. " in page and s.name in page
        assert (s.with_name(s.stem + "_results.md").read_text().strip().splitlines()[0]) in page
        assert f"gallery/figures/{s.stem}.png" in page
    assert page.count("![") == 6
    assert "Run:" not in page  # the command line is for the script, not the page
    cases = hooks.case_studies_page()
    for name in hooks.CASE_TITLES:
        assert f"case-studies/{name}.md" in cases and (ROOT / "case_studies" / name).is_dir()
    assert set(hooks.CASE_TITLES) == {
        p.name for p in (ROOT / "case_studies").iterdir() if p.is_dir()
    }


BLOG = DOCS / "blog" / "failure-gallery.md"
GALLERY_RESULTS = sorted((ROOT / "validation" / "failure_gallery").glob("*_results.md"))


def test_blog_percentages_all_come_from_the_gallery_results():
    """A figure like 31.4% in the write-up must be a gallery result (0.314) rounded to 0.1%."""
    results = {
        f"{float(v) * 100:.1f}%"
        for f in GALLERY_RESULTS
        for v in re.findall(r"(?<![\d.])[01]\.\d{3}(?!\d)", f.read_text())
    }
    quoted = set(re.findall(r"(?<![\d.])\d{1,3}\.\d%", BLOG.read_text()))
    assert len(GALLERY_RESULTS) == 6 and len(quoted) > 30
    assert not quoted - results, f"not in any gallery results table: {sorted(quoted - results)}"


def test_blog_figures_and_method_cards_exist():
    text = BLOG.read_text()
    for fig in re.findall(r"!\[[^\]]*\]\(\.\./gallery/figures/([^)]+)\)", text):
        assert (ROOT / "validation" / "failure_gallery" / "figures" / fig).exists(), fig
    cards = set(re.findall(r"\]\(\.\./methods/([a-z-]+)\.md\)", text))
    assert {"nadeau-bengio", "cluster-robust", "diebold-mariano", "confidence-sequences"} <= cards
    for card in cards:
        assert (DOCS / "methods" / f"{card}.md").exists(), card
