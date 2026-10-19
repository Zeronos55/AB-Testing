"""The Markdown docs: every method card has its sections, every page and link resolves."""

import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CARDS = sorted(p for p in (DOCS / "methods").glob("*.md") if p.name != "index.md")
SECTIONS = ["When to use it", "What it assumes", "How it fails", "Use it", "Checked against",
            "References"]  # fmt: skip


def test_every_card_has_all_sections_in_order():
    assert len(CARDS) >= 20
    for card in CARDS:
        text = card.read_text()
        assert text.startswith("# "), card.name
        positions = [text.find(f"## {s}\n") for s in SECTIONS]
        assert all(p >= 0 for p in positions), f"{card.name} lacks a section: {positions}"
        assert positions == sorted(positions), f"{card.name} sections out of order"
        assert "**Dependence:**" in text or "**Applies to:**" in text, card.name


def test_every_card_is_in_the_index():
    index = (DOCS / "methods" / "index.md").read_text()
    for card in CARDS:
        assert f"]({card.name})" in index, f"{card.name} is not in the methods index"


def test_internal_links_resolve():
    link = re.compile(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")
    bad = []
    for page in DOCS.rglob("*.md"):
        for target in link.findall(page.read_text()):
            if target.startswith("http"):
                continue
            resolved = (page.parent / target).resolve()
            if not resolved.exists():
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
