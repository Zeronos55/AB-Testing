"""The calibration table script: wording rules, README block handling and no drift."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from isbetter.simulate.calibration import CalibrationResult, mc_band

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "calibration_table", ROOT / "validation/calibration_table.py"
)
ct = importlib.util.module_from_spec(spec)
sys.modules["calibration_table"] = ct  # dataclasses look the module up by name
spec.loader.exec_module(ct)


def sim(rate, alpha=0.05, reps=10_000, measures="false positives", role=ct.RECOMMENDED):
    r = CalibrationResult(rate, reps, alpha, mc_band(alpha, reps))
    return ct.Row("s", "m", role, measures, r)


def exact(value, role=ct.RECOMMENDED):
    return ct.Row("s", "m", role, "coverage", None, value)


@pytest.mark.parametrize(
    ("row", "reading"),
    [
        (sim(0.050), "calibrated"),
        (sim(0.0545), "slightly liberal"),  # just outside the 2-SE band, within 1.5 points
        (sim(0.075), "inflated"),
        (sim(0.568), "inflated"),
        (sim(0.025), "conservative (valid)"),  # below nominal never counts as a failure
        (sim(0.95, 0.95, measures="coverage"), "calibrated"),
        (sim(0.924, 0.95, 1000, "coverage"), "slightly under-covers"),
        (sim(0.879, 0.95, 1000, "coverage"), "under-covers"),
        (sim(0.99, 0.95, 1000, "coverage"), "conservative (valid)"),
        (exact(0.984), "calibrated"),
        (exact(0.939), "slightly under-covers"),
        (exact(0.782), "under-covers"),
    ],
)
def test_reading_is_computed_from_the_numbers(row, reading):
    assert row.reading() == reading


@pytest.fixture(scope="module")
def tiny_rows():
    return ct.build_rows(reps=40)


def test_every_scenario_appears_with_its_naive_foil(tiny_rows):
    by_scenario = {}
    for r in tiny_rows:
        by_scenario.setdefault(r.scenario, set()).add(r.role)
    methods = {r.method for r in tiny_rows}
    expected = [
        "Welch t",
        "paired t",
        "CUPED",
        "SRM chi-square at 0.05",
        "CR1 by user",
        "Nadeau-Bengio",
        "Diebold-Mariano + HLN",
        "confidence sequence",
        "stationary block bootstrap 95% CI",
        "Clopper-Pearson 95% CI",
        "Tango score CI",
    ]
    for needle in expected:
        assert needle in methods, needle
    foil_scenarios = {s for s, roles in by_scenario.items() if ct.FOIL in roles}
    assert len(foil_scenarios) >= 8
    for s in foil_scenarios:
        assert ct.RECOMMENDED in by_scenario[s], s  # a foil is never shown without its fix
    assert all(0.0 <= r.observed <= 1.0 for r in tiny_rows)


def test_tables_have_one_line_per_row_and_equal_columns(tiny_rows):
    for table, cols in ((ct.render_table(tiny_rows), 9), (ct.render_compact(tiny_rows), 6)):
        lines = table.strip().splitlines()
        assert len(lines) == len(tiny_rows) + 2
        assert {ln.count("|") for ln in lines} == {cols + 1}
    page = ct.render_page(tiny_rows, 40)
    assert "40 replications" in page and "expected by chance" in page


def test_update_readme_replaces_only_the_marked_block(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{ct.START}\n\nold\n\n{ct.END}\noutro\n")
    monkeypatch.setattr(ct, "README", readme)
    ct.update_readme("| new |\n")
    assert readme.read_text() == f"intro\n{ct.START}\n\n| new |\n\n{ct.END}\noutro\n"
    readme.write_text("no markers here\n")
    with pytest.raises(SystemExit, match="calibration-table"):
        ct.update_readme("x")


def _cells(line: str) -> list[str]:
    return [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]


def test_readme_table_agrees_with_the_committed_full_table():
    """The README copy is a view of validation/calibration_table.md, never edited by hand."""
    readme = (ROOT / "README.md").read_text()
    block = re.search(re.escape(ct.START) + r"(.*?)" + re.escape(ct.END), readme, re.S)
    assert block, "README.md lost its calibration-table block"
    short = [_cells(ln) for ln in block.group(1).strip().splitlines()[2:]]
    full = [
        _cells(ln)
        for ln in (ROOT / "validation/calibration_table.md").read_text().splitlines()
        if ln.startswith("| ") and not ln.startswith("| Scenario") and "---" not in ln[:6]
    ]
    assert len(short) == len(full) > 20
    for s, f in zip(short, full, strict=True):
        scenario, method, measures, nominal, observed, reading = s
        assert (scenario, measures, nominal, observed, reading) == (f[0], f[3], f[4], f[5], f[8])
        assert method.split(" (")[0] == f[1]


def test_csv_holds_the_same_numbers_as_the_markdown_table(tiny_rows):
    import csv
    import io

    out = list(csv.DictReader(io.StringIO(ct.render_csv(tiny_rows))))
    assert tuple(out[0]) == ct.CSV_COLUMNS and len(out) == len(tiny_rows)
    md = [_cells(ln) for ln in ct.render_table(tiny_rows).strip().splitlines()[2:]]
    for row, src, cells in zip(out, tiny_rows, md, strict=True):
        assert (row["scenario"], row["method"], row["role"], row["measures"]) == tuple(cells[:4])
        assert float(row["nominal"]) == src.nominal and float(row["observed"]) == src.observed
        assert ct._pct(float(row["observed"])) == cells[5] and row["reading"] == cells[8]
        if src.result is None:
            assert (row["band_low"], row["band_high"], row["reps"]) == ("", "", "")
            assert row["exact"] == "true" and cells[6:8] == ["exact", "exact"]
        else:
            assert (float(row["band_low"]), float(row["band_high"])) == src.result.band
            assert int(row["reps"]) == src.result.reps and row["exact"] == "false"


def test_main_writes_the_csv_only_when_asked(tiny_rows, tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "build_rows", lambda reps, progress=None: tiny_rows)
    md, out = tmp_path / "t.md", tmp_path / "bi" / "calibration.csv"
    monkeypatch.setattr(sys, "argv", ["x", "--reps", "40", "--out", str(md)])
    ct.main()
    assert md.exists() and not out.exists()
    monkeypatch.setattr(sys, "argv", ["x", "--reps", "40", "--out", str(md), "--csv", str(out)])
    ct.main()
    assert out.read_text() == ct.render_csv(tiny_rows)
