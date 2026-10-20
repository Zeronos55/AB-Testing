"""``isbetter export-bi``: the CSV tables hold exactly what the saved result JSON holds."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from isbetter.cli import app
from isbetter.core.result import MethodResult
from isbetter.decision.verdict import MIRROR
from isbetter.report.bi import (
    COMPARISON_COLUMNS,
    CROSSCHECK_COLUMNS,
    VERDICT_ORDER_COLUMNS,
    agrees,
    verdict_order_rows,
)
from isbetter.report.markdown import LABEL_TEXT
from isbetter.simulate.ab_data import hillstrom_like, write_csv

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples").glob("*.yaml"))


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_cli(config: Path, json_out: Path, tmp_path: Path):
    res = CliRunner().invoke(
        app, ["run", str(config), "--out", str(tmp_path / "r.md"), "--json-out", str(json_out)]
    )
    assert res.exit_code == 0, res.output
    return res


def export(jsons: list[Path], out_dir: Path):
    res = CliRunner().invoke(app, ["export-bi", *map(str, jsons), "--out-dir", str(out_dir)])
    assert res.exit_code == 0, res.output
    return read(out_dir / "comparisons.csv"), read(out_dir / "crosschecks.csv")


def saved_comparisons(path: Path) -> list[tuple[dict, dict]]:
    """(comparison, plan) dict pairs of a Result or ExperimentResult JSON."""
    d = json.loads(path.read_text())
    results = d["results"] if "results" in d else [d]
    return [(c, r["plan"]) for r in results for c in r["comparisons"]]


def num(x: str) -> float | None:
    return None if x == "" else float(x)


@pytest.mark.parametrize("config", EXAMPLES, ids=lambda p: p.stem)
def test_csv_equals_json_and_verdict_counts_match_run(config, tmp_path):
    out = tmp_path / "res.json"
    run = run_cli(config, out, tmp_path)
    comps, checks = export([out], tmp_path / "bi")

    saved = saved_comparisons(out)
    assert len(comps) == len(saved)
    for row, (c, plan) in zip(comps, saved, strict=True):
        assert row["candidate"] == c["candidate"] and row["metric"] == c["metric"]
        assert row["verdict"] == c["verdict"]["label"]
        assert row["primary_method"] == c["primary"]["method"]
        assert row["primary_rationale"] == c["primary"]["rationale"]
        assert row["dependence"] == plan["dependence"]["type"]
        assert float(row["min_effect"]) == plan["min_effect"]
        for col in ("oriented_estimate", "oriented_ci_low", "oriented_ci_high", "n_pairs"):
            assert float(row[col]) == pytest.approx(c[col], rel=1e-12, abs=0)
        assert num(row["adjusted_p"]) == c["adjusted_p"]
        assert num(row["n_needed"]) == c["verdict"]["n_needed"]
        assert row["warnings"] == " | ".join(c["warnings"])

    # the verdict counts equal those `isbetter run` printed
    printed = [
        ln.rsplit(": ", 1)[1] for ln in run.output.splitlines() if " vs " in ln and ln[0] != " "
    ]
    assert printed == [r["verdict"] for r in comps]

    # every cross-check hangs off an exported comparison
    keys = {r["comparison_key"] for r in comps}
    assert len(keys) == len(comps)
    assert {r["comparison_key"] for r in checks} <= keys
    assert sum(len(c["cross_checks"]) for c, _ in saved) == len(checks)
    assert all(r["agrees"] in ("true", "false") for r in checks)


def test_experiment_result_adds_ship_and_srm(tmp_path):
    data = hillstrom_like(np.random.default_rng(7), counts=(4000, 4000, 4000))
    write_csv(tmp_path / "hillstrom.csv", data)
    text = (ROOT / "case_studies/02_ab_hillstrom_criteo/config.yaml").read_text()
    (tmp_path / "config.yaml").write_text(
        text.replace("data: out/hillstrom.csv", "data: hillstrom.csv")
    )
    out = tmp_path / "exp.json"
    run = run_cli(tmp_path / "config.yaml", out, tmp_path)
    comps, _ = export([out], tmp_path / "bi")

    d = json.loads(out.read_text())
    ship = {x["candidate"]: x["ship"] for x in d["decisions"]}
    assert len(comps) == sum(len(r["comparisons"]) for r in d["results"]) == 6  # 3 metrics x 2 arms
    assert {r["role"] for r in comps} == {"success", "guardrail"}
    for row in comps:
        assert row["ship"] == ("true" if ship[row["candidate"]] else "false")
        assert row["srm_flagged"] == ("true" if d["srm"]["flagged"] else "false")
    # same verdicts as `isbetter run` printed for the decisions
    for cand, flag in ship.items():
        assert f"{cand} vs No E-Mail: {'SHIP' if flag else 'DO NOT SHIP'}" in run.output


def test_two_runs_stay_apart_and_join_on_key(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    run_cli(ROOT / "examples/toy_paired.yaml", a, tmp_path)
    run_cli(ROOT / "examples/toy_paired.yaml", b, tmp_path)
    data = json.loads(b.read_text())
    data["created_at"] = "2030-01-01T00:00:00+00:00"
    b.write_text(json.dumps(data))
    comps, checks = export([a, b], tmp_path / "bi")
    assert len(comps) % 2 == 0
    assert len({r["comparison_key"] for r in comps}) == len(comps)  # the second run is not a dup
    assert {r["comparison_key"] for r in checks} == {r["comparison_key"] for r in comps}


def test_columns_are_stable(tmp_path):
    out = tmp_path / "res.json"
    run_cli(EXAMPLES[0], out, tmp_path)
    export([out], tmp_path / "bi")
    for name, cols in (("comparisons", COMPARISON_COLUMNS), ("crosschecks", CROSSCHECK_COLUMNS)):
        header = (tmp_path / "bi" / f"{name}.csv").read_text().splitlines()[0]
        assert header.split(",") == list(cols)


def test_verdict_order_lists_every_label_once_in_reading_order(tmp_path):
    out = tmp_path / "res.json"
    run_cli(EXAMPLES[0], out, tmp_path)
    export([out], tmp_path / "bi")
    rows = read(tmp_path / "bi" / "verdict_order.csv")
    assert list(rows[0]) == list(VERDICT_ORDER_COLUMNS)
    assert [r["label"] for r in rows] == [r["label"] for r in verdict_order_rows()]
    assert {r["label"] for r in rows} == set(MIRROR)  # no label missing, none invented
    assert [int(r["sort_order"]) for r in rows] == list(range(1, 7))
    assert all(r["text"] == LABEL_TEXT[r["label"]] for r in rows)
    assert all(len(r["colour"]) == 7 and r["colour"][0] == "#" for r in rows)
    assert len({r["colour"] for r in rows}) == 6  # colour is a cue, so no two verdicts share one
    # reading order mirrors: the better a label is for the candidate, the earlier it comes
    order = {r["label"]: int(r["sort_order"]) for r in rows}
    for label, mirror in MIRROR.items():
        if label.startswith("A_"):
            assert order[label] < order[mirror]


def test_cli_prints_the_verdict_counts_the_dashboard_must_reproduce(tmp_path):
    out = tmp_path / "res.json"
    run_cli(EXAMPLES[0], out, tmp_path)
    res = CliRunner().invoke(app, ["export-bi", str(out), "--out-dir", str(tmp_path / "bi")])
    assert res.exit_code == 0, res.output
    comps = read(tmp_path / "bi" / "comparisons.csv")
    printed = {
        ln.split(":")[0].strip(): int(ln.split(":")[1])
        for ln in res.output.splitlines()
        if ln.startswith("  ")
    }
    assert set(printed) == set(MIRROR)
    assert printed == {lab: sum(r["verdict"] == lab for r in comps) for lab in MIRROR}
    assert sum(printed.values()) == len(comps)


def test_bad_input_exits_2(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text('{"hello": 1}')
    res = CliRunner().invoke(app, ["export-bi", str(bad), "--out-dir", str(tmp_path / "bi")])
    assert res.exit_code == 2
    assert "not an isbetter" in res.output


# -- the definition of `agrees` ---------------------------------------------------------------


def method(est, lo=None, hi=None, p=None):
    return MethodResult("m", est, lo, hi, 0.95 if lo is not None else None, p, None, None, 10, "")


@pytest.mark.parametrize(
    ("primary", "cross", "expected"),
    [
        ("A_better_practical", method(0.3, 0.2, 0.4), True),  # clears +0.1 as well
        ("A_better_practical", method(0.2, 0.05, 0.35), False),  # above 0 but reaches into margin
        ("equivalent", method(0.0, -0.05, 0.05), True),
        ("equivalent", method(0.0, -0.3, 0.3), False),  # wider: now inconclusive
        ("inconclusive", method(0.0, -0.3, 0.3), True),
        ("B_better_practical", method(-0.3, -0.4, -0.2), True),
    ],
)
def test_agrees_with_interval_uses_the_verdict_rule(primary, cross, expected):
    assert agrees(primary, cross, True, 0.1, 0.05) is expected


def test_agrees_orients_lower_is_better():
    # raw interval is below 0, so the candidate is better when lower is better
    cross = method(-0.3, -0.4, -0.2)
    assert agrees("A_better_practical", cross, False, 0.1, 0.05)
    assert not agrees("A_better_practical", cross, True, 0.1, 0.05)


def test_agrees_without_interval_compares_the_side_of_zero():
    sig = method(0.3, p=0.001)
    assert agrees("A_better_practical", sig, True, 0.1, 0.05)
    assert agrees("A_better_maybe_trivial", sig, True, 0.1, 0.05)
    assert not agrees("inconclusive", sig, True, 0.1, 0.05)
    assert agrees("inconclusive", method(0.3, p=0.4), True, 0.1, 0.05)
    assert agrees("B_better_practical", method(-0.3, p=0.001), True, 0.1, 0.05)
    assert not agrees("A_better_practical", method(0.3), True, 0.1, 0.05)  # no CI, no p
