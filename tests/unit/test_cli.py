from pathlib import Path

from typer.testing import CliRunner

from isbetter.cli import app

EX = Path(__file__).resolve().parents[2] / "examples"


def test_cli_end_to_end(tmp_path):
    out = tmp_path / "REPORT.md"
    res = CliRunner().invoke(app, ["run", str(EX / "toy_paired.yaml"), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "Analysis plan" in res.output  # plan printed before results
    text = out.read_text()
    for needle in ("Verdict", "paired_t", "95.0% CI", "Why this method", "p-value"):
        assert needle in text


def test_cli_unsupported_exits_2(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "experiment: x\ndata: d.csv\nvariants: {baseline: a, candidates: [b]}\n"
        "metric: {name: m}\ndependence: {type: timeseries}\n"
    )
    res = CliRunner().invoke(app, ["run", str(cfg), "--out", str(tmp_path / "r.md")])
    assert res.exit_code == 2
    assert "not implemented" in res.output
