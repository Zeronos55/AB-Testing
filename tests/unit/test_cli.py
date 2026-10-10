from pathlib import Path

import pytest
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


def run_size(*args):
    return CliRunner().invoke(app, ["plan-size", *args])


def test_plan_size_proportions():
    res = run_size("--kind", "proportions", "--baseline", "0.10", "--mde", "0.01")
    assert res.exit_code == 0, res.output
    assert "control   : 14,7" in res.output  # about 14.7k per arm for 10% -> 11%
    assert "achieved 0.80" in res.output


def test_plan_size_means_with_cuped_needs_fewer_units():
    plain = run_size("--kind", "means", "--sd", "10", "--mde", "1")
    cuped = run_size("--kind", "means", "--sd", "10", "--mde", "1", "--rho", "0.7")
    n = lambda r: int(r.output.split("control   :")[1].split()[0].replace(",", ""))  # noqa: E731
    assert plain.exit_code == cuped.exit_code == 0
    assert n(cuped) / n(plain) == pytest.approx(1 - 0.49, rel=0.03)


def test_plan_size_reports_power_and_mde_for_fixed_n():
    res = run_size(
        "--kind", "proportions", "--baseline", "0.1", "--mde", "0.01", "--n-control", "5000"
    )
    assert res.exit_code == 0, res.output
    assert "power to detect" in res.output and "minimum detectable lift" in res.output


@pytest.mark.parametrize(
    "args, msg",
    [
        (["--kind", "proportions", "--mde", "0.01"], "--baseline"),
        (["--kind", "means", "--mde", "1"], "--sd"),
        (["--kind", "banana", "--mde", "1"], "must be"),
        (["--kind", "proportions", "--baseline", "1.5", "--mde", "0.01"], "p_control"),
    ],
)
def test_plan_size_errors_exit_2(args, msg):
    res = run_size(*args)
    assert res.exit_code == 2 and msg in res.output
