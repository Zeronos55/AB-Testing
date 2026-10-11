"""Command line interface: ``isbetter run config.yaml``."""

from __future__ import annotations

from pathlib import Path

import typer

from isbetter.adapters.csv_long import read_csv
from isbetter.adapters.load import load_data, resolve_data_path
from isbetter.core.config import Config, ConfigError, load_config
from isbetter.core.experiment import build_experiment_plan, run_experiment
from isbetter.core.plan import UnsupportedPlan, build_plan
from isbetter.core.runner import run as run_plan
from isbetter.core.schema import SchemaError
from isbetter.report.markdown import render_experiment_markdown, render_markdown

app = typer.Typer(add_completion=False, help="Dependence-aware A vs B comparison.")


@app.callback()
def _main() -> None:
    """isbetter: is A really better than B?"""


def _run_experiment(cfg: Config, config: Path, out: Path, json_out: Path | None) -> None:
    eplan = build_experiment_plan(cfg)
    typer.echo(eplan.pretty())  # print the plan before running
    data = load_data(cfg, eplan.plans, config.parent)
    result = run_experiment(eplan, data)
    out.write_text(render_experiment_markdown(result))
    if json_out:
        json_out.write_text(result.to_json())
    if result.srm is not None:
        state = "FAILED" if result.srm.flagged else "passed"
        typer.echo(f"SRM check: {state} (p = {result.srm.p_value:.3g})")
    for d in result.decisions:
        typer.echo(
            f"{d.candidate} vs {eplan.baseline}: {'SHIP' if d.ship else 'DO NOT SHIP'}: {d.reason}"
        )
    typer.echo(f"report written to {out}")


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, help="YAML config"),
    out: Path = typer.Option(Path("REPORT.md"), help="Markdown report path"),
    json_out: Path | None = typer.Option(None, help="Also write the Result as JSON"),
) -> None:
    """Run the analysis described by CONFIG and write a report.

    Configs with ``adapter: ab_table`` or a ``metrics`` list run as A/B experiments: automatic
    sample-ratio check, multiplicity across success metrics, guardrails and a ship decision.
    """
    try:
        cfg = load_config(config)
        if cfg.is_experiment:
            _run_experiment(cfg, config, out, json_out)
            return
        plan = build_plan(cfg)
        typer.echo(plan.pretty())  # print the plan before running
        table = read_csv(resolve_data_path(cfg, config.parent), plan.dependence)
        result = run_plan(plan, table)
    except (ConfigError, SchemaError, UnsupportedPlan, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    out.write_text(render_markdown(result))
    if json_out:
        json_out.write_text(result.to_json())
    for c in result.comparisons:
        typer.echo(f"{c.candidate} vs {c.baseline}: {c.verdict.label}")
    typer.echo(f"report written to {out}")


if __name__ == "__main__":
    app()
