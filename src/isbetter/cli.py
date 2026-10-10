"""Command line interface: ``isbetter run config.yaml``."""

from __future__ import annotations

from pathlib import Path

import typer

from isbetter.adapters.csv_long import read_csv
from isbetter.core.config import ConfigError, load_config
from isbetter.core.plan import UnsupportedPlan, build_plan
from isbetter.core.runner import run as run_plan
from isbetter.core.schema import SchemaError
from isbetter.report.markdown import render_markdown

app = typer.Typer(add_completion=False, help="Dependence-aware A vs B comparison.")


@app.callback()
def _main() -> None:
    """isbetter: is A really better than B?"""


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, help="YAML config"),
    out: Path = typer.Option(Path("REPORT.md"), help="Markdown report path"),
    json_out: Path | None = typer.Option(None, help="Also write the Result as JSON"),
) -> None:
    """Run the analysis described by CONFIG and write a report."""
    try:
        cfg = load_config(config)
        plan = build_plan(cfg)
        typer.echo(plan.pretty())  # print the plan before running
        data = Path(cfg.data)
        if not data.is_absolute():
            data = config.parent / data
        table = read_csv(data, plan.dependence)
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
