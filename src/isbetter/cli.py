"""Command line interface: ``isbetter run config.yaml``."""

from __future__ import annotations

from pathlib import Path

import typer

from isbetter.adapters.load import load_data, load_deflation
from isbetter.core.config import Config, ConfigError, load_config
from isbetter.core.experiment import build_experiment_plan, run_experiment
from isbetter.core.plan import UnsupportedPlan, build_plan
from isbetter.core.runner import run as run_plan
from isbetter.core.schema import SchemaError
from isbetter.methods.power import (
    mde_means,
    mde_proportions,
    power_means,
    power_proportions,
    sample_size_means,
    sample_size_proportions,
)
from isbetter.report.bi import export_bi
from isbetter.report.html import render_experiment_html, render_html
from isbetter.report.markdown import render_experiment_markdown, render_markdown

app = typer.Typer(add_completion=False, help="Dependence-aware A vs B comparison.")


@app.callback()
def _main() -> None:
    """isbetter: is A really better than B?"""


def _html_path(out: Path, html_out: Path | None) -> Path:
    return html_out if html_out is not None else out.with_suffix(".html")


def _run_experiment(
    cfg: Config, config: Path, out: Path, html_out: Path | None, json_out: Path | None
) -> None:
    eplan = build_experiment_plan(cfg)
    typer.echo(eplan.pretty())  # print the plan before running
    data = load_data(cfg, eplan.plans, config.parent)
    result = run_experiment(eplan, data)
    written = []
    if "markdown" in cfg.report.format:
        out.write_text(render_experiment_markdown(result))
        written.append(out)
    if "html" in cfg.report.format:
        _html_path(out, html_out).write_text(render_experiment_html(result))
        written.append(_html_path(out, html_out))
    if json_out:
        json_out.write_text(result.to_json())
    if result.srm is not None:
        state = "FAILED" if result.srm.flagged else "passed"
        typer.echo(f"SRM check: {state} (p = {result.srm.p_value:.3g})")
    for d in result.decisions:
        typer.echo(
            f"{d.candidate} vs {eplan.baseline}: {'SHIP' if d.ship else 'DO NOT SHIP'}: {d.reason}"
        )
    typer.echo("report written to " + ", ".join(str(w) for w in written))


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, help="YAML config"),
    out: Path = typer.Option(Path("REPORT.md"), help="Markdown report path"),
    html_out: Path | None = typer.Option(
        None,
        help="HTML report path (default: --out with .html), written when report.format has html",
    ),
    json_out: Path | None = typer.Option(None, help="Also write the Result as JSON"),
) -> None:
    """Run the analysis described by CONFIG and write a report.

    Configs with ``adapter: ab_table`` or a ``metrics`` list run as A/B experiments: automatic
    sample-ratio check, multiplicity across success metrics, guardrails and a ship decision.
    The config's ``report.format`` picks the outputs: ``markdown`` (default) and/or ``html``.
    """
    try:
        cfg = load_config(config)
        if cfg.is_experiment:
            _run_experiment(cfg, config, out, html_out, json_out)
            return
        plan = build_plan(cfg)
        typer.echo(plan.pretty())  # print the plan before running
        table = load_data(cfg, [plan], config.parent)
        result = run_plan(plan, table, deflation=load_deflation(cfg, config.parent))
    except (ConfigError, SchemaError, UnsupportedPlan, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    written = []
    if "markdown" in cfg.report.format:
        out.write_text(render_markdown(result))
        written.append(out)
    if "html" in cfg.report.format:
        _html_path(out, html_out).write_text(render_html(result))
        written.append(_html_path(out, html_out))
    if json_out:
        json_out.write_text(result.to_json())
    for c in result.comparisons:
        typer.echo(f"{c.candidate} vs {c.baseline}: {c.verdict.label}")
    typer.echo("report written to " + ", ".join(str(w) for w in written))


@app.command("export-bi")
def export_bi_cmd(
    results: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Result JSON files written by `run --json-out`"
    ),
    out_dir: Path = typer.Option(Path("bi"), help="Folder for the CSV tables"),
) -> None:
    """Flatten saved results into CSV tables for a Power BI dashboard.

    Reads both single-metric and A/B experiment results. Numbers are copied from the JSON, never
    recomputed; estimates are oriented so that positive means the candidate is better.
    """
    try:
        tables = export_bi(results, out_dir)
    except (ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo("tables written to " + ", ".join(str(p) for p in tables[:3]))
    typer.echo("verdicts (the dashboard's cards must show these counts):")
    for label, n in tables.verdict_counts.items():
        typer.echo(f"  {label}: {n}")


@app.command("plan-size")
def plan_size(
    kind: str = typer.Option(..., help="'proportions' (conversion-style) or 'means'"),
    mde: float = typer.Option(..., help="Absolute difference to detect (a lift, or metric units)"),
    baseline: float | None = typer.Option(None, help="Baseline rate (proportions only), e.g. 0.10"),
    sd: float | None = typer.Option(None, help="Outcome standard deviation (means only)"),
    alpha: float = typer.Option(0.05, help="Significance level"),
    power: float = typer.Option(0.8, help="Target power"),
    ratio: float = typer.Option(1.0, help="Treatment size / control size"),
    one_sided: bool = typer.Option(False, "--one-sided", help="One-sided test"),
    rho: float = typer.Option(0.0, help="Correlation with a CUPED covariate (0 = no CUPED)"),
    method: str = typer.Option("arcsine", help="Proportions: 'arcsine' (statsmodels) or 'pooled'"),
    n_control: int | None = typer.Option(
        None, help="Instead report power and MDE for this many control units"
    ),
) -> None:
    """Plan an A/B test: units needed to detect MDE, or the power / MDE of a fixed size."""
    two_sided = not one_sided
    try:
        if kind == "proportions":
            if baseline is None:
                raise ValueError("--baseline is required for kind=proportions")
            if n_control is not None:
                n_eff = n_control / (1 - rho * rho) if rho else n_control
                got = power_proportions(n_eff, baseline, mde, alpha, ratio, two_sided, method)
                best = mde_proportions(
                    n_control, baseline, alpha, power, ratio, two_sided, method, rho,
                    direction=1.0 if mde > 0 else -1.0,
                )  # fmt: skip
                typer.echo(f"{n_control:,} control units (treatment {ratio:g}x):")
                typer.echo(f"  power to detect {mde:+g} from {baseline:g}: {got:.3f}")
                typer.echo(f"  minimum detectable lift at {power:.0%} power: {best:g}")
                return
            plan = sample_size_proportions(
                baseline, mde, alpha, power, ratio, two_sided, method, rho
            )
        elif kind == "means":
            if sd is None:
                raise ValueError("--sd is required for kind=means")
            if n_control is not None:
                got = power_means(
                    n_control, mde, sd * (1 - rho * rho) ** 0.5, alpha, ratio, two_sided
                )
                best = mde_means(n_control, sd, alpha, power, ratio, two_sided, rho)
                typer.echo(f"{n_control:,} control units (treatment {ratio:g}x):")
                typer.echo(f"  power to detect {mde:+g}: {got:.3f}")
                typer.echo(f"  minimum detectable difference at {power:.0%} power: {best:g}")
                return
            plan = sample_size_means(mde, sd, alpha, power, ratio, two_sided, rho)
        else:
            raise ValueError("--kind must be 'proportions' or 'means'")
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Sample size ({plan.kind}, {plan.method}; {plan.note})")
    typer.echo(f"  control   : {plan.n_control:,}")
    typer.echo(f"  treatment : {plan.n_treatment:,}")
    typer.echo(f"  total     : {plan.n_total:,}")
    sided = "two-sided" if two_sided else "one-sided"
    typer.echo(
        f"  alpha={alpha:g} ({sided}), target power {power:.0%}, achieved {plan.power_achieved:.3f}"
    )


if __name__ == "__main__":
    app()
