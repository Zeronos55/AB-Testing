"""Flat CSV tables of saved results, for loading into Power BI (``isbetter export-bi``).

A viewer, not a second place to compute statistics: every number is copied from the saved
``Result`` / ``ExperimentResult`` JSON. The only derived columns are ``agrees`` (below) and
``comparison_key``. Two tables are written:

* ``comparisons``: one row per baseline x candidate x metric, with the primary method;
* ``crosschecks``: one row per secondary method of a comparison, joined on ``comparison_key``.

Estimates and intervals are the ``oriented_*`` ones, so a positive number always means the
candidate is better. Cross-checks are oriented the same way before they are compared.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from isbetter.core.dependence import dependence_to_dict
from isbetter.core.experiment import ExperimentResult
from isbetter.core.plan import AnalysisPlan
from isbetter.core.result import ComparisonResult, MethodResult, Result
from isbetter.decision.verdict import decide

COMPARISON_COLUMNS = (
    "comparison_key",
    "experiment",
    "created_at",
    "package_version",
    "seed",
    "dependence",
    "baseline",
    "candidate",
    "metric",
    "role",
    "higher_is_better",
    "primary_method",
    "primary_rationale",
    "estimate",
    "oriented_estimate",
    "oriented_ci_low",
    "oriented_ci_high",
    "ci_level",
    "p_value",
    "adjusted_p",
    "alpha_used",
    "family_size",
    "n_pairs",
    "min_effect",
    "verdict",
    "verdict_explanation",
    "n_needed",
    "n_needed_note",
    "warnings",
    "ship",
    "srm_flagged",
)

CROSSCHECK_COLUMNS = (
    "comparison_key",
    "method",
    "oriented_estimate",
    "oriented_ci_low",
    "oriented_ci_high",
    "ci_level",
    "p_value",
    "agrees",
)

Row = dict[str, Any]


def comparison_key(
    experiment: str, created_at: str, baseline: str, candidate: str, metric: str
) -> str:
    """Identify one comparison of one run, so two runs of the same experiment stay apart."""
    return "|".join((experiment, created_at, baseline, candidate, metric))


def _orient(m: MethodResult, higher_is_better: bool) -> tuple[float, float | None, float | None]:
    """Estimate and interval of a method with positive = the candidate is better."""
    if higher_is_better:
        return m.estimate, m.ci_low, m.ci_high
    low = None if m.ci_high is None else -m.ci_high
    high = None if m.ci_low is None else -m.ci_low
    return -m.estimate, low, high


def _side(label: str) -> int:
    """+1 candidate better, -1 baseline better, 0 neither, read off a verdict label."""
    if label.startswith("A_better"):
        return 1
    if label.startswith("B_better"):
        return -1
    return 0


def agrees(
    primary_label: str,
    cross: MethodResult,
    higher_is_better: bool,
    margin: float,
    alpha: float,
) -> bool:
    """Does a cross-check land where the primary method did?

    With an interval, the cross-check is put through the same three-way verdict as the primary
    (:func:`isbetter.decision.verdict.decide`) and must give the same label: the same side of 0
    and of +/-margin. Without an interval (a permutation test gives only a p-value) only the
    side of 0 is comparable: the sign of the estimate when ``p < alpha``, else "neither".
    """
    estimate, low, high = _orient(cross, higher_is_better)
    if low is not None and high is not None:
        return decide(low, high, margin).label == primary_label
    if cross.p_value is None:
        return False
    side = (1 if estimate > 0 else -1 if estimate < 0 else 0) if cross.p_value < alpha else 0
    return side == _side(primary_label)


def _is_nan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _rows_for_result(
    result: Result,
    experiment: str,
    created_at: str,
    package_version: str,
    ship: dict[str, bool] | None,
    srm_flagged: bool | None,
) -> tuple[list[Row], list[Row]]:
    plan = result.plan
    comps: list[Row] = []
    checks: list[Row] = []
    for c in result.comparisons:
        key = comparison_key(experiment, created_at, c.baseline, c.candidate, c.metric)
        comps.append(
            _comparison_row(
                c, plan, key, experiment, created_at, package_version, result.seed,
                ship.get(c.candidate) if ship is not None else None, srm_flagged,
            )
        )  # fmt: skip
        for m in c.cross_checks:
            est, low, high = _orient(m, plan.higher_is_better)
            checks.append(
                {
                    "comparison_key": key,
                    "method": m.method,
                    "oriented_estimate": est,
                    "oriented_ci_low": low,
                    "oriented_ci_high": high,
                    "ci_level": m.ci_level,
                    "p_value": m.p_value,
                    "agrees": agrees(
                        c.verdict.label, m, plan.higher_is_better, plan.min_effect, c.alpha_used
                    ),
                }
            )
    return comps, checks


def _comparison_row(
    c: ComparisonResult,
    plan: AnalysisPlan,
    key: str,
    experiment: str,
    created_at: str,
    package_version: str,
    seed: int,
    ship: bool | None,
    srm_flagged: bool | None,
) -> Row:
    return {
        "comparison_key": key,
        "experiment": experiment,
        "created_at": created_at,
        "package_version": package_version,
        "seed": seed,
        "dependence": dependence_to_dict(plan.dependence)["type"],
        "baseline": c.baseline,
        "candidate": c.candidate,
        "metric": c.metric,
        "role": plan.role,
        "higher_is_better": plan.higher_is_better,
        "primary_method": c.primary.method,
        "primary_rationale": c.primary.rationale,
        "estimate": c.primary.estimate,
        "oriented_estimate": c.oriented_estimate,
        "oriented_ci_low": c.oriented_ci_low,
        "oriented_ci_high": c.oriented_ci_high,
        "ci_level": c.primary.ci_level,
        "p_value": c.primary.p_value,
        "adjusted_p": c.adjusted_p,
        "alpha_used": c.alpha_used,
        "family_size": c.family_size,
        "n_pairs": c.n_pairs,
        "min_effect": plan.min_effect,
        "verdict": c.verdict.label,
        "verdict_explanation": c.verdict.explanation,
        "n_needed": c.verdict.n_needed,
        "n_needed_note": c.verdict.n_needed_note,
        "warnings": " | ".join(c.warnings),
        "ship": ship,
        "srm_flagged": srm_flagged,
    }


def tables_from_json(text: str) -> tuple[list[Row], list[Row]]:
    """Comparison and cross-check rows of one saved ``Result`` or ``ExperimentResult`` JSON."""
    shape = json.loads(text)
    if isinstance(shape, dict) and "results" in shape and "decisions" in shape:
        exp = ExperimentResult.from_json(text)
        ship = {d.candidate: d.ship for d in exp.decisions}
        srm = None if exp.srm is None else bool(exp.srm.flagged)
        comps: list[Row] = []
        checks: list[Row] = []
        for r in exp.results:
            c, k = _rows_for_result(
                r, exp.experiment, exp.created_at, exp.package_version, ship, srm
            )
            comps += c
            checks += k
        return comps, checks
    if isinstance(shape, dict) and "comparisons" in shape:
        res = Result.from_json(text)
        return _rows_for_result(
            res, res.experiment, res.created_at, res.package_version, None, None
        )
    raise ValueError("not an isbetter Result or ExperimentResult JSON")


def _cell(v: object) -> object:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if _is_nan(v):
        return "nan"
    return v


def write_table(path: Path, columns: Sequence[str], rows: Iterable[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for row in rows:
            w.writerow([_cell(row[c]) for c in columns])


def export_bi(inputs: Sequence[Path], out_dir: Path) -> tuple[Path, Path]:
    """Write ``comparisons.csv`` and ``crosschecks.csv`` for the saved result files."""
    comps: list[Row] = []
    checks: list[Row] = []
    for p in inputs:
        try:
            c, k = tables_from_json(p.read_text(encoding="utf-8"))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{p}: {exc}") from exc
        comps += c
        checks += k
    out_dir.mkdir(parents=True, exist_ok=True)
    cpath, kpath = out_dir / "comparisons.csv", out_dir / "crosschecks.csv"
    write_table(cpath, COMPARISON_COLUMNS, comps)
    write_table(kpath, CROSSCHECK_COLUMNS, checks)
    return cpath, kpath
