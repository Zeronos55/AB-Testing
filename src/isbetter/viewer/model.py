"""Load saved results and shape them into plain rows for display. Pure: no Streamlit, no I/O.

Kept separate from the Streamlit script so the logic can be tested without a UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from isbetter.core.experiment import ExperimentResult
from isbetter.core.result import ComparisonResult, Result
from isbetter.report.markdown import LABEL_TEXT, _fmt, _pval

SavedResult = Result | ExperimentResult


class ViewerError(ValueError):
    """The file is not a result saved by ``isbetter run --json-out``."""


def load_saved(text: str) -> SavedResult:
    """Parse a saved ``Result`` or ``ExperimentResult`` (told apart by their keys)."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ViewerError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ViewerError("expected a JSON object saved by `isbetter run --json-out`")
    try:
        if "decisions" in raw and "results" in raw:
            return ExperimentResult.from_json(text)
        if "comparisons" in raw:
            return Result.from_json(text)
    except (KeyError, TypeError) as exc:
        raise ViewerError(f"incomplete or out-of-date result file (problem with {exc})") from exc
    raise ViewerError("this does not look like an isbetter result (no 'comparisons' or 'results')")


@dataclass(frozen=True)
class Entry:
    """One comparison with the sub-result it belongs to (an experiment holds one per metric)."""

    label: str
    comparison: ComparisonResult
    parent: Result
    role: str  # "success" | "guardrail"


def entries(saved: SavedResult) -> list[Entry]:
    if isinstance(saved, Result):
        return [
            Entry(f"{c.candidate} vs {c.baseline} — {c.metric}", c, saved, "success")
            for c in saved.comparisons
        ]
    out = []
    for res in saved.results:
        for c in res.comparisons:
            role = res.plan.role
            tag = f" ({role})" if role != "success" else ""
            out.append(Entry(f"{c.candidate} vs {c.baseline} — {c.metric}{tag}", c, res, role))
    return out


def ci_text(lo: float | None, hi: float | None) -> str:
    return "n/a" if lo is None else f"[{_fmt(lo)}, {_fmt(hi)}]"


def summary_rows(saved: SavedResult) -> list[dict[str, str]]:
    """One row per comparison: the verdict at a glance."""
    rows = []
    for e in entries(saved):
        c = e.comparison
        rows.append(
            {
                "comparison": e.label,
                "verdict": LABEL_TEXT[c.verdict.label],
                "effect": _fmt(c.oriented_estimate),
                "CI": ci_text(c.oriented_ci_low, c.oriented_ci_high),
                "adjusted p": _pval(c.adjusted_p) if c.adjusted_p is not None else "n/a",
                "n": str(c.n_pairs),
            }
        )
    return rows


def method_rows(c: ComparisonResult) -> list[dict[str, str]]:
    """The primary method and its cross-checks, primary first."""
    level = f"{(c.primary.ci_level or 0) * 100:.1f}% CI"
    return [
        {
            "method": m.method,
            "role": "primary" if m is c.primary else "cross-check",
            "estimate": _fmt(m.estimate),
            level: ci_text(m.ci_low, m.ci_high),
            "p-value": _pval(m.p_value),
        }
        for m in (c.primary, *c.cross_checks)
    ]


def decision_rows(exp: ExperimentResult) -> list[dict[str, str]]:
    return [
        {
            "candidate": d.candidate,
            "decision": "SHIP" if d.ship else "DO NOT SHIP",
            "why": d.reason,
        }
        for d in exp.decisions
    ]


def check_rows(exp: ExperimentResult, candidate: str) -> list[dict[str, str]]:
    for d in exp.decisions:
        if d.candidate == candidate:
            return [
                {
                    "metric": k.metric,
                    "role": k.role,
                    "result": k.status,
                    "effect": _fmt(k.oriented_estimate),
                    "CI": ci_text(k.oriented_ci_low, k.oriented_ci_high),
                    "note": k.detail,
                }
                for k in d.checks
            ]
    return []


def srm_rows(exp: ExperimentResult) -> list[dict[str, str]]:
    if exp.srm is None:
        return []
    s = exp.srm
    return [
        {
            "arm": arm,
            "observed": f"{s.observed[arm]:,}",
            "expected": f"{s.expected[arm]:,.1f}",
            "planned share": f"{s.expected_share[arm] * 100:.1f}%",
        }
        for arm in s.observed
    ]
