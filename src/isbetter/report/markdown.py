"""Render a Result as Markdown with Jinja2."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined

from isbetter.core.result import MethodResult, Result

LABEL_TEXT = {
    "A_better_practical": "candidate is better (practically)",
    "A_better_maybe_trivial": "candidate is better, but the effect may be trivial",
    "equivalent": "practically equivalent",
    "inconclusive": "inconclusive",
    "B_better_maybe_trivial": "baseline is better, but the difference may be trivial",
    "B_better_practical": "baseline is better (practically)",
}


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4g}"


def _row(m: MethodResult) -> str:
    ci = "n/a" if m.ci_low is None else f"[{_fmt(m.ci_low)}, {_fmt(m.ci_high)}]"
    return f"| {m.method} | {_fmt(m.estimate)} | {ci} | {_fmt(m.p_value)} |"


def render_markdown(result: Result) -> str:
    env = Environment(
        loader=PackageLoader("isbetter.report", "templates"),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    tpl = env.get_template("report.md.j2")
    return tpl.render(r=result, plan_text=result.plan.pretty(), label_text=LABEL_TEXT, row=_row)
