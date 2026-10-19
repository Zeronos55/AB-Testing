"""Render a Result as a single self-contained HTML page with Jinja2.

Same content as the Markdown report, laid out for reading: a verdict badge per comparison, and
an interval figure that shows the confidence interval against the zero line and the
equivalence band (plus or minus the minimum effect of interest). No scripts, fonts or other
network resources: the file can be emailed or archived as is.
"""

from __future__ import annotations

import math
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined
from markupsafe import Markup

from isbetter.core.experiment import ExperimentResult
from isbetter.core.result import ComparisonResult, MethodResult, Result
from isbetter.report.markdown import LABEL_TEXT, _fmt, _pval

# verdict label -> (css class, text). The text is always shown, so colour is never the only cue.
_TONE = {
    "A_better_practical": "good",
    "A_better_maybe_trivial": "good-weak",
    "equivalent": "neutral",
    "inconclusive": "neutral",
    "B_better_maybe_trivial": "bad-weak",
    "B_better_practical": "bad",
}

_W, _H = 560, 74
_PAD = 28


def _finite(*xs: float) -> list[float]:
    return [x for x in xs if math.isfinite(x)]


# the figure's colours when it is used as an image, outside the report's stylesheet; mid-tones
# that read on both light and dark backgrounds
_STANDALONE_STYLE = (
    "<style>.band{fill:#2ea043;fill-opacity:.25}"
    ".zero{stroke:#888;stroke-width:1;stroke-dasharray:4 3}"
    ".ci{stroke:#5b8def;stroke-width:2.5}.est{fill:#5b8def}text{fill:#888;font-size:11px}</style>"
)


def interval_svg(
    estimate: float, low: float, high: float, delta: float, standalone: bool = False
) -> str:
    """SVG: oriented effect with its CI, the zero line and the band +-delta.

    Positive means the candidate wins. A limit that is infinite or NaN is drawn running off
    the axis; if the estimate itself is not finite there is nothing to draw. By default the
    SVG is styled by the report's stylesheet; ``standalone`` embeds its own style (and the
    namespace) so it can be used as an image.
    """
    if not math.isfinite(estimate):
        return ""
    vals = _finite(estimate, low, high, delta, -delta, 0.0)
    lo_v, hi_v = min(vals), max(vals)
    span = (hi_v - lo_v) or 1.0
    lo_v, hi_v = lo_v - 0.08 * span, hi_v + 0.08 * span

    def x(v: float) -> float:
        v = min(max(v, lo_v), hi_v)
        return _PAD + (v - lo_v) / (hi_v - lo_v) * (_W - 2 * _PAD)

    y = 34
    ns = ' xmlns="http://www.w3.org/2000/svg"' if standalone else ""
    parts = [
        f'<svg{ns} class="interval" viewBox="0 0 {_W} {_H}" role="img" '
        f'aria-label="Effect {_fmt(estimate)} with interval {_fmt(low)} to {_fmt(high)}; '
        f'equivalence band plus or minus {_fmt(delta)}">'
    ]
    if standalone:
        parts.append(_STANDALONE_STYLE)
    if delta > 0:
        parts.append(
            f'<rect class="band" x="{x(-delta):.1f}" y="10" width="{x(delta) - x(-delta):.1f}" '
            f'height="48" rx="3"/>'
        )
    parts.append(f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="6" y2="62"/>')
    xl, xh = (
        x(low) if math.isfinite(low) else _PAD - 8,
        x(high) if math.isfinite(high) else _W - _PAD + 8,
    )
    parts.append(f'<line class="ci" x1="{xl:.1f}" x2="{xh:.1f}" y1="{y}" y2="{y}"/>')
    for xe in (xl, xh):
        parts.append(f'<line class="ci" x1="{xe:.1f}" x2="{xe:.1f}" y1="{y - 7}" y2="{y + 7}"/>')
    parts.append(f'<circle class="est" cx="{x(estimate):.1f}" cy="{y}" r="5"/>')
    parts.append(
        f'<text x="{x(0):.1f}" y="{_H - 4}" text-anchor="middle">0</text>'
        f'<text x="{_PAD}" y="{_H - 4}" text-anchor="start">{_fmt(lo_v)}</text>'
        f'<text x="{_W - _PAD}" y="{_H - 4}" text-anchor="end">{_fmt(hi_v)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _method_row(m: MethodResult) -> dict[str, str]:
    ci = "n/a" if m.ci_low is None else f"[{_fmt(m.ci_low)}, {_fmt(m.ci_high)}]"
    return {"method": m.method, "estimate": _fmt(m.estimate), "ci": ci, "p": _pval(m.p_value)}


def _figure(c: ComparisonResult, delta: float) -> Markup:
    # interval_svg builds the markup from numbers only, so it is safe to skip autoescaping
    return Markup(interval_svg(c.oriented_estimate, c.oriented_ci_low, c.oriented_ci_high, delta))


def _env() -> Environment:
    env = Environment(
        loader=PackageLoader("isbetter.report", "templates"),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=True,
    )
    env.filters["pval"] = _pval
    env.filters["fmt"] = _fmt
    return env


def _context(**extra: Any) -> dict[str, Any]:
    return {"label_text": LABEL_TEXT, "tone": _TONE, "row": _method_row, "figure": _figure, **extra}


def render_html(result: Result) -> str:
    tpl = _env().get_template("report.html.j2")
    return tpl.render(**_context(r=result, plan_text=result.plan.pretty()))


def render_experiment_html(exp: ExperimentResult) -> str:
    """Render an A/B experiment: SRM check, ship decisions, then each metric's comparisons."""
    env = _env()
    tpl = env.get_template("experiment.html.j2")
    return tpl.render(
        **_context(
            e=exp,
            baseline=exp.plan.baseline,
            plan_text=exp.plan.pretty(),
            pairs=list(zip(exp.plan.plans, exp.results, strict=True)),
            fmt=_fmt,
        )
    )
