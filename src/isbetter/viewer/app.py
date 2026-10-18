"""Streamlit page: browse a saved result. Start it with ``isbetter view result.json``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from isbetter.core.experiment import ExperimentResult
from isbetter.report.html import _TONE, interval_svg
from isbetter.report.html import render_experiment_html as experiment_html
from isbetter.report.html import render_html as comparison_html
from isbetter.report.markdown import LABEL_TEXT
from isbetter.viewer.model import (
    SavedResult,
    ViewerError,
    check_rows,
    decision_rows,
    entries,
    load_saved,
    method_rows,
    srm_rows,
    summary_rows,
)

_BOX = {
    "good": "success",
    "good-weak": "info",
    "neutral": "info",
    "bad-weak": "warning",
    "bad": "error",
}


def _show(st: Any, saved: SavedResult) -> None:
    is_exp = isinstance(saved, ExperimentResult)
    st.title(saved.experiment)
    st.caption(f"isbetter {saved.package_version} · seed {saved.seed} · {saved.created_at}")

    if isinstance(saved, ExperimentResult):
        if saved.srm is not None:
            box = st.error if saved.srm.flagged else st.success
            box(
                f"Sample-ratio check {'FAILED' if saved.srm.flagged else 'passed'}. "
                f"{saved.srm.rationale}"
            )
            st.dataframe(srm_rows(saved), hide_index=True)
        st.subheader("Decision")
        st.dataframe(decision_rows(saved), hide_index=True)
        who = st.selectbox(
            "Metrics behind the decision for", [d.candidate for d in saved.decisions]
        )
        st.dataframe(check_rows(saved, who), hide_index=True)

    st.subheader("All comparisons")
    st.dataframe(summary_rows(saved), hide_index=True)

    items = entries(saved)
    if not items:
        st.info("This result has no comparisons.")
        return
    pick = st.selectbox(
        "Open a comparison", range(len(items)), format_func=lambda i: items[i].label
    )
    e = items[pick]
    c, res = e.comparison, e.parent

    box = getattr(st, _BOX[_TONE[c.verdict.label]])
    box(f"**{LABEL_TEXT[c.verdict.label]}**  \n{c.verdict.explanation}")
    if is_exp and e.role == "guardrail":
        st.caption("Guardrail: the margin is a non-inferiority margin, not an equivalence band.")
    for w in c.warnings:
        st.warning(w)
    left, mid, right = st.columns(3)
    left.metric("Effect (candidate wins if positive)", f"{c.oriented_estimate:.4g}")
    mid.metric("CI", f"{c.oriented_ci_low:.4g} to {c.oriented_ci_high:.4g}")
    right.metric("Adjusted p", "n/a" if c.adjusted_p is None else f"{c.adjusted_p:.3g}")
    figure = interval_svg(
        c.oriented_estimate, c.oriented_ci_low, c.oriented_ci_high, res.plan.min_effect, True
    )
    if figure:
        st.image(figure, width=560)
    st.caption(f"Dashed line: no difference. Band: ± the minimum effect δ = {res.plan.min_effect}.")
    if c.verdict.n_needed:
        st.write(
            f"**How much more data?** About {c.verdict.n_needed:.0f} observations in total "
            f"(currently {c.n_pairs}). {c.verdict.n_needed_note or ''}"
        )
    st.dataframe(method_rows(c), hide_index=True)
    st.markdown(f"**Why this method:** {c.primary.rationale}")
    for m in c.cross_checks:
        st.markdown(f"- *{m.method}:* {m.rationale}")
    with st.expander("Diagnostics"):
        st.json({m.method: m.diagnostics for m in (c.primary, *c.cross_checks)}, expanded=False)

    family = res.family_checks
    if family or res.family_warnings:
        st.subheader("Checks on the whole set of candidates")
        for m in family:
            st.markdown(f"**{m.method} = {m.estimate:.2f}.** {m.rationale}")
        for w in res.family_warnings:
            st.warning(w)

    with st.expander("Plan"):
        st.code(saved.plan.pretty())
    html = experiment_html(saved) if isinstance(saved, ExperimentResult) else comparison_html(saved)
    st.download_button("Download the HTML report", html, f"{saved.experiment}.html", "text/html")


def main(argv: list[str] | None = None) -> None:
    """Entry point. ``argv[0]``, when given, is the result file to open at start."""
    import streamlit as st

    args = sys.argv[1:] if argv is None else argv
    st.set_page_config(page_title="isbetter viewer", layout="wide")
    text: str | None = None
    with st.sidebar:
        st.header("isbetter viewer")
        st.caption("Reads a saved result only: no raw data is loaded and no test is run.")
        up = st.file_uploader("Result JSON", type="json")
    if up is not None:
        text = up.getvalue().decode("utf-8", errors="replace")
    elif args:
        try:
            text = Path(args[0]).read_text()
        except OSError as exc:
            st.error(f"Cannot read {args[0]}: {exc}")
            return
    if text is None:
        st.info("Upload a result saved with `isbetter run config.yaml --json-out result.json`.")
        return
    try:
        saved = load_saved(text)
    except ViewerError as exc:
        st.error(f"Cannot open this file: {exc}")
        return
    _show(st, saved)


if __name__ == "__main__":
    main()
