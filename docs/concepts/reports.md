# Reports and the viewer

`isbetter run config.yaml` always prints the plan first, then writes the report.

## Markdown and HTML

`report: {format: [markdown, html]}` in the config selects the outputs (default: Markdown only).
The Markdown goes to `--out` (default `REPORT.md`) and the HTML next to it with an `.html` suffix,
or to `--html-out`.

The HTML report is one self-contained file: inline CSS and SVG, no scripts, fonts or network
requests, light and dark themes, and it prints cleanly. It has the same plan, verdict, tables and
rationale as the Markdown, plus a verdict badge per comparison and a figure of the interval
against zero and the ± δ band. A/B experiments add the sample-ratio table and ship / do-not-ship
badges.

## The viewer

```bash
uv sync --extra app                       # installs Streamlit
uv run isbetter run config.yaml --json-out result.json
uv run isbetter view result.json
```

The viewer reads only the saved result JSON. It never loads raw data and never reruns a test,
so it opens instantly and cannot disagree with the report. It shows the sample-ratio check and
ship decisions for experiments, a table of all comparisons, and for each one the verdict,
warnings, effect and interval, the figure, the primary method and cross-checks with the reason
each was chosen, diagnostics, any checks on the whole set of candidates (for example PBO), and a
download button for the HTML report. You can also upload a result file in the sidebar.
