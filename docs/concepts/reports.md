# Reports

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

