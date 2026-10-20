# Power BI dashboard plan

Status: P0 step 1 built (`isbetter export-bi` writes `comparisons.csv` and `crosschecks.csv`); the calibration CSV and the `.pbix` are not built. Copied from the Claude Docs page "Power BI Dashboard Plan for the isbetter Toolkit" (2026-10-20); this copy is the one to edit from now on.

## Overview

The Power BI dashboard should be a viewer for the `Result` objects that `isbetter` already produces, not a second place where statistics are computed. This follows the project's [BLUEPRINT.md](BLUEPRINT.md).

An earlier idea had DAX computing lift, standard errors and confidence intervals. Those numbers come from the toolkit's tested methods, so recomputing them in DAX would bypass the oracle and calibration tests that are the point of the project. DAX only selects, counts and formats.

The blueprint removed the Streamlit viewer, so Power BI fills that gap. Keep the `.pbix` local and do not publish it to the Power BI Service, in line with the blueprint's "personal, local, unpublished" scope.

## Data model: four flat tables

Built so far: `comparisons` and `crosschecks`, from `isbetter export-bi RESULT.json... --out-dir bi` (input is the file `isbetter run --json-out` writes; `bi/` is git-ignored). `agrees` means: the cross-check's oriented interval, put through the same three-way verdict rule as the primary, gives the same label, so it lands on the same side of 0 and of ±δ. A cross-check with no interval (the sign-flip permutation test) is compared on the side of 0 only: the sign of its estimate if its p-value is below the comparison's alpha, otherwise "neither".

One `isbetter export-bi` command writes four flat CSVs from the saved result JSON. At this size (dozens to a few hundred comparisons) a seven-table star schema only adds relationships to maintain.

| Table | One row per | Key columns |
| --- | --- | --- |
| `comparisons` | baseline × candidate × metric | experiment, created_at, dependence, primary method and rationale, oriented estimate, CI low, CI high, adjusted p, alpha, n_pairs, min effect δ, verdict label, `n_needed` and its note, warnings (joined), and for A/B runs `ship` and `srm_flagged` |
| `crosschecks` | secondary method per comparison | comparison key, method, estimate, CI, `agrees` (computed in the export: same side of 0 and of ±δ as the primary) |
| `calibration` | method × scenario | scenario, method, role (recommended or naive foil), nominal rate, observed rate, Monte Carlo band low and high, reading |
| `verdict_order` | the six verdict labels | label, sort order, colour, plain-English text. The only hand-typed table |

Ship decisions and the sample-ratio result live in `ExperimentResult` (`decisions`, `srm`), not in `Result`, so the export reads both JSON shapes. Use the `oriented_*` fields so a positive number always means the candidate is better. `diagnostics` is free-form per method and is left out: the rationale already carries what matters.

## Report pages: three

Three pages, in the order you read a result: what to do, why, and whether to trust it.

| Page | What it shows | What you learn |
| --- | --- | --- |
| 1. Decision | Experiment slicer, verdict cards, one table coloured by verdict with `n_needed` and warnings visible, plus ship and SRM columns for A/B runs | The answer first (better, equivalent or inconclusive) and how much more data would settle an inconclusive one |
| 2. Evidence | Forest plot of the oriented estimate with CI error bars and a shaded ±δ band, cross-check agreement, rationale on drill-through | Whether the interval clears the band and whether independent methods agree |
| 3. Trust | Observed versus nominal rate per method with the Monte Carlo band, the naive foil beside the recommended method for each scenario | Why the numbers can be believed: naive tests give 26 to 57% false positives where the recommended ones give about 5% |

## Phased plan

Three steps. After P1 the dashboard is usable, so P2 can be skipped. Start with the fraud and Hillstrom results, then add the others as you generate them.

| Phase | Goal | Deliverables | Done when |
| --- | --- | --- | --- |
| P0 | Export | `isbetter export-bi` writing the comparisons and crosschecks tables from `Result` and `ExperimentResult` JSON (in the CLI layer, since the stats core stays I/O-free), and a CSV output for `validation/calibration_table.py`. Two small PRs | A round-trip test shows the CSV values equal the JSON, and the verdict counts equal those of `isbetter run` for every case-study result. Outputs are git-ignored |
| P1 | The `.pbix` | Import the tables, colour from `verdict_order`, build the three pages, add a short refresh runbook in `docs/`. The `.pbix` itself is not committed | Verdict counts in Power BI equal the Python side, and a new run appears after re-export and Refresh |
| P2 | Polish (optional) | Theme, tooltips, one-page PDF export | Only if P1 gets used |

## 8 GB RAM and what to confirm

Power BI loads only the small summary tables above and never the raw data. That matches the blueprint's rule that reports are built from the saved `Result` only.

The earlier open items are answered by the code (see Data model). Three things remain to confirm:

- Power BI Desktop runs only on Windows. Confirm the machine you will use.
- The forest plot needs error bars drawn from fields (`ci_low` and `ci_high`). Check that your Desktop version supports them.
- Define `agrees` exactly during P0 and cover it with a test, so the cross-check column means one thing.

## Sources

- [BLUEPRINT.md](BLUEPRINT.md)
- [`core/result.py`](../src/isbetter/core/result.py): `Result`, `ComparisonResult` and `MethodResult` fields
- [`core/experiment.py`](../src/isbetter/core/experiment.py): `ExperimentResult`, with `decisions` and `srm`
- [`decision/guardrails.py`](../src/isbetter/decision/guardrails.py): `Decision` and `MetricCheck`
- [`validation/calibration_table.md`](../validation/calibration_table.md): the calibration numbers on the Trust page
