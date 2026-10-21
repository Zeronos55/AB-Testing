# Power BI runbook

How to build and refresh the local dashboard described in the [Power BI plan](powerbi-plan.md).
Everything the report shows is a column or a count of the CSV tables below; DAX selects, counts
and formats, and never computes a statistic. The `.pbix` stays on your machine and is not
committed (it is also not published to the Power BI Service).

> **Not yet tried in Power BI Desktop.** The Python side (the CSV tables, `verdict_order`, the
> verdict counts) is built and tested. The Power Query and DAX below were written against the
> table columns but have not been run in Desktop, because Desktop is Windows-only. Section 7 is
> the check that settles it; fix anything that does not match and edit this page.

## 1. Make the tables

```bash
uv run isbetter run examples/toy_paired.yaml --out REPORT.md --json-out results/toy.json
uv run isbetter run case_studies/01_fraud_classifiers/config.yaml --json-out results/fraud.json
uv run isbetter export-bi results/*.json --out-dir bi
uv run python validation/calibration_table.py --csv bi/calibration.csv   # a few minutes
```

- A case study's `config.yaml` reads `out/…`, so run its `run.py` once first to write that data.
- `export-bi` prints the verdict counts. Keep that output: it is the number the dashboard must
  reproduce (section 7).
- `bi/` and `results/` are git-ignored data. If Desktop runs on another machine, copy `bi/` over.

| File | One row per | Used for |
|---|---|---|
| `comparisons.csv` | baseline × candidate × metric of a run | Decision and Evidence pages |
| `crosschecks.csv` | secondary method of a comparison | Evidence page; joins on `comparison_key` |
| `verdict_order.csv` | the six verdict labels | sort order, colour and wording of every verdict |
| `calibration.csv` | method × scenario | Trust page |

## 2. Load them (Power Query)

*Home → Transform data → New Source → Blank Query → Advanced Editor*, once per table. First add a
text parameter `BiFolder` (*Manage Parameters*) holding the folder, for example `C:\isbetter\bi`,
so a moved folder is a one-place change.

`comparisons` (`crosschecks`, `verdict_order` and `calibration` follow the same pattern with
their own column lists; the full lists are `COMPARISON_COLUMNS`, `CROSSCHECK_COLUMNS` and
`VERDICT_ORDER_COLUMNS` in `src/isbetter/report/bi.py`, and `CSV_COLUMNS` in
`validation/calibration_table.py`):

```m
let
    Source = Csv.Document(File.Contents(BiFolder & "\comparisons.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    // a NaN is written as the text "nan"; make it empty so the number columns can be typed
    NoNaN = Table.ReplaceValue(Headers, "nan", null, Replacer.ReplaceValue,
        {"estimate", "oriented_estimate", "oriented_ci_low", "oriented_ci_high", "p_value",
         "adjusted_p"}),
    Typed = Table.TransformColumnTypes(NoNaN, {
        {"created_at", type datetimezone}, {"seed", Int64.Type},
        {"higher_is_better", type logical}, {"estimate", type number},
        {"oriented_estimate", type number}, {"oriented_ci_low", type number},
        {"oriented_ci_high", type number}, {"ci_level", type number},
        {"p_value", type number}, {"adjusted_p", type number}, {"alpha_used", type number},
        {"family_size", Int64.Type}, {"n_pairs", Int64.Type}, {"min_effect", type number},
        {"n_needed", type number}, {"ship", type logical}, {"srm_flagged", type logical}},
        "en-US")
in
    Typed
```

Pass the culture (`"en-US"`) so a decimal point is read as a point on any Windows locale.
Empty cells (no interval, no `n_needed`, not an A/B run) become null, and a null is shown as blank,
never as 0. In `crosschecks` type `agrees` as logical; in `verdict_order` `sort_order` is
whole-number and the rest text; in `calibration` the rates (`nominal`, `observed`, `band_low`,
`band_high`) are numbers (fractions: format them as %), `reps` whole-number and `exact` logical.

## 3. Model

*Model view.* Three relationships, all single direction (the "one" side filters the "many" side):

| From (one) | To (many) |
|---|---|
| `comparisons[comparison_key]` | `crosschecks[comparison_key]` |
| `verdict_order[label]` | `comparisons[verdict]` |

`calibration` stands alone. Then:

- Select `verdict_order[label]` → *Column tools → Sort by column* → `sort_order`. Show
  `verdict_order[label]` (not `comparisons[verdict]`) whenever a verdict is on an axis, a legend or
  a slicer, so the six always come in reading order.
- Hide the `comparison_key` columns in report view; they are for the join only.
- Set `calibration[observed]`, `nominal`, `band_low` and `band_high` to *Percentage*.

## 4. Measures

A *Measures* table (*Enter data*, one empty column, then delete the column) holds them. They only
count, select and format.

```dax
Comparisons = COUNTROWS ( comparisons )

Clearly better =
    CALCULATE ( [Comparisons], comparisons[verdict] = "A_better_practical" )
Better, may be trivial =
    CALCULATE ( [Comparisons], comparisons[verdict] = "A_better_maybe_trivial" )
Equivalent = CALCULATE ( [Comparisons], comparisons[verdict] = "equivalent" )
Inconclusive = CALCULATE ( [Comparisons], comparisons[verdict] = "inconclusive" )
Baseline better, may be trivial =
    CALCULATE ( [Comparisons], comparisons[verdict] = "B_better_maybe_trivial" )
Baseline clearly better =
    CALCULATE ( [Comparisons], comparisons[verdict] = "B_better_practical" )

Verdict colour = MAX ( verdict_order[colour] )
Verdict text = MAX ( verdict_order[text] )

More data needed = MAX ( comparisons[n_needed] )          -- blank when nothing is needed
Estimate = MAX ( comparisons[oriented_estimate] )         -- one row per comparison in a table
CI low = MAX ( comparisons[oriented_ci_low] )
CI high = MAX ( comparisons[oriented_ci_high] )
Min effect = MAX ( comparisons[min_effect] )
Min effect, negative = -[Min effect]

Cross-checks = COUNTROWS ( crosschecks )
Cross-checks agreeing =
    CALCULATE ( COUNTROWS ( crosschecks ), crosschecks[agrees] = TRUE () )
Agreement = DIVIDE ( [Cross-checks agreeing], [Cross-checks] )

Observed rate = MAX ( calibration[observed] )
Nominal rate = MAX ( calibration[nominal] )
Band low = MAX ( calibration[band_low] )
Band high = MAX ( calibration[band_high] )
```

`MAX` is used only to read the single value of a one-row context (a table row, a forest-plot
point); never put these measures in a visual that groups several comparisons into one row.
The six verdict counts add up to `[Comparisons]`; a label that has no rows shows 0, not blank.

## 5. The three pages

Order: what to do, why, and whether to trust it. A *Verdict* conditional format uses
*Format by → Field value* on `[Verdict colour]`, and the wording is always shown next to the
colour, so colour is never the only cue.

**1. Decision**
- Slicers: `comparisons[experiment]`, `comparisons[created_at]` (so two runs of an experiment
  are never summed), `comparisons[role]`.
- Six cards, one per verdict measure, in `verdict_order` order.
- A table of `baseline`, `candidate`, `metric`, `verdict_order[text]` (background from
  `[Verdict colour]`), `[More data needed]`, `n_needed_note`, `warnings`, and for A/B runs `ship`
  and `srm_flagged`. Keep `warnings` visible: they are part of the answer.

**2. Evidence**
- Forest plot: a clustered bar chart, category = `baseline | candidate | metric` (a column made in
  Power Query from the three), value = `[Estimate]`, with *Analytics → Error bars* bound to
  `[CI low]` and `[CI high]`. Mark ±δ as two constant lines, bound to `[Min effect]` and `[Min effect, negative]` when one
  experiment is selected (*Constant line → fx*). Both are to be confirmed in your Desktop version
  (see "what to confirm" in the plan); if error bars cannot take a field, fall back to a table with
  a data bar on the interval.
- A matrix of `method`, `[Cross-checks]`, `[Cross-checks agreeing]`, `[Agreement]`.
- A drill-through page from a comparison showing `primary_rationale`, `verdict_explanation` and the
  cross-check rows.

**3. Trust**
- Clustered bars: axis = `scenario`, legend = `method`, value = `[Observed rate]`, error bars from
  `[Band low]` and `[Band high]`, a reference line at `[Nominal rate]`. A slicer on `calibration[role]`
  puts each recommended method beside the naive foil it replaces.
- Rows with `exact = true` have no band (no simulation noise): leave them out of the error bars.
- A table of `measures` and `reading`, the script's own wording of each row.

## 6. Refresh

New results appear after three steps and no edits to the report:

1. Run the experiment: `uv run isbetter run config.yaml --json-out results/NAME.json`.
2. Re-export everything: `uv run isbetter export-bi results/*.json --out-dir bi`
   (and, only if the methods changed, the `calibration.csv` command in section 1).
3. In Desktop, *Home → Refresh*.

`comparison_key` contains the run time, so a re-run is a new set of rows next to the old one, not
a replacement. Filter with the `created_at` slicer, or delete the old JSON before exporting if you
want only the latest.

## 7. Check it

The plan's "done when": the verdict counts in Power BI equal the Python side.

1. Run the export and note the six counts it prints.
2. On the Decision page with no slicer selected, the six cards must show the same six numbers, and
   the table must have as many rows as the sum.
3. Add a new run, re-export, Refresh: the counts must move by exactly the new comparisons.

If a count differs, look first at `created_at` (a slicer still selected), then at a typed column
turning an unparseable value into an error row (*Transform data → View → Column quality*).
The Python side of this contract is `tests/unit/test_export_bi.py`: it asserts the CSV equals the
saved JSON, the printed counts equal the CSV, and `verdict_order` lists every label once.
