# Using it on your own data

This page takes you from "I have a file" to a verdict you can trust. Each recipe below was run end
to end on a small file with the columns it describes, and `tests/unit/test_runbook.py` re-runs them
on every `pytest`, so the configs and column lists here cannot drift away from the code.

```bash
uv run isbetter run config.yaml --out REPORT.md --json-out result.json
```

- A relative `data:` path is resolved against the folder holding the config, not your shell's.
- The report goes to `REPORT.md` in the current folder unless you pass `--out`.
- A bad config or bad input stops with a message starting `error:` and exit code 2, before
  anything is analysed. [Section 5](#5-messages-you-will-see) lists the common ones.
- Keep your data and reports out of git (`data/` and `case_studies/*/out/` are the ignored places).

## 1. Decide the dependence first

The one choice that matters is how your observations depend on each other. It is a fact about how
the data were collected, so settle it before you look at any result.

| Question about your data | If yes |
|---|---|
| Did every unit get measured under **both** variants (same questions, same CV folds, same origins)? | `paired` |
| Do several rows belong to one randomisation unit (sessions per user, questions per passage)? | `clustered` |
| Are the scores from **resampled train/test splits** (repeated k-fold)? | `cv_overlap` |
| Are rows **forecast origins** or **periods in time**, so neighbours are related? | `timeseries` |
| None of these: each row is its own independent unit, one variant per unit | `iid` |

If two apply, the combination usually exists (`clustered` with `paired: true` for items grouped
in passages and scored by two models). If it does not, the router refuses and says why; it never
substitutes a weaker test. See [declaring dependence](concepts/dependence.md) for what each choice
routes to.

## 2. Pick a recipe

| You have | Recipe | `adapter` | `dependence` |
|---|---|---|---|
| Two models scored on the same items | [A](#a-two-models-on-the-same-items) | `csv_long` | `paired` |
| A randomised experiment, one row per user | [B](#b-a-randomised-experiment) | `ab_table` | `iid` |
| An experiment with several rows per user | [B](#several-rows-per-user) | `ab_table` | `clustered` |
| A 10-million-row experiment table | [B](#a-very-large-table) | `ab_table`, `mode: stats` | `iid` |
| Repeated k-fold CV scores | [C](#c-cross-validation-scores) | `csv_long` | `cv_overlap` |
| Forecast losses per origin | [D](#d-forecast-losses) | `csv_long` | `timeseries` |
| Returns of trading strategies | [E](#e-strategy-returns) | `strategy_returns` | `timeseries` |
| Per-item LLM eval scores | [F](#f-llm-eval-scores) | `llm_eval` | `clustered` + `paired` |
| An experiment you are still watching | [G](#g-a-live-experiment) | `csv_long` | `iid` + `sequential` |

All recipes read CSV except F (JSON Lines); `ab_table` also reads Parquet.

## 3. Recipes

### A. Two models on the same items

One row per item **per variant**. `value` is the score; `pair_id` is the key that ties the two
rows of an item together. Every `pair_id` must appear under every variant exactly once.

```text
unit_id,pair_id,variant,metric,value
q0,q0,baseline,accuracy,0.7929
q0,q0,candidate,accuracy,0.7984
```

<!-- recipe: paired -->
```yaml
experiment: model_a_vs_b
adapter: csv_long
data: paired_scores.csv
variants: {baseline: baseline, candidates: [candidate]}
metric: {name: accuracy, higher_is_better: true}
dependence: {type: paired}
decision: {min_effect: 0.01, alpha: 0.05}
seed: 1
```

`metric.name` selects the rows whose `metric` column equals it, so one file can hold several
metrics and you run one config per metric. For right/wrong outcomes (0/1 per item) add
`kind: binary`: the router then uses a paired score interval with McNemar's p-value, which is
sound at small N where the usual normal interval is not
([small-N intervals](methods/small-n.md)). Without `kind: binary`, 0/1 scores are treated as
continuous.

### B. A randomised experiment

One row per randomisation unit, one column per metric. The arm column holds the variant labels.

```text
user_id,arm,converted,revenue,revenue_pre,sessions,clicks
u0,control,0,0.0,12.4,3,5
u1,treatment,1,31.2,40.1,5,11
```

<!-- recipe: ab -->
```yaml
experiment: checkout_test
adapter: ab_table
data: experiment.csv
table: {variant_col: arm, unit_col: user_id}
variants: {baseline: control, candidates: [treatment]}
metrics:
  - {name: converted, kind: binary, min_effect: 0.005}
  - {name: revenue, kind: continuous, role: guardrail, min_effect: 0.5, variance_reduction: cuped, covariate: revenue_pre}
dependence: {type: iid}
decision: {alpha: 0.05, correction: holm}
design: {expected_split: {control: 0.5, treatment: 0.5}}
seed: 1
```

What this adds over recipe A:

- **Sample-ratio check.** It always runs. `design.expected_split` states the split you intended;
  without it an equal split is assumed. If it fails (p < `srm_alpha`, default 0.001) the
  randomisation is suspect and the effect estimates should not be trusted until you know why.
- **Success metrics and guardrails.** `role: success` metrics are tested for improvement, with
  `correction` applied across all of them and all candidates. A `role: guardrail` metric must not
  get worse by more than its `min_effect` (a non-inferiority margin, required and > 0). The ship
  decision needs a success win, a passed SRM and no failed guardrail.
- **CUPED.** `variance_reduction: cuped` with a `covariate` column tightens the interval. The
  covariate must be measured **before** the experiment started, or it is contaminated by the
  treatment.
- **Binary vs continuous.** A 0/1 column is `kind: binary` (two-proportion z-test); anything else
  numeric is `continuous` (Welch t).
- **Ratio metrics.** For clicks per session or revenue per visit, give both columns:
  `{name: clicks, kind: ratio, denominator: sessions}`. The delta method works over users.
  Do not divide in a spreadsheet and analyse the result as a plain mean (see
  [way 2](#4-ways-to-get-a-wrong-answer)).

#### Several rows per user

If the file has one row per session or per page view but users were randomised, the rows of one
user are not independent. Name the user column as the cluster:

<!-- recipe: clustered -->
```yaml
experiment: session_test
adapter: ab_table
data: sessions.csv
table: {variant_col: arm, unit_col: session_id, cluster_col: user_id}
variants: {baseline: control, candidates: [treatment]}
metric: {name: time_on_page, kind: continuous}
dependence: {type: clustered, cluster_col: user_id}
decision: {min_effect: 0.1, alpha: 0.05}
seed: 1
```

Set `cluster_col` in both places, to the same column. The sample-ratio check then counts users, not
rows.

#### A very large table

`table: {variant_col: arm, mode: stats}` streams the file through DuckDB and keeps only counts and
sums per arm, so memory does not grow with the row count. It supports `dependence: iid` only and
needs no `unit_col`. The ratio metric's bootstrap cross-check needs unit rows and cannot run in
this mode.

<!-- recipe: stats -->
```yaml
experiment: big_test
adapter: ab_table
data: experiment.csv
table: {variant_col: arm, mode: stats}
variants: {baseline: control, candidates: [treatment]}
metrics:
  - {name: converted, kind: binary, min_effect: 0.005}
dependence: {type: iid}
decision: {alpha: 0.05}
design: {expected_split: {control: 0.5, treatment: 0.5}}
seed: 1
```

Check first that your rows really are the randomisation unit. A table of impressions or events,
where one user appears many times, needs `clustered` (or aggregating to users first), and
`mode: stats` cannot do that.

### C. Cross-validation scores

One row per (variant, repeat, fold). All variants must have been scored on identical folds.

```text
unit_id,pair_id,variant,metric,value,repeat,fold
r0f0,r0f0,logreg,pr_auc,0.6701,0,0
r0f0,r0f0,rf,pr_auc,0.8516,0,0
```

<!-- recipe: cv -->
```yaml
experiment: lgbm_vs_logreg
adapter: csv_long
data: cv_scores.csv
variants: {baseline: logreg, candidates: [lightgbm]}
metric: {name: pr_auc, higher_is_better: true}
dependence: {type: cv_overlap, n_train: 8000, n_test: 2000, repeats: 5, folds: 5}
decision: {min_effect: 0.01, alpha: 0.05}
seed: 1
```

`n_train` and `n_test` are the sizes of **one** train and one test split (the correction for
overlapping training sets depends on their ratio), and `repeats` × `folds` must equal the number
of rows per variant. `isbetter` checks that count and refuses a mismatch.

To produce the file from scikit-learn estimators (needs the `ml` extra), `run_cv` scores every
estimator on identical repeated stratified folds and returns the sizes to copy into the config:

```python
import csv
from isbetter.adapters.sklearn_cv import run_cv

out = run_cv(X, y, {"logreg": make_logreg, "rf": make_rf}, n_splits=5, n_repeats=3, seed=1)
cols = out.table.columns
with open("cv_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    w.writerows(zip(*cols.values(), strict=True))
print(out.n_train, out.n_test, out.repeats, out.folds)
```

`run_cv` scores PR-AUC only. If you ran the CV yourself, write the columns above and set the four
numbers from your own splitter.

### D. Forecast losses

One row per (variant, forecast origin) holding that origin's loss, with `time` as a number that
increases strictly within each variant. Use one `metric` per horizon (`mae_h3`, `mase_h7`) and
analyse each horizon on its own.

```text
unit_id,pair_id,variant,metric,value,time
o0,o0,seasonal_naive,mae_h3,2.8019,0
o0,o0,arima,mae_h3,2.4639,0
```

<!-- recipe: forecast -->
```yaml
experiment: arima_vs_naive
adapter: csv_long
data: forecast_losses.csv
variants: {baseline: seasonal_naive, candidates: [arima]}
metric: {name: mae_h3, higher_is_better: false}
dependence: {type: timeseries, horizon: 3}
decision: {min_effect: 0.05, alpha: 0.05}
seed: 1
```

- A loss is lower-is-better: set `higher_is_better: false` or the verdict is reversed.
- `horizon` is the forecast step *h*. Errors *h* steps ahead overlap across neighbouring origins,
  and the Diebold–Mariano test uses *h* − 1 autocovariances to account for it. It must match the
  horizon the losses were made at.
- Dates must be converted to numbers (an index or a day count) before `csv_long` will read them.
- To compute the losses, `isbetter.adapters.forecast_backtest.rolling_origin_backtest` runs the
  forecasters over expanding windows and produces the table in this layout.

### E. Strategy returns

A wide CSV: one date column and one column of **per-period returns** per strategy, all over the
same periods. Not prices, not cumulative returns.

```text
date,buy_and_hold,trend_fast,trend_slow
2023-01-02,0.0007,0.0042,-0.0136
2023-01-03,0.0140,-0.0016,0.0062
```

<!-- recipe: returns -->
```yaml
experiment: trend_vs_hold
adapter: strategy_returns
data: returns.csv
table: {time_col: date}
variants: {baseline: buy_and_hold, candidates: [trend_fast, trend_slow]}
metric: {name: return, kind: returns}
dependence: {type: timeseries}
decision: {min_effect: 0.02, alpha: 0.05}
finance: {n_trials: 2, sr_variance: 0.0016}
seed: 1
```

- `finance` is required, because the best of many variants looks good by luck. `n_trials` is the
  number of variants **you tried in total**, including the ones you dropped, not the number of
  columns in this file. `sr_variance` is the variance of the trials' per-period Sharpe ratios. For
  a real search, log every variant to a `TrialRegistry` and point `finance.registry` at it instead.
- Sharpe ratios are per period, not annualised, so `min_effect` is in those units (0.02 per day is
  large).
- `decision.correction` must stay `none`; selection across variants is handled by the Deflated
  Sharpe Ratio and the probability of backtest overfitting.

### F. LLM eval scores

JSON Lines, one object per (model, item): `model`, `item_id` and `score` are required; `cluster_id`
(the passage, task or source the item came from) and `metric` are optional. `score` may be a number
or `true`/`false`. Every item must be scored by every model.

```text
{"model": "model_a", "item_id": "p0q0", "cluster_id": "p0", "score": 1}
{"model": "model_b", "item_id": "p0q0", "cluster_id": "p0", "score": 0}
```

<!-- recipe: llm -->
```yaml
experiment: model_a_vs_b_llm
adapter: llm_eval
data: llm_items.jsonl
variants: {baseline: model_a, candidates: [model_b]}
metric: {name: correct, higher_is_better: true}
dependence: {type: clustered, paired: true}
decision: {min_effect: 0.03, alpha: 0.05}
seed: 1
```

Items that share a passage are correlated, and both models answered the same items, so the right
declaration is `clustered` with `paired: true` when you have a `cluster_id`, and plain `paired`
(with `kind: binary` for right/wrong scores) when items are independent. Do not give some items a
`cluster_id` and not others. No model is run by `isbetter`; it reads scores you already have.

### G. A live experiment

If you will look at the result before the planned sample size is reached, use a confidence
sequence, which stays valid however often you look. The file needs a `time` column giving arrival
order.

```text
unit_id,variant,metric,value,time
c0,control,revenue,18.16,0
t0,treatment,revenue,0.68,0
```

<!-- recipe: stream -->
```yaml
experiment: live_revenue
adapter: csv_long
data: stream.csv
variants: {baseline: control, candidates: [treatment]}
metric: {name: revenue, higher_is_better: true}
dependence: {type: iid}
decision: {min_effect: 0.5, alpha: 0.05}
sequential: {n_target: 2000}
seed: 1
```

`n_target` is the sample size per arm you plan to reach, and you must fix it **before** looking.
Setting it from the data you already have is refused, because tuning to the data breaks the
guarantee. The fixed-horizon test is printed beside it only as a warning of what peeking would have
said. Supported for one metric under `iid` or `paired`; clustered data and full A/B experiments
(SRM, guardrails) are not routed.

## 4. Ways to get a wrong answer

Most are not arithmetic mistakes; the code does what you told it to. In rough order of how often
they happen:

1. **The wrong `dependence`.** The classic way to a false win. Declaring `iid` for data that are
   clustered, from CV folds, or from overlapping forecast errors makes the interval too narrow and
   the p-value too small. On a test file with eight sessions per user, ignoring the user cluster
   turned a DO NOT SHIP into a SHIP, and treating CV folds as paired shrank the interval to well
   under half its honest width. The rates for the textbook choices are in the
   [calibration table](calibration.md) and the
   [failure gallery](../validation/failure_gallery/): a pooled test on 10 sessions per user gives 31%
   false positives, a multi-step forecast test that ignores overlap 17%. The opposite mistake
   (declaring `iid` for paired data) is safe but wasteful: the pairing is thrown away and a real
   win can come back `inconclusive`.
2. **A ratio metric analysed as a plain mean.** Clicks per session, revenue per visit and
   conversion per impression are ratios of two sums. Dividing per row and averaging answers a
   different question (it weights every row equally) and ignores that numerator and denominator
   move together. Use `kind: ratio` with the denominator column, one row per randomisation unit
   ([delta method](methods/delta-method.md)).
3. **The wrong direction.** A loss, a latency, or an error rate is lower-is-better. Leave
   `higher_is_better` at its default of `true` and a clear improvement is reported as the
   candidate being worse. The report labels the direction on the metric line of the plan; read it.
4. **Peeking.** Re-running a fixed-horizon test on a growing experiment until it crosses p < 0.05
   gives false positives in 46% of A/A experiments read after every observation
   (`validation/failure_gallery/04_peeking.py`). Use recipe G, or look once at a sample size you
   fixed in advance.
5. **`min_effect` left at 0.** With δ = 0 the `equivalent` verdict cannot occur and any tiny
   positive interval counts as a win. Set δ to the smallest difference you would act on, before
   seeing the data ([the verdict](concepts/verdict.md)).
6. **Many candidates or metrics with no correction.** Five candidates at α = 0.05 give about a 23%
   chance of at least one false win. Set `decision.correction` (`holm` is the safe default). For
   strategy returns use the `finance` block instead.
7. **Counting only the survivors.** Picking the best of many variants and then testing it against a
   baseline is a biased comparison. Count every variant you tried (`n_trials`), or compare on data
   not used for the selection.
8. **Ignoring a failed sample-ratio check.** If the arms are not the size you randomised to, some
   units were lost or reassigned non-randomly, and any difference may be bias rather than effect.
9. **A covariate measured after the treatment started.** CUPED needs a pre-experiment covariate.
10. **Treating rows as units.** A table of events or impressions is not a table of users. If users
    were randomised, analyse users (cluster them, or aggregate first).

## 5. Messages you will see

| Message contains | What it means and what to do |
|---|---|
| `missing required column(s)` | A `csv_long` file needs `unit_id`, `variant`, `metric`, `value`. Rename your columns, or use `ab_table`. |
| `partner(s), e.g. pair_id=` | A paired item is missing under one variant. Drop it from both or fill the gap; do not pair by position. |
| `duplicate (variant, metric, pair_id) rows` | An item appears twice under the same variant. De-duplicate, or you meant `clustered`. |
| `no rows for variant` | A name in `variants` (or `metric.name`) is not in the file. Check spelling and case against the data. |
| `must be numeric` | `value` has blanks or text. Remove or impute those rows deliberately. |
| `convert dates to numbers first` | `csv_long` needs a numeric `time`. Use an index or day count. (`strategy_returns` reads a date column as is.) |
| `must be strictly increasing within variant` | Sort by time, and remove repeated timestamps within a variant. |
| `scores per variant, found` | `repeats` × `folds` does not match the file. Fix the config or the file. |
| `Extra inputs are not permitted` | A key in the config is misspelt or not a setting; unknown keys are errors, not ignored. |
| `contains arm(s) not named in the config` | The file has a variant the config does not list. Add it to `variants` or filter the file. |
| ``needs a `finance` block`` | Returns need `finance.n_trials` or a registry; there is no default. |
| `Refusing to` | The dependence and metric combination has no valid method. Change one of them; do not expect a weaker test. |

## 6. Before you trust a verdict

- The plan printed before the results names the primary method. Does it match how your data were
  collected?
- Is the direction (`higher is better` / `lower is better`) on the plan's metric line what you
  meant?
- Was `min_effect` chosen before looking, in the metric's own units?
- For an experiment: did the sample-ratio check pass?
- For anything picked from many: is `correction`, or the `finance` block, accounting for all of
  them?
- If the verdict is `inconclusive`, the report says how much more data would settle it, or that
  more data will not help. Believe it over rerunning with a different test.
