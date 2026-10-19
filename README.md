# AB-Testing: `isbetter`

**Is A really better than B?** You tell it how your data are dependent, and it gives you a valid statistical test, a three-way verdict (better / equivalent / inconclusive) and proof that the test is calibrated.

`isbetter` is planned as a cross-domain comparison toolkit. It treats several kinds of comparison as the same problem: ML model comparisons, A/B experiments, forecast backtests, trading strategies and LLM evals. In each case you have paired or unpaired scores with a known dependence structure (`iid`, `paired`, `clustered`, `cv_overlap`, `timeseries`), and that structure determines which test is valid.

## Is it calibrated?

Every method is checked on worlds where the truth is known: two equal arms (how often does it
declare a difference?) or a known parameter (does the 95% interval contain it?). Rows marked
*naive* are the textbook methods that `isbetter` replaces, shown beside the method it uses instead.
Rates are from `validation/calibration_table.py` (seeded, up to 10,000 replications per row; the
full table with Monte Carlo bands is in [validation/calibration_table.md](validation/calibration_table.md)).

<!-- calibration-table:start -->

| Scenario | Method | Measures | Nominal | Observed | Reading |
|---|---|---|---:|---:|---|
| A/A, n = 40 per arm, normal | Welch t | false positives | 5.0% | **5.0%** | calibrated |
| A/A, 40 pairs, normal | paired t | false positives | 5.0% | **5.1%** | calibrated |
| A/A, n = 1,500 per arm, conversion rate 10% | two-proportion z | false positives | 5.0% | **4.6%** | calibrated |
| A/A, n = 400 per arm, covariate correlation 0.7, curved | CUPED | false positives | 5.0% | **4.8%** | calibrated |
| A/A, n = 400 per arm, covariate correlation 0.7 | unadjusted Welch *(reference)* | false positives | 5.0% | **5.4%** | calibrated |
| True 40/30/30 split, 20,000 users | SRM chi-square at 0.05 | false alarms | 5.0% | **5.2%** | calibrated |
| A/A, 100 users x 10 sessions per arm, ICC 0.3 | row-level Welch *(naive)* | false positives | 5.0% | **31.2%** | inflated |
| A/A, 100 users x 10 sessions per arm, ICC 0.3 | CR1 by user | false positives | 5.0% | **5.5%** | slightly liberal |
| Equal models, 30 passages x 5 items | paired t ignoring passages *(naive)* | false positives | 5.0% | **26.2%** | inflated |
| Equal models, 30 passages x 5 items | paired CR1 by passage | false positives | 5.0% | **4.9%** | calibrated |
| Equal learners, 10 x 10-fold CV, n = 100 | k-fold t *(naive)* | false positives | 5.0% | **56.8%** | inflated |
| Equal learners, 10 x 10-fold CV, n = 100 | Nadeau-Bengio | false positives | 5.0% | **4.8%** | calibrated |
| Equal forecasters, h = 1, 250 origins | paired t on losses *(naive)* | false positives | 5.0% | **4.5%** | calibrated |
| Equal forecasters, h = 1, 250 origins | Diebold-Mariano + HLN | false positives | 5.0% | **4.5%** | calibrated |
| Equal forecasters, h = 3, 250 origins | paired t on losses *(naive)* | false positives | 5.0% | **17.5%** | inflated |
| Equal forecasters, h = 3, 250 origins | Diebold-Mariano + HLN | false positives | 5.0% | **5.1%** | calibrated |
| Equal forecasters, h = 7, 250 origins | paired t on losses *(naive)* | false positives | 5.0% | **34.6%** | inflated |
| Equal forecasters, h = 7, 250 origins | Diebold-Mariano + HLN | false positives | 5.0% | **5.1%** | calibrated |
| A/A, read after every observation (20 to 2,000 per arm) | fixed-horizon test, stopped at first p < 0.05 *(naive)* | false positives | 5.0% | **48.9%** | inflated |
| A/A, read after every observation (20 to 2,000 per arm) | confidence sequence | false positives | 5.0% | **2.5%** | conservative (valid) |
| A/A, read once at n = 2,000 per arm | fixed-horizon Welch t | false positives | 5.0% | **5.8%** | calibrated |
| AR(1)-GARCH returns, T = 500, phi = 0.2 | iid bootstrap 95% CI *(naive)* | coverage | 95.0% | **87.9%** | under-covers |
| AR(1)-GARCH returns, T = 500, phi = 0.2 | stationary block bootstrap 95% CI | coverage | 95.0% | **92.4%** | slightly under-covers |
| True accuracy 0.95, N = 30 items | normal-theory 95% CI *(naive)* | coverage | 95.0% | **78.2%** | under-covers |
| True accuracy 0.95, N = 30 items | Wilson 95% CI | coverage | 95.0% | **93.9%** | slightly under-covers |
| True accuracy 0.95, N = 30 items | Jeffreys 95% CI | coverage | 95.0% | **98.4%** | calibrated |
| True accuracy 0.95, N = 30 items | Clopper-Pearson 95% CI | coverage | 95.0% | **98.4%** | calibrated |
| Worst true accuracy in 0.02-0.98, N = 30 items | normal-theory 95% CI *(naive)* | worst-case coverage | 95.0% | **45.4%** | under-covers |
| Worst true accuracy in 0.02-0.98, N = 30 items | Wilson 95% CI | worst-case coverage | 95.0% | **93.0%** | slightly under-covers |
| Worst true accuracy in 0.02-0.98, N = 30 items | Jeffreys 95% CI | worst-case coverage | 95.0% | **88.9%** | under-covers |
| Worst true accuracy in 0.02-0.98, N = 30 items | Clopper-Pearson 95% CI | worst-case coverage | 95.0% | **95.6%** | calibrated |
| Paired accuracy difference, N = 60, 0.95 vs 0.90 | normal-theory paired CI *(naive)* | coverage | 95.0% | **91.1%** | under-covers |
| Paired accuracy difference, N = 60, 0.95 vs 0.90 | Tango score CI | coverage | 95.0% | **96.1%** | conservative (valid) |

<!-- calibration-table:end -->

A calibrated method falls outside its Monte Carlo band about 1 time in 20, so a row marked
*slightly liberal* or *slightly under-covers* is not by itself a failure; the repeated nightly
runs (`pytest -m slow`) are the check. The known limits are written down in the sections below.
Regenerate the table with `uv run python validation/calibration_table.py --readme`.

## Documentation

Method cards (when to use each method, what it assumes, how it fails, what it was checked against,
references), the verdict logic, the quickstart and the [runbook for your own data](docs/own-data.md)
are plain Markdown in [`docs/`](docs/index.md).
The failure gallery is in `validation/failure_gallery/` and the case studies in `case_studies/`.

## Status

**Phases 0-6 built** (iid/paired/clustered/CV-overlap/time-series comparison, verdict with `n_needed`, Holm/BH, report, calibration harness, fraud case study; A/B testing with proportions, CUPED, ratio metrics, automatic SRM check, guardrail decisions, power planning, 14M-row aggregation; forecast backtests with Diebold-Mariano, SPA/StepM/MCS and block bootstraps; trading-strategy evaluation with PSR/DSR/MinTRL, a trial registry, PBO-CSCV and a block-bootstrap Sharpe interval; LLM evals with paired + clustered errors, small-N intervals and confidence sequences for peeking; HTML reports, a Streamlit viewer, a generated calibration table and a docs site with method cards; the 0.1.0 release is prepared but not published). Try it:

Personal, local and unpublished: install from a clone (`uv sync`, or `uv tool install --editable .`). The package is marked `Private :: Do Not Upload`.

```bash
uv sync
uv run isbetter run examples/toy_paired.yaml
make check               # ruff + mypy + fast tests (what CI runs)
make calibrate           # slow calibration tests (10,000 replications)
make gallery             # all failure-gallery scripts
make cases               # all five case studies (synthetic stand-ins)
uv run python validation/failure_gallery/01_naive_cv_ttest.py   # naive CV t-test fails
uv run python case_studies/01_fraud_classifiers/run.py          # Phase 1 case study
```

Phase 2 (A/B testing):

```bash
uv run python case_studies/02_ab_hillstrom_criteo/run.py        # email test + 14M-row scale test
uv run isbetter run case_studies/02_ab_hillstrom_criteo/config.yaml   # SRM, Holm, guardrail, ship decision
uv run isbetter plan-size --kind proportions --baseline 0.10 --mde 0.01   # sample size / power / MDE
uv run python validation/failure_gallery/05_ignored_clustering.py       # ignoring user clustering fails
uv run pytest -m slow                                                   # + CUPED/z/SRM calibration, 14M-row memory
```

Phase 3 (forecasting):

```bash
uv run isbetter run examples/toy_forecast.yaml                           # DM + HLN on a 3-step backtest
uv run python case_studies/03_forecasting_m5/run.py                      # seasonal-naive vs ARIMA vs LightGBM
uv run python validation/failure_gallery/02b_dm_horizon.py               # naive t at h > 1 over-rejects
uv run pytest -m slow                                                    # + Diebold-Mariano calibration
```

`dependence: {type: timeseries, horizon: h}` routes per-origin losses to the Diebold-Mariano test
with the Harvey-Leybourne-Newbold correction (`h - 1` autocovariance lags) and a stationary block
bootstrap cross-check whose block length comes from Politis-White. `adapters/forecast_backtest.py`
turns rolling-origin forecasts into those losses (MAE / MASE / RMSSE per origin and horizon), and
`methods/superior_predictive.py` wraps `arch` for many models against a benchmark (the `[forecast]`
extra). Known limit: DM + HLN is well calibrated at h = 1 and 3 but still mildly liberal at long
horizons with few origins (about 6-7% at h = 7-10, T = 100); the naive t-test is far worse.

Phase 4 (trading, "edge or luck?"):

```bash
uv run python case_studies/04_trading_edge/run.py                        # 272 MA rules: naive t vs DSR vs PBO
uv run python validation/failure_gallery/02_iid_bootstrap_sharpe.py      # iid bootstrap Sharpe CI undercovers
uv run isbetter run examples/toy_strategies.yaml                         # same checks from a config
uv run pytest -m slow                                                    # + Sharpe-interval coverage
```

`adapters/strategy_returns.py` turns per-period returns into the canonical table and provides the
append-only `TrialRegistry` (every evaluated variant logged by a decorator or context manager).
`methods/finance.py` has the Probabilistic and Deflated Sharpe Ratio and MinTRL, a block-bootstrap
Sharpe interval, and SPA/StepM against buy-and-hold; the DSR refuses to run unless the number of
trials comes from a registry or is given explicitly. `methods/pbo.py` is the vectorised
combinatorially symmetric CV estimate of the Probability of Backtest Overfitting (12,870 splits
at 16 blocks in about a second). Known limits: the block-bootstrap Sharpe interval stays about
2-4 points short of nominal coverage at T = 500, and DSR treats the registry's trials as
independent although neighbouring variants are near-duplicates.

From a config (`adapter: strategy_returns`, `metric: {kind: returns}`, `dependence: timeseries`,
and a `finance:` block with a trial `registry` or `n_trials` + `sr_variance`), `isbetter run`
compares each strategy with the baseline (for example buy-and-hold) by a block-bootstrap interval
for the Sharpe ratio difference, and reports the PSR, DSR and MinTRL of each strategy and the PBO
of the whole set alongside. They add warnings (DSR below 0.95, PBO above 0.5, a track record
shorter than MinTRL) but never change the verdict. Sharpe ratios and `min_effect` are per period,
not annualised, and Holm/BH is refused here because the DSR is what accounts for the search.

Phase 5 (LLM evals and sequential testing):

```bash
uv run python validation/failure_gallery/03_clt_small_n.py               # normal-theory interval undercovers at N = 30-200
uv run python validation/failure_gallery/04_peeking.py                   # peeking inflates false positives to ~46%
uv run python case_studies/05_llm_eval/run.py                            # two models, 200 questions in 50 passages
uv run pytest -m slow                                                    # + small-N coverage, peeking calibration
```

`adapters/llm_eval.py` reads per-item JSON Lines (`model`, `item_id`, `score`, optional `cluster_id`)
into a table paired by item (`adapter: llm_eval`), and `ResponseCache` is an append-only cache of
model responses keyed by (model, prompt, item, seed) that can run offline; no model is run by the
package. `dependence: {type: clustered, paired: true}` analyses per-item differences with a
cluster-robust standard error (items on the same passage are correlated), and a paired right/wrong
metric (`kind: binary` under `paired`) gets a Tango score interval with McNemar's p-value, which
keeps its coverage at small N where the normal-theory interval does not (Clopper-Pearson, Wilson,
Jeffreys and Beta/Dirichlet posteriors are in `methods/small_n.py`). `methods/sequential.py` has
normal-mixture confidence sequences and always-valid p-values: an interval you may read after every
observation (about 2% false positives under continuous peeking, versus 46% for a fixed-horizon test).
Add `sequential: {n_target: 1000}` to a single-metric `iid` or `paired` config (rows need a `time`
column: arrival order) and `isbetter run` reports the confidence sequence and its always-valid
p-value as the primary result, with the fixed-horizon test beside it as the foil and a warning
when the two disagree (`examples/toy_stream.yaml`). `n_target` (or `rho`) must be fixed in advance.
Known limits: sequential monitoring is not yet wired into A/B experiments (SRM, guardrails),
clustered or time-series data; with plug-in variance the sequence is asymptotic and starts at 20
observations; with few clusters the CR1
interval is mildly liberal; the formulas of Miller (2024) and Bowyer et al. (2025) were implemented
from their published descriptions and checked with independent oracles, not against the papers' own
numbers (the clustered and paired intervals also agree with the third-party `errorbars` package),
and the paired interval is Tango's rather than that paper's own recipe.

An A/B config (`adapter: ab_table`, or a `metrics:` list) gets an automatic sample-ratio check,
multiplicity across success metrics only, one-sided non-inferiority guardrails and a ship /
no-ship decision per candidate. Large files can be aggregated in place with `table.mode: stats`.

Build plan:

➡️ **[docs/BLUEPRINT.md](docs/BLUEPRINT.md)**: architecture, data contract, method routing table, formulas, phased tickets with Definitions of Done, testing strategy and the 8 GB RAM rules.

## Planned phases

| Phase | Theme | Headline deliverable |
|---|---|---|
| 0 | Foundations | YAML → plan → test → verdict → Markdown report; A/A calibration |
| 1 | ML (credit-card fraud) | Nadeau–Bengio vs naive CV t-test failure gallery |
| 2 | A/B (Hillstrom, Criteo) | CUPED, SRM, guardrails; 14M rows in under 1 GB of RAM |
| 3 | Forecasting (M5) | Diebold–Mariano + HLN, MCS/SPA |
| 4 | Trading | Deflated Sharpe, PBO-CSCV, trial registry |
| 5 | LLM evals + sequential | Small-N intervals, confidence sequences (built) |
| 6 | Local polish | HTML report, calibration table, Markdown method notes, Makefile (built). No docs site, Streamlit viewer or PyPI release. |
