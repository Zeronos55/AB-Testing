# AB-Testing: `isbetter`

**Is A really better than B?** You tell it how your data are dependent, and it gives you a valid statistical test, a three-way verdict (better / equivalent / inconclusive) and proof that the test is calibrated.

`isbetter` is planned as a cross-domain comparison toolkit. It treats several kinds of comparison as the same problem: ML model comparisons, A/B experiments, forecast backtests, trading strategies and LLM evals. In each case you have paired or unpaired scores with a known dependence structure (`iid`, `paired`, `clustered`, `cv_overlap`, `timeseries`), and that structure determines which test is valid.

## Status

**Phases 0-5 built** (iid/paired/clustered/CV-overlap/time-series comparison, verdict with `n_needed`, Holm/BH, report, calibration harness, fraud case study; A/B testing with proportions, CUPED, ratio metrics, automatic SRM check, guardrail decisions, power planning, 14M-row aggregation; forecast backtests with Diebold-Mariano, SPA/StepM/MCS and block bootstraps; trading-strategy evaluation with PSR/DSR/MinTRL, a trial registry, PBO-CSCV and a block-bootstrap Sharpe interval; LLM evals with paired + clustered errors, small-N intervals and confidence sequences for peeking). Try it:

```bash
uv sync
uv run isbetter run examples/toy_paired.yaml
uv run pytest            # fast tests
uv run pytest -m slow    # calibration (10,000 replications)
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

Phase 5 (LLM evals and sequential testing):

```bash
uv run python validation/failure_gallery/03_clt_small_n.py               # normal-theory interval undercovers at N = 30-200
uv run python validation/failure_gallery/04_peeking.py                   # peeking inflates false positives to ~46%
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
Known limits: the sequential methods are a Python API with no `config.yaml` route yet; with plug-in
variance the sequence is asymptotic and starts at 20 observations; with few clusters the CR1
interval is mildly liberal; the formulas of Miller (2024) and Bowyer et al. (2025) were implemented
from their published descriptions and checked with independent oracles, not against the papers' own
numbers, and the paired interval is Tango's rather than that paper's own recipe.

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
| 6 | Polish | Docs site, Streamlit viewer, PyPI release |
