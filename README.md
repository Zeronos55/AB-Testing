# AB-Testing: `isbetter`

**Is A really better than B?** You tell it how your data are dependent, and it gives you a valid statistical test, a three-way verdict (better / equivalent / inconclusive) and proof that the test is calibrated.

`isbetter` is planned as a cross-domain comparison toolkit. It treats several kinds of comparison as the same problem: ML model comparisons, A/B experiments, forecast backtests, trading strategies and LLM evals. In each case you have paired or unpaired scores with a known dependence structure (`iid`, `paired`, `clustered`, `cv_overlap`, `timeseries`), and that structure determines which test is valid.

## Status

**Phases 0-2 built** (iid/paired/clustered/CV-overlap comparison, verdict with `n_needed`, Holm/BH, report, calibration harness, fraud case study; A/B testing with proportions, CUPED, ratio metrics, automatic SRM check, guardrail decisions, power planning, 14M-row aggregation). Try it:

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
| 5 | LLM evals + sequential | Small-N intervals, confidence sequences |
| 6 | Polish | Docs site, Streamlit viewer, PyPI release |
