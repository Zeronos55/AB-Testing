# AB-Testing: `isbetter`

**Is A really better than B?** You tell it how your data are dependent, and it gives you a valid statistical test, a three-way verdict (better / equivalent / inconclusive) and proof that the test is calibrated.

`isbetter` is planned as a cross-domain comparison toolkit. It treats several kinds of comparison as the same problem: ML model comparisons, A/B experiments, forecast backtests, trading strategies and LLM evals. In each case you have paired or unpaired scores with a known dependence structure (`iid`, `paired`, `clustered`, `cv_overlap`, `timeseries`), and that structure determines which test is valid.

## Status

**Phases 0-1 built** (iid/paired/clustered/CV-overlap comparison, verdict with `n_needed`, Holm/BH, report, calibration harness, fraud case study). Try it:

```bash
uv sync
uv run isbetter run examples/toy_paired.yaml
uv run pytest            # fast tests
uv run pytest -m slow    # calibration (10,000 replications)
uv run python validation/failure_gallery/01_naive_cv_ttest.py   # naive CV t-test fails
uv run python case_studies/01_fraud_classifiers/run.py          # Phase 1 case study
```

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
