# Blueprint: `isbetter`, a dependence-aware "Is A really better than B?" toolkit

> Audience: junior engineers and interns joining the build. Read sections 1–5 before writing any code; after that, each phase in section 7 can be read on its own.
> Status: **Phases 0–4 built** (see README for current commands). Later phases are still planned; build them in the order given in section 12.

---

## 0. Context: why this exists

The repo (`zeronos55/ab-testing`) starts empty. The prior-art review found that every individual statistical method we need already exists, but each one lives in a library tied to a single domain:

- A/B testing: GrowthBook, Spotify Confidence, tea-tasting.
- ML model comparison: mlxtend, autorank, baycomp.
- Forecast comparison: statsmodels Diebold–Mariano, arch's SPA and MCS.
- Trading strategies: Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO) repos.
- LLM evals: errorbars, bayes_evals.

None of them treats **the dependence structure of the data** as the thing the user declares first. A toolkit that only "runs t-tests and CUPED" would copy existing work.

**What we build instead.** The thesis is one sentence: *declare how your data are dependent, and get a valid test, a three-way verdict, and proof that the test is calibrated.*

The three pillars:
1. **Dependence first.** Every analysis starts from a declared unit of independence: `iid`, `paired`, `clustered`, `cv_overlap` or `timeseries`. That declaration decides which method runs.
2. **Decision, not just a p-value.** The output is one of: better, equivalent, inconclusive or worse. The user supplies a minimum effect of interest δ, and the tool estimates how much more data is needed (`n_needed`).
3. **Validated in public.** A simulation harness and a "failure gallery" show that each method keeps its nominal Type I error and coverage, and that the naive alternative does not.

Hard constraint: **everything must run on an 8 GB laptop.**

---

## 1. Glossary (read this first if the statistics are new to you)

| Term | Plain-English meaning |
|---|---|
| **Variant** | One of the things being compared: model A vs model B, control vs treatment, strategy X vs buy-and-hold. |
| **Unit** | The thing one measurement comes from: a user, a test example, a time period, a CV fold. |
| **Paired** | A and B were measured on the *same* units, for example both models scored on the same test rows. Pairing removes shared noise, which makes the test more powerful. |
| **Clustered** | Units come in groups that are correlated with each other, such as several sessions per user or several questions per passage. If you ignore this, you are pretending you have more data than you do. |
| **CV overlap** | In k-fold cross-validation the training sets overlap, so the k fold scores are *not* independent. A naive t-test on them is too confident. |
| **Time-series dependence** | Today's loss is correlated with yesterday's. Plain bootstraps and plain standard errors are wrong here; use block bootstraps or HAC variance. |
| **Type I error** | Declaring "A is better" when it isn't (a false positive). It should equal α (for example 5%) when nothing is going on. |
| **Coverage** | How often a 95% confidence interval actually contains the true value. It should be about 95%. |
| **δ (min effect)** | The smallest difference the user actually cares about. Anything smaller counts as "practically equivalent". |
| **Oracle test** | A unit test that compares our hand-written estimator with a trusted reference implementation (SciPy, statsmodels, arch, stored R output). |
| **Calibration test** | A slow test that simulates thousands of datasets where the truth is known and checks that Type I error and coverage come out at their nominal values. |
| **Sufficient statistics** | A few numbers per group, such as n, Σx, Σx² and Σxy, that are enough to compute a mean-based test. This is how we handle 14 million rows using kilobytes of memory. |

---

## 2. Guiding engineering principles (non-negotiable)

1. **The stats core is pure.** `core/`, `methods/` and `decision/` do no file I/O, no plotting and no printing. They take arrays or dataclasses in and return dataclasses out. This keeps them testable.
2. **Every random call takes a `numpy.random.Generator`.** Never call `np.random.seed` or use global state. The seed is recorded in every `Result`.
3. **Every hand-written estimator has an oracle test** before it is merged. No exceptions. (A 2026 package paper found a Diebold–Mariano variance bug that only showed up at forecast horizon h>1. Oracle tests catch exactly that kind of bug.)
4. **Memory-aware by default.**
   - Bootstraps run in batches; never build an `n × B` index matrix.
   - Large tables are aggregated in DuckDB or polars before they reach NumPy.
   - Use float32 or int8 where precision allows.
5. **Explain every choice.** Each method returns a `rationale: str`, for example "Used Nadeau–Bengio because dependence=cv_overlap; naive t-test would understate variance". The report prints it.
6. **Wrap rather than rewrite the hard things:** SciPy BCa, arch SPA/StepM/MCS, baycomp. Implement the small, checkable things ourselves (listed in section 6).
7. **Small PRs.** One estimator, plus its oracle test, plus its docstring, per PR.

---

## 3. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python ≥ 3.11 | Ecosystem; `dataclass(slots=True)`; faster interpreter. |
| Packaging | `uv` + `pyproject.toml` (hatchling backend) | Fast and reproducible locks. |
| Numerics | `numpy`, `scipy` | Core maths and reference implementations. |
| Tables | `polars` (primary), `duckdb` (SQL aggregation), `pandas` (adapter interop only) | Lazy scans and low memory. |
| Validation | `pydantic` v2 | Config and schema validation with readable errors. |
| CLI | `typer` | Typed CLI with little code. |
| Reports | `jinja2` + `matplotlib` | Static Markdown/HTML; no server needed. |
| Tests | `pytest`, `hypothesis`, `pytest-xdist` | Unit, property and parallel calibration tests. |
| Quality | `ruff` (lint + format), `mypy --strict` on `core/`, `methods/` and `decision/` | Catches bugs early. |
| Optional extras | `[ab]` duckdb · `[ml]` scikit-learn, lightgbm · `[forecast]` statsmodels, statsforecast, arch · `[finance]` arch · `[bayes]` baycomp · `[app]` streamlit · `[docs]` mkdocs-material | Keeps the base install light. |

---

## 4. Architecture

### 4.1 Data flow

```
 domain data ──► Adapter ──► CanonicalTable ──► AnalysisPlan ──► Method(s) ──► MethodResult(s)
 (CV scores,     (adapters/)  (core/schema)     (core/plan:      (methods/)        │
  A/B table,                                     routing by                          ▼
  returns, ...)                                  Dependence)                  Verdict engine
                                                      ▲                       (decision/)
                                     YAML config ─────┘                              │
                                                                                     ▼
                                                                Result (JSON) ──► Report (MD/HTML)
                                                                                 └─► Streamlit viewer
```

### 4.2 The canonical data contract (`core/schema.py`)

Every adapter outputs one long table with these columns:

| column | type | required | notes |
|---|---|---|---|
| `unit_id` | str/int | yes | Row, user, period or fold identifier. |
| `variant` | str (categorical) | yes | For example `"logreg"` or `"lightgbm"`. |
| `metric` | str (categorical) | yes | For example `"pr_auc"` or `"conversion"`. |
| `value` | float64 | yes | One observation. |
| `pair_id` | str/int | if paired | Links A and B rows measured on the same unit. |
| `cluster_id` | str/int | if clustered | User, question group and so on. |
| `time` | datetime/int | if timeseries | Must be strictly increasing within each variant. |
| `fold`, `repeat` | int | if cv_overlap | CV bookkeeping. |
| `cov_*` | float | optional | Pre-period covariates used by CUPED. |

Validation rules, which must **fail loudly** with a clear message:
- a declared pairing has a missing partner;
- duplicate `(variant, metric, pair_id)` rows;
- a `time` column that is not monotone within a variant;
- fewer than 2 clusters per variant;
- NaN in `value`.

Large A/B tables skip this path and use the `SufficientStats` dataclass instead (`n`, `sum`, `sum_sq` and, optionally, `sum_xy`, `sum_x`, `sum_x_sq` per arm).

### 4.3 The `Dependence` declaration (`core/dependence.py`)

```python
@dataclass(frozen=True) class Iid: ...
@dataclass(frozen=True) class Paired: ...
@dataclass(frozen=True) class Clustered:   cluster_col: str = "cluster_id"
@dataclass(frozen=True) class CVOverlap:   n_train: int; n_test: int; repeats: int; folds: int
@dataclass(frozen=True) class TimeSeries:  block_len: int | None = None; hac_lags: int | None = None; horizon: int = 1
Dependence = Iid | Paired | Clustered | CVOverlap | TimeSeries
```

### 4.4 Method routing table (the heart of `core/plan.py`)

| Dependence | Metric kind | Primary method | Secondary / cross-check | Phase |
|---|---|---|---|---|
| `Iid` | continuous | Welch t | permutation test; BCa bootstrap | 0 |
| `Iid` | binary | two-proportion z (Newcombe CI for small n) | Fisher exact | 0/2 |
| `Iid` | ratio (e.g. revenue/visit) | delta method | bootstrap | 2 |
| `Iid` + covariate | any mean | CUPED / regression adjustment | unadjusted Welch (for comparison) | 2 |
| `Paired` | continuous | paired t | paired bootstrap; sign-flip permutation | 0 |
| `Paired` | binary per item | McNemar (exact if b+c < 25, else χ² with continuity correction) | paired bootstrap | 1 |
| `Paired` | small-N proportion (LLM) | Wilson / Jeffreys on each arm + paired-difference CI | Bayesian Beta posterior | 5 |
| `Clustered` | any | cluster-robust SE (CR1) with t(G−1) | cluster bootstrap | 1/5 |
| `CVOverlap` | per-fold score | **Nadeau–Bengio corrected resampled t** | 5×2cv paired t; Bayesian correlated t (baycomp) | 1 |
| `TimeSeries` | loss differential | Diebold–Mariano + HLN correction | stationary/circular block bootstrap | 3 |
| `TimeSeries` | many models vs benchmark | Hansen SPA, Romano–Wolf StepM, MCS (wrap `arch`) | — | 3/4 |
| `TimeSeries` | Sharpe ratio | PSR / DSR + block-bootstrap Sharpe CI | PBO-CSCV across the grid | 4 |
| sequential (any) | mean | normal-mixture confidence sequence | fixed-horizon test, shown as the "peeking" foil | 5 |

When a combination is not in the table, the router raises `UnsupportedPlan` and explains what is missing. **It must never silently fall back to a naive test.**

### 4.5 Result objects (`core/result.py`)

```python
@dataclass class MethodResult:
    method: str; estimate: float; ci_low: float; ci_high: float; ci_level: float
    p_value: float | None; statistic: float | None; df: float | None
    n_effective: float; rationale: str; diagnostics: dict[str, Any]

@dataclass class Verdict:
    label: Literal["A_better_practical","A_better_maybe_trivial","equivalent",
                   "inconclusive","B_better_maybe_trivial","B_better_practical"]
    explanation: str; n_needed: float | None; n_needed_note: str | None

@dataclass class Result:
    experiment: str; plan: AnalysisPlan; comparisons: list[ComparisonResult]
    seed: int; package_version: str; created_at: str
    def to_json(self) -> str: ...
```

`ComparisonResult` holds one (baseline, candidate, metric) triple. It contains the primary `MethodResult`, the cross-checks, and the `Verdict`.

---

## 5. Repository layout

```
ab-testing/
├── pyproject.toml            README.md            LICENSE (MIT)
├── docs/BLUEPRINT.md         # this document, committed to the repo
├── src/isbetter/
│   ├── core/       schema.py  dependence.py  plan.py  result.py  config.py
│   ├── methods/    parametric.py  proportions.py  classification.py  resampling.py
│   │               cluster.py  cv.py  forecasting.py  superior_predictive.py  finance.py
│   │               variance_reduction.py  multiplicity.py  power.py  sequential.py
│   ├── decision/   verdict.py  guardrails.py
│   ├── adapters/   sklearn_cv.py  ab_table.py  forecast_backtest.py  strategy_returns.py  llm_eval.py
│   ├── simulate/   dgp.py  calibration.py
│   ├── report/     markdown.py  html.py  plots.py  templates/
│   ├── cli.py
│   └── app/streamlit_app.py
├── tests/  unit/  oracle/ (+ fixtures/)  property/  calibration/  (marked @slow)
├── case_studies/  01_fraud_classifiers/  02_ab_hillstrom_criteo/  03_forecasting_m5/
│                  04_trading_edge/  05_llm_eval/
├── validation/failure_gallery/   # 5 "naive method breaks" scripts + figures
├── data/          # .gitignored; download scripts only
└── .github/workflows/  ci.yml (fast)   nightly.yml (calibration)
```

Note: the package is called `methods/`, not `tests_stats/`. That avoids confusion with `tests/`, and every file in it is named after the statistics it implements.

---

## 6. Build vs wrap

**We implement these ourselves.** Each one is small, has a closed form and an oracle test:
- Welch / paired t
- Cohen's d / Hedges' g
- Wilson / Jeffreys / Newcombe intervals
- McNemar
- Nadeau–Bengio
- Bonferroni / Holm / Benjamini–Hochberg
- CUPED and delta method
- batched paired / cluster / Poisson / block bootstrap
- sign-flip permutation
- cluster-robust SE
- Diebold–Mariano + HLN
- PSR / DSR / MinTRL
- PBO-CSCV
- `n_needed`
- simulation-based power
- a confidence sequence

**We wrap these:**
- `scipy.stats.bootstrap` (BCa)
- `scipy.stats.permutation_test` (used as an oracle)
- `statsmodels` `multipletests` (oracle) and its power solvers
- `arch.bootstrap` SPA / StepM / MCS and its block bootstraps
- `baycomp` (optional Bayesian view)
- `autorank` (optional CD diagrams)

### Formula cheat-sheet (implementers: put the formula in the docstring and cite the paper)

| Method | Formula |
|---|---|
| Welch t | t = (x̄_A − x̄_B)/√(s²_A/n_A + s²_B/n_B), with Welch–Satterthwaite df. |
| Nadeau–Bengio | d_j = per-fold score difference, j = 1..rk. t = d̄ / √((1/(rk) + n_test/n_train)·s²_d), df = rk − 1. Valid but **conservative**. |
| McNemar | b = A right & B wrong, c = A wrong & B right. Exact: p = 2·BinomCDF(min(b,c); b+c, 0.5), capped at 1. χ² = (\|b−c\|−1)²/(b+c). |
| Wilson | (p̂ + z²/2n ± z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n). |
| Jeffreys | Beta(x+½, n−x+½) quantiles. |
| CUPED | θ = cov(Y,X)/var(X), pooled across arms. Y′ = Y − θ(X − X̄). Variance shrinks by a factor of (1 − ρ²). |
| Delta method (ratio Ȳ/X̄) | Var ≈ (1/X̄²)·(Var Ȳ − 2R·Cov(Ȳ,X̄) + R²·Var X̄), where R = Ȳ/X̄. |
| Diebold–Mariano + HLN | d_t = L(e_A,t) − L(e_B,t). DM = d̄ / √(LRV/T), where LRV uses autocovariances up to lag h−1 **divided by T, not T−lag**. DM_HLN = √((T+1−2h+h(h−1)/T)/T)·DM, compared against t(T−1). |
| PSR | PSR(SR*) = Φ( (ŜR − SR*)·√(T−1) / √(1 − γ₃·ŜR + (γ₄−1)/4·ŜR²) ). γ₃ = skew, γ₄ = raw kurtosis; per-period SR. |
| DSR | PSR evaluated at SR* = √V[SR_trials]·((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))), where γ ≈ 0.5772 (Euler–Mascheroni) and N = number of trials **from the trial registry**. |
| MinTRL | 1 + (1 − γ₃·ŜR + (γ₄−1)/4·ŜR²)·(z_α/(ŜR − SR*))². |
| PBO-CSCV | Split T into S blocks. For each of the C(S, S/2) splits: take the IS-best config, find its OOS relative rank ω, compute λ = ln(ω/(1−ω)). PBO = share of splits with λ ≤ 0. |
| Holm | Sort the p-values. Reject p₍ᵢ₎ while p₍ᵢ₎ ≤ α/(m−i+1); stop at the first failure. |
| Benjamini–Hochberg | Find the largest k with p₍ₖ₎ ≤ kα/m; reject hypotheses 1..k. |
| SRM | χ² goodness-of-fit of the arm counts against the planned split. Flag if p < 0.001. |

### 6.1 Verdict logic (`decision/verdict.py`)

Inputs: a CI [L, U] for Δ = A − B, oriented so that positive means better; the margin δ > 0.

| Condition | Label |
|---|---|
| L > +δ | `A_better_practical` |
| L > 0 and L ≤ δ | `A_better_maybe_trivial` |
| −δ < L and U < +δ | `equivalent` (a TOST/ROPE-style check using the 1−2α CI) |
| U < −δ | `B_better_practical` |
| U < 0 and U ≥ −δ | `B_better_maybe_trivial` |
| anything else | `inconclusive`, with `n_needed` |

How `n_needed` works:
- A CI half-width h scales like 1/√n. Let h* be the half-width needed for the CI to clear the nearest decision boundary (0, +δ or −δ). Then `n_needed ≈ n·(h/h*)²`.
- **Special case for CVOverlap.** The variance factor is (1/(rk) + n_test/n_train). If the n_test/n_train term already accounts for more than 80% of it, set `n_needed_note = "more CV repeats won't help — you need more data"`. Example: 10-fold CV gives n_test/n_train = 1/9 ≈ 0.111, while 1/(rk) is only 0.01 at r = 10.

Guardrails (`decision/guardrails.py`, following Spotify's decision rules):
- Success metrics use superiority tests, with multiplicity correction across the success metrics only.
- Guardrail metrics use non-inferiority tests at unadjusted α.
- The decision is "ship" only if at least one success metric wins and no guardrail fails.

---

## 7. Phased build plan

Every phase ends with a project that can be shown on its own. Tickets are sized at about 0.5–2 days for a junior engineer. **DoD** means Definition of Done.

### Phase 0: Foundations and one honest paired comparison

**Goal:** a correct end-to-end pipeline for `Iid` and `Paired`, from YAML config to Markdown report.

| # | Ticket | Files | DoD |
|---|---|---|---|
| 0.1 | Scaffold: `pyproject` with extras, ruff, mypy, pytest config, `uv.lock`, MIT license, CI workflow (lint, types, fast tests on 3.11 and 3.12) | `pyproject.toml`, `.github/workflows/ci.yml` | `uv run pytest` and `uv run ruff check` pass in CI. |
| 0.2 | Canonical schema + validation errors | `core/schema.py` | Unit tests cover each failure rule in 4.2. |
| 0.3 | `Dependence` types + YAML config model | `core/dependence.py`, `core/config.py` | The example config in section 9 parses; a bad config gives a readable pydantic error. |
| 0.4 | Welch t, paired t, Cohen's d / Hedges' g, with CIs | `methods/parametric.py` | Oracle tests against `scipy.stats.ttest_ind(equal_var=False)` and `ttest_rel`, tolerance 1e-8. |
| 0.5 | Batched paired bootstrap (percentile + BCa via SciPy) and sign-flip permutation | `methods/resampling.py` | Memory stays O(batch·n) (assert peak with `tracemalloc` at n = 300k, B = 9,999, batch = 200: < 600 MB). Permutation p-value matches `scipy.stats.permutation_test` within Monte Carlo error. |
| 0.6 | Router v1 and `AnalysisPlan` pretty-print | `core/plan.py` | The plan prints *before* running; an unsupported combination raises `UnsupportedPlan`. |
| 0.7 | `Result` / `Verdict` dataclasses + JSON round-trip + verdict engine | `core/result.py`, `decision/verdict.py` | Property test: swapping A and B flips the sign of the estimate and mirrors the label. Table-driven test covers all 6 verdict cells. |
| 0.8 | Markdown report (Jinja2) + CLI `isbetter run config.yaml` | `report/markdown.py`, `cli.py` | Running the CLI on a toy CSV produces a report with the estimate, CI, p-value, verdict and rationale. |
| 0.9 | Simulation harness v1: an A/A Type I error check for Welch and paired t | `simulate/dgp.py`, `simulate/calibration.py`, `tests/calibration/` | 10,000 vectorised replications at α = 0.05 give a rejection rate within 0.0456–0.0544. |

**Skills learned:** hypothesis testing, typed Python, test discipline.

### Phase 1: ML case study, fraud classifiers (ULB credit-card fraud, 284,807 rows)

**Goal:** the demo a recruiter understands immediately. Logistic regression vs LightGBM vs random forest, compared on PR-AUC (not accuracy, because only 0.172% of rows are fraud).

| # | Ticket | DoD |
|---|---|---|
| 1.1 | `adapters/sklearn_cv.py`: run a repeated stratified k-fold over several estimators and emit per-fold scores *and* per-row out-of-fold predictions | Output passes the schema; seeds are recorded. |
| 1.2 | `methods/cv.py`: Nadeau–Bengio, plus mlxtend-style 5×2cv t and naive k-fold t (the naive one is kept *only* for the gallery) | Oracle: a hand-computed fixture plus mlxtend `paired_ttest_5x2cv` on the iris example. |
| 1.3 | `methods/classification.py`: McNemar (exact + χ²) | Oracle against `statsmodels.stats.contingency_tables.mcnemar`. |
| 1.4 | `methods/multiplicity.py`: Bonferroni, Holm, BH (with adjusted p-values) | Oracle against `statsmodels.stats.multitest.multipletests` on 1,000 random p-vectors. |
| 1.5 | `methods/power.py`: `n_needed`, including the CV-specific note | Unit test with a known half-width ratio. |
| 1.6 | `methods/cluster.py`: CR1 cluster-robust SE + cluster bootstrap | Oracle against statsmodels OLS with `cov_type="cluster"`. |
| 1.7 | Failure-gallery #1: naive CV t-test vs Nadeau–Bengio Type I error on a null DGP (two equal learners) | Figure shows the naive test's inflated false-positive rate against nominal 5%. |
| 1.8 | Case study `01_fraud_classifiers/`: `config.yaml`, `run.py`, `REPORT.md`, README written to the template in section 10 | Report shows Δ PR-AUC with a CI, a verdict, `n_needed` and the Holm-adjusted family. |

### Phase 2: A/B testing (Hillstrom first, then Criteo)

| # | Ticket | DoD |
|---|---|---|
| 2.1 | `adapters/ab_table.py`: unit-level path **and** a `SufficientStats` path built with DuckDB/polars lazy scans | Criteo (3 columns, about 14M rows) aggregates with peak RSS < 1 GB. |
| 2.2 | `methods/proportions.py`: two-proportion z, Newcombe, Wilson, Jeffreys, Fisher | Oracle against `statsmodels.stats.proportion`. |
| 2.3 | `methods/variance_reduction.py`: CUPED (θ pooled), regression adjustment, delta method for ratios | Oracle: CUPED estimate equals the OLS coefficient on treatment with a centred covariate. A calibration test shows CUPED keeps Type I error at nominal. |
| 2.4 | SRM check, run automatically on every A/B plan | Hillstrom arm counts (21,307 / 21,387 / 21,306) pass; a planted 50.5/49.5 split at n = 1M fails. |
| 2.5 | `decision/guardrails.py`: success/guardrail decision rule across the 3 Hillstrom arms | Table-driven tests cover ship / no-ship combinations. |
| 2.6 | Power and sample-size planner (`isbetter plan-size ...`) | Matches `statsmodels.stats.power` within 1%. |
| 2.7 | Failure-gallery #5: ignoring user-level clustering inflates Type I error | Figure produced. |
| 2.8 | Case study `02_ab_hillstrom_criteo/` | Report shows CI width with and without CUPED and the % variance reduction. Criteo first verifies the dataset URL and checks the "rows are impressions, not users" caveat. |

### Phase 3: Forecasting with rolling-origin backtests (M5 subset)

| # | Ticket | DoD |
|---|---|---|
| 3.1 | `adapters/forecast_backtest.py`: per-origin, per-horizon losses (MAE, MASE, RMSSE) | Schema passes; `time` is monotone. |
| 3.2 | `methods/forecasting.py`: DM + HLN | Oracle against statsmodels (if the installed version has `diebold_mariano_test`) **and** stored R `forecast::dm.test` fixtures at h = 1, 3 and 7. *Built:* statsmodels 0.15 has no DM test, so the oracles are statsmodels' HAC covariance (uniform and Bartlett kernels) and a transcription of `dm.test`; the R fixture test is in place but skips until `tests/oracle/fixtures/make_dm_fixtures.R` has been run where R is available. |
| 3.3 | Wrap `arch` SPA / StepM / MCS behind our `MethodResult` | Wrapper tests on the documented `arch` example. |
| 3.4 | Block-length suggestion diagnostic (Politis–White, via `arch.bootstrap.optimal_block_length`) | Appears in the report diagnostics. |
| 3.5 | Failure-gallery #2b: a DM test with h > 1 that ignores autocorrelation over-rejects | Figure produced. |
| 3.6 | Case study `03_forecasting_m5/`: seasonal-naive vs ARIMA vs LightGBM on one store or on the 70 store × department series | Runs in < 4 GB; the MCS result is explained in plain English. |

### Phase 4: Trading, "edge or luck?"

| # | Ticket | DoD |
|---|---|---|
| 4.1 | `adapters/strategy_returns.py` + `TrialRegistry` (append-only JSONL; every evaluated variant is logged automatically through a decorator/context manager) | DSR refuses to run without a registry or an explicit `n_trials`. *Built:* `resolve_deflation` raises `MissingTrialsError` without either; `deflated_sharpe` has no default for `n_trials` / `sr_variance`. The same variant logged twice counts once. |
| 4.2 | `methods/finance.py`: PSR, DSR, MinTRL | Oracle against the published worked examples (Bailey & López de Prado 2012 and 2014). *Built:* the papers' worked examples could not be fetched where this was built, so the oracles are independent of the implementation instead: quadrature, simulation and the tabulated expected maximum of N normals (Harter) for `E[max SR]`; a Monte Carlo check of the Sharpe-variance formula; PSR coverage; and the exact `PSR(MinTRL) = 1 − α` identity. Add the paper fixtures when they can be transcribed. The closed-form `E[max SR]` is a known approximation: about 8% low at N = 2 and 0.5–3% high for N ≥ 3. |
| 4.3 | PBO-CSCV, vectorised (S = 16 gives 12,870 splits) | On pure noise, PBO ≈ 0.5; with one planted strong edge, PBO is low. Runs in < 30 s. *Built:* `methods/pbo.py`, checked against a literal loop transcription; S = 16 on 2,500 periods × 400 configurations takes about 1 s. One dataset's PBO on noise has sd ≈ 0.14 (the splits overlap); its mean over datasets is 0.5. |
| 4.4 | Block-bootstrap Sharpe CI + SPA/StepM versus buy-and-hold | Coverage test on AR(1)-GARCH returns. *Built:* at T = 500, φ = 0.2 the block-bootstrap interval covers about 91–93% at nominal 95% (iid: about 86–89%); it is better than iid, not exact. |
| 4.5 | Failure-gallery #2: iid vs block bootstrap Sharpe CI coverage | Figure produced. |
| 4.6 | Case study `04_trading_edge/`: a few hundred moving-average variants on daily ETF returns | Report contrasts the naive t-stat of the best variant with its DSR and PBO. |

### Phase 5 (stretch): LLM evals and sequential testing

| # | Ticket | DoD |
|---|---|---|
| 5.1 | `adapters/llm_eval.py`: per-item JSONL (model, item_id, cluster_id, score), with response caching keyed by (model, prompt, item, seed) | Uses public per-item results or cached API outputs; **no local models.** *Built:* `adapters/llm_eval.py` reads per-item JSONL (`model`, `item_id`, `score`, optional `cluster_id`, `metric`) into a table paired by item; `ResponseCache` is an append-only JSONL keyed by SHA-256 of (model, prompt, item, seed), stores a prompt hash rather than the prompt, and in `offline` mode raises `CacheMiss` instead of calling. No model is run or imported. `adapter: llm_eval` works from the CLI. |
| 5.2 | Paired + clustered SEs following Miller (2024); Wilson/Jeffreys/Beta-posterior intervals for small N following Bowyer et al. (2025) | Oracle on the papers' worked numbers where they are available. *Built:* the papers could not be fetched where this was built, so the oracles are independent of the implementations (as in 3.2 and 4.2). `paired_cluster_robust` (`dependence: {type: clustered, paired: true}`) is the CR1 standard error of the per-item differences with t(G − 1); it matches statsmodels' cluster covariance to 1e-12, reduces exactly to the paired t-test when every item is its own cluster, and equals Miller's `SE_CLT² + cross-terms` form (transcribed as triple loops) times G/(G − 1). Paired right/wrong scores (`kind: binary`) route to `paired_binary_score`: a Tango (1998) score interval for the accuracy difference plus McNemar's p-value, with per-arm Wilson and Jeffreys intervals in the diagnostics and a Dirichlet-posterior cross-check (`paired_beta_posterior`). Its oracles: the restricted MLE against a numeric optimiser, the score statistic at δ = 0 equal to McNemar's uncorrected χ² (statsmodels), the endpoints solving the score equation, and zero inside the interval iff that test does not reject. Also `clopper_pearson_interval` (SciPy exact) and `beta_posterior_interval`. This is **not** Bowyer et al.'s own paired recipe, which was not available to check; Tango's interval was chosen because it can be verified independently. At n = 60, accuracies 0.95 / 0.90 the score interval covers 96.1% and the normal-theory paired interval 91.1%. With 30 clusters the paired t that ignores clusters rejects 26% of true nulls; CR1 rejects 4.9%. |
| 5.3 | Failure-gallery #3: CLT interval coverage at N = 30–200 | Reproduces the qualitative under-coverage finding. *Built:* `validation/failure_gallery/03_clt_small_n.py` computes single-arm coverage exactly (the correct count is binomial, so no simulation) for the normal-theory, Wilson, Jeffreys and Clopper–Pearson intervals at N = 30, 50, 100, 200, plus simulated coverage of the paired difference. At N = 30 the normal interval covers 78–81% at accuracy 0.90–0.95 (45% at its worst accuracy); Wilson stays above 0.91 and Clopper–Pearson above 0.95. Coverage of a discrete count is a sawtooth, so no interval is exact everywhere (Jeffreys dips to about 0.89). |
| 5.4 | `methods/sequential.py`: normal-mixture confidence sequence | Under continuous peeking, the false positive rate stays ≤ α. |
| 5.5 | Failure-gallery #4: peeking at a fixed-horizon test | Figure shows inflation to well above 5%. |

### Phase 6: Polish and distribution

- mkdocs site with **method cards**: when to use, assumptions, failure modes, references.
- HTML report.
- Streamlit viewer that reads only saved `Result` JSON.
- PyPI release (`isbetter` 0.1.0).
- README with the calibration table.
- A blog write-up of the failure gallery.

---

## 8. Testing strategy (how juniors should write tests)

| Layer | Folder | Speed | What it checks | Example |
|---|---|---|---|---|
| Unit | `tests/unit/` | ms | Edge cases, input validation, shapes. | n = 1 raises; NaN raises; zero variance gives an explicit error. |
| Oracle | `tests/oracle/` | ms–s | Our numbers match the reference to 1e-8 (closed form) or within Monte Carlo SE (resampling). | `welch_t` vs `scipy.stats.ttest_ind(equal_var=False)`. |
| Property | `tests/property/` | s | Invariances, using `hypothesis`. | Swap A/B flips the sign; adding a constant to both arms leaves p unchanged; scaling by c>0 scales the estimate by c. |
| Calibration | `tests/calibration/` (`@pytest.mark.slow`) | min | Type I error and coverage within Monte Carlo bands. | 10k replications nightly; 2k locally. |

The Monte Carlo band for a rejection rate is α ± 2·√(α(1−α)/R). That gives ±0.0044 at R = 10,000 and ±0.0097 at R = 2,000.

CI setup:
- `ci.yml` runs lint, mypy, unit, oracle and property tests on every PR.
- `nightly.yml` runs the calibration tests and the failure-gallery scripts, and uploads the figures as artifacts.

---

## 9. Example config (the user-facing shape)

```yaml
experiment: fraud_models_v1
adapter: sklearn_cv            # sklearn_cv | ab_table | forecast_backtest | strategy_returns | llm_eval
data: results/fraud_cv_scores.parquet
variants: {baseline: logreg, candidates: [lightgbm, rf]}
metric: {name: pr_auc, higher_is_better: true}
dependence: {type: cv_overlap, n_train: 255000, n_test: 28000, repeats: 10, folds: 10}
decision: {min_effect: 0.01, alpha: 0.05, correction: holm}
seed: 20261010
report: {format: [markdown, html], include_simulation_check: true}
```

---

## 10. Case-study write-up template

1. Question.
2. Data and dependence structure.
3. Naive analysis, and why it misleads.
4. Correct analysis.
5. Verdict, effect size in natural units, CI and `n_needed`.
6. What would change the conclusion.

---

## 11. 8 GB RAM rules of thumb

- **Criteo:** select `treatment, visit, conversion` only, cast to int8 (about 42 MB), and aggregate to sufficient statistics.
- **Bootstraps:**
  - Always pass `batch=100..200`. The default `batch` on 284,807 rows would allocate about 23 GB.
  - Bootstrap the *difference vector*, not both arms.
  - Prefer Poisson(1) weights in float32.
- **M5:** use one store, or the store × department aggregation; downcast to int16/float32 with categorical IDs.
- **Calibration:** vectorise across replications by simulating an `R × n` matrix in chunks of R.
- **Streamlit:** never loads raw data.

---

## 12. What happens on "build" (execution plan once the lead gives the command)

1. **This approval step:** commit this blueprint to the repo as `docs/BLUEPRINT.md` and expand `README.md` with a pitch and a link to it. Push to a feature branch (not `main`). **No code yet.**
2. **On the build command:** run Phase 0 tickets 0.1 → 0.9 in order on the same branch, with one commit per ticket. Then report back with test output.
3. Every later phase starts only on an explicit command, so the lead can review each phase first.

### Verification for Phase 0 (once built)

- `uv sync --all-extras && uv run ruff check . && uv run mypy src/isbetter/core src/isbetter/methods src/isbetter/decision`
- `uv run pytest -m "not slow"`: all unit, oracle and property tests pass.
- `uv run pytest -m slow -k type1 --reps 2000`: the A/A rejection rate is inside the band.
- `uv run isbetter run examples/toy_paired.yaml` produces `REPORT.md` with the plan, estimate, CI, verdict and rationale.
