# CLAUDE.md

`isbetter`: a dependence-aware "is A really better than B?" toolkit. The user declares
the dependence structure (`iid`, `paired`, `clustered`, `cv_overlap`, `timeseries`);
that picks the valid test, returns a three-way verdict and shows the test is calibrated.
Full design, method routing, formulas and phase tickets: `docs/BLUEPRINT.md`. Read the
relevant phase there before starting a ticket.

## Commands (mirror CI)
```bash
uv sync
uv run ruff check .
uv run mypy src/isbetter/core src/isbetter/methods src/isbetter/decision   # strict
uv run pytest                     # fast tests (slow deselected by default)
uv run pytest -m slow             # calibration, 10k replications; nightly in CI
uv run isbetter run examples/toy_paired.yaml
```
Run ruff, mypy and fast pytest before every commit. CI runs Python 3.11 and 3.12.

## Layout
- `core/` config, schema, plan, runner, result, dependence · `methods/` estimators ·
  `decision/` verdict · `adapters/` input loaders · `report/` Markdown output ·
  `simulate/` DGPs and the calibration harness
- `tests/{unit,oracle,property,calibration}`; `validation/failure_gallery/`;
  `case_studies/`

## Rules (non-negotiable)
1. `core/`, `methods/`, `decision/` are pure: no file I/O, plotting or printing.
2. Randomness goes through a passed-in `numpy.random.Generator`. Never
   `np.random.seed` or global state. Record the seed in `Result`.
3. Every hand-written estimator ships with an oracle test against a trusted reference
   (SciPy, statsmodels, mlxtend, arch). Wrap hard methods rather than rewriting them.
4. Memory: must run on 8 GB. Batch bootstraps (never build an n×B index matrix),
   aggregate big tables in DuckDB/polars first.
5. Each method returns a `rationale` string explaining why it was chosen; the report prints it.
6. mypy strict on core/methods/decision; ruff line length 100.

## Workflow
- Small changes: one estimator + oracle test + docstring.
- A statistics change is not done until its oracle test passes. If it affects Type I
  error or coverage, run `-m slow` too.
- Don't weaken a test tolerance to make it pass; find the cause.
- Don't commit generated outputs (`data/`, `case_studies/*/out/`).

## Gotchas
- `case_studies/01_fraud_classifiers` uses a synthetic stand-in, not real data.
- Optional extras (`ab`, `ml`, `forecast`, ...) are not in the base install; dev tests
  need the `dev` dependency group.
