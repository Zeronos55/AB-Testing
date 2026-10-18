# Quickstart

## Install

The package is not on PyPI yet (version 0.1.0 is prepared for release), so install from a clone. It is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Zeronos55/AB-Testing
cd AB-Testing
uv sync
```

Python 3.11 or newer. Optional extras: `ab` (DuckDB and polars for large A/B tables), `ml`
(scikit-learn and LightGBM for the CV adapter), `forecast` and `finance` (`arch`),
`app` (the [viewer](concepts/reports.md)) and `docs` (this site).

## Run an example

`examples/toy_paired.yaml` compares two variants measured on the same 120 units:

```yaml
experiment: toy_paired
adapter: csv_long
data: toy_paired.csv
variants: {baseline: baseline, candidates: [candidate]}
metric: {name: score, higher_is_better: true}
dependence: {type: paired}
decision: {min_effect: 0.1, alpha: 0.05}
seed: 20261010
```

```console
$ uv run isbetter run examples/toy_paired.yaml
Analysis plan: toy_paired
  dependence : {'type': 'paired'}
  compare    : candidate vs baseline 'baseline'
  metric     : score (higher is better)
  primary    : paired_t
  cross-check: paired_bootstrap, sign_flip_permutation
  alpha=0.05  min_effect=0.1  correction=none  seed=20261010
candidate vs baseline: A_better_practical
```

The plan prints before anything runs. The report (`REPORT.md`, and `REPORT.html` when the config
asks for it) shows the estimate and interval from the primary method, the cross-checks, the
verdict and a plain-English reason for the choice of method:

| Method | Estimate | 95.0% CI | p-value |
|---|---:|---|---:|
| paired_t | 0.3246 | [0.24, 0.4092] | 7.567e-12 |
| paired_bootstrap_percentile | 0.3246 | [0.2427, 0.4091] | n/a |
| sign_flip_permutation | 0.3246 | n/a | 0.0001 |

The interval lies entirely above the minimum effect of interest (0.1), so the verdict is
`A_better_practical`.

## The config

| Key | Meaning |
|---|---|
| `adapter` | How the data are read: `csv_long` (default), `sklearn_cv`, `ab_table`, `forecast_backtest`, `strategy_returns` or `llm_eval`. |
| `variants` | The `baseline` and one or more `candidates`. |
| `metric` / `metrics` | One metric, or a list for an A/B experiment. `kind` is `continuous`, `binary`, `ratio` or `returns`. |
| `dependence` | The structure of the data: the key choice. See [declaring dependence](concepts/dependence.md). |
| `decision` | `min_effect` (the smallest difference that matters), `alpha`, and `correction` (`none`, `holm`, `bonferroni`, `bh`). |
| `seed` | Recorded in the result; all randomness goes through it. |
| `report` | `format: [markdown, html]`. |

An unknown key is an error rather than being ignored, and a bad config gives a readable message.

## Other commands

```bash
uv run isbetter run config.yaml --json-out result.json   # also save the Result
uv run isbetter view result.json                         # browse it in the viewer
uv run isbetter plan-size --kind proportions --baseline 0.10 --mde 0.01   # sample size / power
uv run pytest                                            # fast tests
uv run pytest -m slow                                    # calibration, 10,000 replications
```
