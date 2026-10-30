# Case study 1: fraud classifiers (PR-AUC)

> **Data:** SYNTHETIC stand-in (30,000 rows, 1.083% positive) -- NOT real fraud data.
> The real ULB file could not be downloaded here; rerun with `--data creditcard.csv` for the real analysis. Nothing below is a claim about real fraud data.

## 1. Question

Are a random forest or LightGBM *really* better than a logistic-regression baseline at ranking
fraud, measured by PR-AUC (average precision)? Accuracy is not used: with about 1% positives a
model that never flags fraud already scores about 99%. A gain smaller than **0.01 PR-AUC** is
treated as practically irrelevant.

## 2. Data and dependence structure

Repeated stratified 5-fold CV, 6 repeats = 30 fold scores per model, all models on identical
folds, so scores are paired by (repeat, fold). Training sets overlap across folds and repeats, so
the scores are **not independent**: `dependence = cv_overlap` with n_train = 24,000 and
n_test = 6,000.

## 3. Naive analysis, and why it misleads

A plain paired t-test on the 30 fold scores treats them as independent. In the failure gallery
(`validation/failure_gallery/`) that test falsely flags two equally good models in 28-57% of
datasets when 3-10 repeats are used.

| candidate | naive mean diff | naive 95% CI | naive p | corrected 95% CI (Holm level) |
|---|---:|---|---:|---|
| random_forest | -0.0136 | [-0.0188, -0.0083] | 1.1e-05 | [-0.0312, +0.0041] (p=0.0794) |
| lightgbm | +0.0118 | [+0.0076, +0.0160] | 3.34e-06 | [-0.0024, +0.0260] (p=0.059) |

## 4. Correct analysis (Nadeau-Bengio + Holm across the 2 candidates)



Estimates are **candidate − baseline** in the metric's own units. "Oriented" values flip the sign
for lower-is-better metrics so that positive always means the candidate wins.

### random_forest vs logreg — pr_auc

**Verdict: inconclusive** (`inconclusive`)

CI straddles 0 and extends beyond the margin.

**How much more data?** more CV repeats won't help — you need more data (88% of the variance factor is the n_test/n_train term).


| Method | Estimate | 97.5% CI | p-value |
|---|---:|---|---:|
| nadeau_bengio | -0.01356 | [-0.03119, 0.004069] | 0.07937 |

Multiplicity (holm across 2 candidates): adjusted p = 0.1179; CIs use per-comparison alpha = 0.025.

Oriented effect: -0.01356 (CI -0.03119 to 0.004069), minimum effect of interest δ = 0.01.

**Why this method:** Nadeau-Bengio corrected resampled t-test: dependence=cv_overlap. Training sets overlap, so per-fold scores are correlated; the variance is inflated by (1/J + n_test/n_train) instead of 1/J. Valid but conservative.


### lightgbm vs logreg — pr_auc

**Verdict: inconclusive** (`inconclusive`)

CI straddles 0 and extends beyond the margin.

**How much more data?** more CV repeats won't help — you need more data (88% of the variance factor is the n_test/n_train term).


| Method | Estimate | 97.5% CI | p-value |
|---|---:|---|---:|
| nadeau_bengio | 0.01179 | [-0.002388, 0.02598] | 0.05896 |

Multiplicity (holm across 2 candidates): adjusted p = 0.1179; CIs use per-comparison alpha = 0.025.

Oriented effect: 0.01179 (CI -0.002388 to 0.02598), minimum effect of interest δ = 0.01.

**Why this method:** Nadeau-Bengio corrected resampled t-test: dependence=cv_overlap. Training sets overlap, so per-fold scores are correlated; the variance is inflated by (1/J + n_test/n_train) instead of 1/J. Valid but conservative.



## 5. Second opinion: McNemar on frauds caught

Per-row out-of-fold scores from repeat 0; each model flags its top 1% highest-scoring rows
(300 alerts). Items are the 325 true frauds; "correct" means the fraud was in
the alert budget.

| comparison | frauds caught (candidate / logreg) | discordant pairs | test | p |
|---|---|---|---|---:|
| random_forest vs logreg | 264 / 273 of 325 | b=13, c=22 | mcnemar_chi2 | 0.176 |
| lightgbm vs logreg | 282 / 273 of 325 | b=13, c=4 | mcnemar_exact | 0.049 |

## 6. What would change the conclusion

* A smaller margin than 0.01 would turn "equivalent"/"may be trivial" verdicts into decisive
  ones only if the CI narrows; check the `n_needed` note above for whether repeats or more data
  are required.
* Different hyperparameters, features or a different alert budget could reorder the models.
* Real fraud data are non-stationary (the ULB set covers two days); random-split CV can
  overstate how well a model transfers to next week.
