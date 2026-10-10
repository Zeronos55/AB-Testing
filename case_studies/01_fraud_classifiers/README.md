# Case study 1: fraud classifiers

**Question:** is a random forest or LightGBM really better than logistic regression at ranking
fraud (PR-AUC), once you account for the fact that cross-validation folds overlap?

**Run it**

```bash
uv run python case_studies/01_fraud_classifiers/run.py                       # synthetic stand-in
uv run python case_studies/01_fraud_classifiers/run.py --data creditcard.csv # real ULB data
```

Output: [`REPORT.md`](REPORT.md) (full write-up) and `out/cv_scores.csv` (per-fold scores, which
`uv run isbetter run case_studies/01_fraud_classifiers/config.yaml` can also analyse).

**Data caveat.** The real ULB credit-card file (Kaggle login required) was not reachable from the
build environment, so the committed report uses a **synthetic** imbalanced dataset (30,000 rows,
about 1% positive). It demonstrates the method, not real fraud performance. Rerun with `--data`
to get the real analysis; `n_train`/`n_test` are then taken from the actual fold sizes.

**Headline on the synthetic data**

| candidate vs logreg | naive paired t on 30 fold scores | Nadeau-Bengio + Holm |
|---|---|---|
| random forest | significant (p about 1e-5), worse | inconclusive (p = 0.079) |
| LightGBM | significant (p about 3e-6), better | inconclusive (p = 0.059) |

The naive test "finds" two significant differences that the overlap-aware test cannot support,
and the report says why more CV repeats will not fix it: 88% of the variance factor is the
`n_test/n_train` term, so more data is needed.
