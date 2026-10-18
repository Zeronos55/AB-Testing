# Nadeau-Bengio corrected resampled t-test

**Dependence:** `cv_overlap` · **Metric:** per-fold score · **Code:** `isbetter.methods.cv` · **Routed by default** for cross-validation scores.

```text
d_j = per-fold score differences, j = 1..J (J = repeats x folds)
t = mean(d) / sqrt((1/J + n_test/n_train) * var(d)),  df = J - 1
```

## When to use it

Comparing two models from repeated k-fold cross-validation scores on one dataset. Training sets overlap between folds, so the per-fold scores are positively correlated and a plain t-test is far too confident. The correction inflates the variance by 1/J + n_test/n_train instead of 1/J, where J = repeats x folds.

## What it assumes

- The scores come from resampling one dataset; you must give `n_train`, `n_test`, `repeats` and `folds`.
- The n_test / n_train correction is a heuristic for the correlation between folds, not an exact variance.

## How it fails

- **Conservative.** No unbiased estimator of the variance exists (Bengio and Grandvalet, 2004), so the test is valid but loses power.
- **More repeats do not help once n_test / n_train dominates.** The verdict then says you need more data.
- **It tests the learning algorithm on this data**, not a fixed fitted model on future data.
- The naive k-fold t-test (`naive_kfold_t`) is kept only for the failure gallery: at 10 x 10-fold with n = 100 it rejects a true null more than half the time.

## Use it

```yaml
dependence: {type: cv_overlap, n_train: 24000, n_test: 6000, repeats: 6, folds: 5}
```

```python
from isbetter.methods.cv import nadeau_bengio, five_by_two_cv
```

## Checked against

- A hand-computed fixture, and `mlxtend.evaluate.paired_ttest_5x2cv` on the iris example for the 5x2cv test.
- A calibration test: Nadeau-Bengio stays at the nominal rate where the naive test is inflated. See the [calibration table](../calibration.md) and [failure gallery #1](../gallery.md).

## References

- Nadeau, C. and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning* 52.
- Bengio, Y. and Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *Journal of Machine Learning Research* 5.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation* 10.
