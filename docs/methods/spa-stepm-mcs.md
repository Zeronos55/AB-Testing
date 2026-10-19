# SPA, StepM and the Model Confidence Set

**Dependence:** `timeseries` · **Metric:** per-origin losses of many models · **Code:** `isbetter.methods.superior_predictive` (wraps `arch`, extra `forecast`) · **Used by** the forecasting and trading case studies.

## When to use it

Many models against one benchmark, where picking the best of many and testing it alone would overstate the result. SPA asks whether *any* model beats the benchmark (one p-value). StepM says *which* models beat it with family-wise error control. The Model Confidence Set returns the models that cannot be told apart from the best. The answer is a set, so it lives in `diagnostics`.

## What it assumes

- Losses are stationary and the block bootstrap (with a block length that covers the dependence) is valid.
- Every model you tried is included, since leaving out the losers invalidates the correction.

## How it fails

- **Many models relative to the number of origins** weakens all three.
- **Excluding trials** from the set: the multiplicity correction is only as honest as the list of models you give it.
- When no block length is given, the largest Politis-White length among the loss differentials is used: the conservative choice for several dependent series.

## Use it

```python
from isbetter.methods.superior_predictive import spa_test, stepm, model_confidence_set
spa_test(benchmark_losses, {"arima": l1, "lgbm": l2}, rng)
model_confidence_set({"naive": l0, "arima": l1, "lgbm": l2}, rng, size=0.1)
```

## Checked against

- `arch.bootstrap` itself: the wrappers must reproduce its results for the same data and random stream, and recover planted structure. The wrappers only need to get the plumbing (loss sign, model names, seed, block length) right.

## References

- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics* 23.
- Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica* 73.
- Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set. *Econometrica* 79.
