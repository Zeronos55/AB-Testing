# Diebold-Mariano test with the HLN correction

**Dependence:** `timeseries` (set `horizon`) · **Metric:** per-origin loss · **Code:** `isbetter.methods.forecasting` · **Routed by default** for forecast losses, with a stationary block bootstrap as cross-check.

```text
DM = mean(d) / sqrt(LRV / T), LRV = gamma_0 + 2 * sum_{k<h} gamma_k  (each gamma divided by T)
DM_HLN = sqrt((T + 1 - 2h + h(h-1)/T) / T) * DM,  compared with t(T - 1)
```

## When to use it

Two forecasters scored on the same targets, one loss per forecast origin. The null is that the mean loss differential d_t = L(A, t) - L(B, t) is zero. An h-step-ahead error overlaps those of the next h - 1 origins, so d_t is autocorrelated and the variance of its mean must include autocovariances up to lag h - 1. The Harvey-Leybourne-Newbold factor corrects the small-sample bias, and the statistic is compared with t(T - 1). Lower loss is better, so set `higher_is_better: false`.

## What it assumes

- The loss differential is covariance stationary, and the h-step forecast errors are at most MA(h - 1) under the null.
- The forecasts being compared are competing forecasts, not nested models fitted on the same data.

## How it fails

- **Ignoring the autocorrelation** (a plain paired t-test on losses) over-rejects as h grows: 17% at h = 3 and 35% at h = 7 for a nominal 5%. See [failure gallery #2b](../../validation/failure_gallery/).
- **HLN is not exact**: with few origins it stays mildly liberal at long horizons (about 6 to 7% at h = 7 to 10 with T = 100).
- **Nested models**: Diebold-Mariano is not valid when one model nests the other (Clark-West is for that case).
- **The loss function is a choice** that can reverse the ranking; it should reflect how the forecast is used.

## Use it

```yaml
metric: {name: mae_h3, higher_is_better: false}
dependence: {type: timeseries, horizon: 3}
```

```python
from isbetter.methods.forecasting import diebold_mariano
diebold_mariano(loss_a, loss_b, horizon=3)      # HLN on by default; hac_lags defaults to h - 1
```

## Checked against

- statsmodels' HAC covariance (uniform and Bartlett kernels), since statsmodels has no Diebold-Mariano test.
- A transcription of R's `forecast::dm.test`, and **stored `dm.test` output** (R 4.5.3, `forecast` 9.0.2; 12 cases at h = 1, 3, 7, T = 40 and 120) matched to a relative 1e-8.
- Calibration at h = 1, 3 and 7: see the [calibration table](../calibration.md).

## References

- Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics* 13.
- Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting* 13.
- Diebold, F. X. (2015). Comparing predictive accuracy, twenty years later. *Journal of Business & Economic Statistics* 33.
