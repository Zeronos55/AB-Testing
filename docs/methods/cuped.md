# CUPED and regression adjustment

**Dependence:** `iid` · **Metric:** continuous or binary, with a pre-experiment covariate · **Code:** `isbetter.methods.variance_reduction` · **Routed when** `variance_reduction` is set; the unadjusted test is the cross-check.

```text
theta = cov(Y, X) / var(X)   (pooled across arms)
Y' = Y - theta * (X - mean(X)) ;  Var(Y') = (1 - rho^2) Var(Y)
```

## When to use it

An A/B test where you have a pre-experiment measurement correlated with the outcome (last month's spend for this month's spend). Subtracting the part of the outcome the covariate predicts shrinks the variance of the estimate by a factor of 1 - rho^2 without changing what is estimated, so the same data give a tighter interval, or the same precision needs fewer users.

## What it assumes

- **The covariate is measured before randomisation** and cannot be affected by the treatment. This is the one assumption that matters.
- CUPED uses one slope for both arms; `regression` fits arm-specific slopes (Lin, 2013).

## How it fails

- **A covariate influenced by the treatment** biases the estimate. Do not use anything measured after assignment.
- **A weak covariate** (rho near 0) gains little. The report shows the unadjusted interval beside the adjusted one so you can see the gain.
- **A curved relation**: a linear adjustment removes only the linear part. It stays valid, since randomisation guarantees that, but gains less.

## Use it

```yaml
metrics:
  - {name: spend, kind: continuous, variance_reduction: cuped, covariate: history}
```

Works on aggregated sufficient statistics (including the covariate cross-products), so it applies to tables too large for memory.

## Checked against

- The CUPED estimate equals the OLS coefficient on treatment with a centred covariate (statsmodels).
- A calibration test keeps Type I error at the nominal level, including a curved covariate-outcome relation: see the [calibration table](../calibration.md).

## References

- Deng, A., Xu, Y., Kohavi, R. and Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data. *WSDM*.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data. *Annals of Applied Statistics* 7.
