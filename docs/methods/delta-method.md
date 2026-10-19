# Delta method for ratio metrics

**Dependence:** `iid` · **Metric:** ratio (revenue per visit, clicks per session) · **Code:** `isbetter.methods.variance_reduction.delta_method_ratio` · **Routed** for `kind: ratio`, with a bootstrap over units as cross-check.

```text
R = mean(Y) / mean(X)
Var(R) ~= (1 / mean(X)^2) * (Var(mean Y) - 2 R Cov(mean Y, mean X) + R^2 Var(mean X))
```

## When to use it

The metric is a ratio of two sums, and users are randomised but contribute different numbers of visits. Treating each visit as an independent row understates the variance. The delta method gives the variance of a ratio of means using the covariance between numerator and denominator across *units*.

## What it assumes

- Units (users) are independent; the analysis is over per-unit numerators and denominators.
- Enough units that the first-order approximation of the ratio is accurate.

## How it fails

- **Small samples or denominators near zero**: the linearisation is poor. The unit-level bootstrap cross-check is there to expose this.
- **Analysing row-level ratios** when randomisation was per user: use this instead.
- Variance reduction is not supported for ratio metrics, and the router says so rather than ignoring a request for it.

## Use it

```yaml
metric: {name: revenue, kind: ratio, denominator: visits}
```

## Checked against

- NumPy and statsmodels reference computations.
- A resampling-over-units bootstrap as an in-report cross-check.

## References

- Deng, A., Knoblich, U. and Lu, J. (2018). Applying the delta method in metric analytics: a practical guide with novel ideas. *KDD*.
