# Small-N intervals: Wilson, Jeffreys, Clopper-Pearson, Tango

**Dependence:** `paired`, binary scores · **Code:** `isbetter.methods.small_n`, `isbetter.methods.proportions` · **Routed** for paired right/wrong metrics (`kind: binary` under `paired`) as `paired_binary_score`, with a Beta/Dirichlet posterior cross-check.

## When to use it

An LLM eval or similar with 30 to 200 items, where the usual error bar `mean +/- 1.96 sd / sqrt(n)` is too narrow. For one accuracy, Wilson, Jeffreys and Clopper-Pearson intervals stay close to nominal coverage. For the difference between two models on the same items, the paired route returns a Tango (1998) score interval for the accuracy difference together with McNemar's p-value, and the Wilson and Jeffreys intervals of each arm.

## What it assumes

- Items are independent (for clustered items use [cluster-robust](cluster-robust.md) with `paired: true`).
- Scores are right or wrong (0/1).

## How it fails

- **Coverage of a discrete count is a sawtooth** in the true accuracy, so no interval is exact everywhere: Jeffreys can dip to about 89% at N = 30 and Wilson to about 93%, while Clopper-Pearson never falls below 95% but is conservative. See the [calibration table](../calibration.md).
- **The normal-theory interval** is kept as `clt_interval` only to be compared: at N = 30 and accuracy 0.95 it covers about 78%, and near 0% or 100% it collapses to a point.
- **This is not Bowyer et al.'s own paired recipe**, which was not available to check. Tango's interval was chosen because it can be verified independently.

## Use it

```yaml
metric: {name: correct, kind: binary}
dependence: {type: paired}
```

```python
from isbetter.methods.small_n import paired_binary_score, clopper_pearson_interval
from isbetter.methods.proportions import wilson_interval, jeffreys_interval
```

## Checked against

- The restricted maximum-likelihood estimate against a numeric optimiser; the score statistic at delta = 0 equals McNemar's uncorrected chi-square (statsmodels); the endpoints solve the score equation; zero is inside the interval exactly when McNemar does not reject.
- SciPy's exact interval for Clopper-Pearson; a Dirichlet-posterior cross-check.
- Single-arm coverage is computed exactly (the count is binomial), with no simulation: [failure gallery #3](../../validation/failure_gallery/).
- The third-party `errorbars` package agrees on Wilson, McNemar and the paired interval.

## References

- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA* 22.
- Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science* 16.
- Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* 26.
- Tango, T. (1998). Equivalence test and confidence interval for the difference in proportions for the paired-sample design. *Statistics in Medicine* 17.
- Bowyer, S., Aitchison, L. and Ivanova, D. (2025). Don't use the CLT in LLM evals with fewer than a few hundred datapoints.
