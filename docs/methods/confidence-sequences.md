# Confidence sequences (always-valid inference)

**Dependence:** `iid` or `paired` with a `sequential:` block · **Code:** `isbetter.methods.sequential` · **Routed as the primary** method when `sequential` is set, with the fixed-horizon test as the foil.

```text
xbar_t +/- (s / t) * sqrt((t + rho) * log((t + rho) / (rho * alpha^2)))
```

## When to use it

A result you will read while the data are still coming in. A fixed-horizon test is valid only if you look once, at the planned sample size: checking daily and stopping at the first p < 0.05 inflates the false-positive rate far above 5%. A confidence sequence is an interval for the mean that holds at **all** sample sizes at once, so it can be read after every observation, with the probability that the truth is ever outside it at most alpha.

## What it assumes

- **`n_target` (or `rho`) is fixed in advance.** It says where the sequence is tightest. The length of the data you already have is refused as a source, because tuning to it would break the guarantee.
- With a known `sigma` (or a sub-Gaussian bound such as 0.5 for scores in [0, 1]) the guarantee is exact. With the running standard deviation plugged in it is asymptotic.

## How it fails

- **The plug-in version needs enough observations**: starting at 2 it has a false-positive rate of about 9%, so `min_n` defaults to 20.
- **A wider interval is the price**: the sequence is tightest around the planned size and wider elsewhere.
- **Not yet routed** for A/B experiments (SRM, guardrails), clustered data or time series.
- Rows need a `time` column giving arrival order.

## Use it

```yaml
metric: {name: revenue}
dependence: {type: iid}
sequential: {n_target: 1000}          # or {rho: ...}; optional sigma, min_n
```

```python
from isbetter.methods.sequential import confidence_sequence_mean
cs = confidence_sequence_mean(x, n_target=1000)
```

## Checked against

- Numerical integration of the mixture martingale over lambda; its expectation of 1 under the truth; the interval edge being exactly where the martingale equals 1/alpha; the optimal rho against brute force and the closed form of Howard et al. (the papers' published numbers could not be fetched where this was built).
- A calibration test: read after every observation from 20 to 2,000 per arm, the sequence's false-positive rate stays below 5% while the fixed-horizon test stopped at its first p < 0.05 rejects in nearly half of the runs. See the [calibration table](../calibration.md) and [failure gallery #4](../../validation/failure_gallery/).

## References

- Robbins, H. (1970). Statistical methods related to the law of the iterated logarithm. *Annals of Mathematical Statistics* 41.
- Howard, S. R., Ramdas, A., McAuliffe, J. and Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. *Annals of Statistics* 49.
- Johari, R., Koomen, P., Pekelis, L. and Walsh, D. (2022). Always valid inference: continuous monitoring of A/B tests. *Operations Research* 70.
