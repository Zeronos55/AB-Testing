# Welch t-test

**Dependence:** `iid` · **Metric:** continuous · **Code:** `isbetter.methods.parametric.welch_t` · **Routed by default** for independent arms.

```text
t = (mean_A - mean_B) / sqrt(s_A^2 / n_A + s_B^2 / n_B), df by Welch-Satterthwaite
```

## When to use it

Two independent groups and a continuous metric (revenue per user, latency, a score): a randomised A/B test on a mean. Unequal group sizes and unequal variances are fine, which is why it is used instead of Student's t. Effect sizes (Cohen's d, Hedges' g) are in the same module.

## What it assumes

- One row per randomised unit, units independent within and across arms.
- A finite variance. The interval uses the t distribution, so it leans on the central limit theorem.

## How it fails

- **Repeated measures per user** (many sessions, many questions per passage): rows are not independent and the test is too confident. Declare `clustered`; see [cluster-robust](cluster-robust.md).
- **Paired data**: ignoring the pairing throws away the power that pairing buys. Declare `paired`.
- **Very skewed metrics at small n**: the interval is approximate, so treat a borderline result with care.

## Use it

```yaml
dependence: {type: iid}
metric: {name: revenue, higher_is_better: true}
```

```python
from isbetter.methods.parametric import welch_t
result = welch_t(a, b, alpha=0.05)   # estimate = mean(a) - mean(b)
```

## Checked against

- `scipy.stats.ttest_ind(equal_var=False)`, to 1e-8.
- An A/A simulation of the Type I error: see the [calibration table](../calibration.md).

## References

- Welch, B. L. (1947). The generalization of 'Student's' problem when several different population variances are involved. *Biometrika* 34.
