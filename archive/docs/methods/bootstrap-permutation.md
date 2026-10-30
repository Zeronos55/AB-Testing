# Paired bootstrap and sign-flip permutation

**Dependence:** `paired` · **Metric:** continuous · **Code:** `isbetter.methods.resampling` · **Routed as cross-checks** to the paired t-test.

## When to use it

A check that the paired t-test's answer does not hang on normality. The percentile (or BCa) bootstrap gives an interval for the mean difference; the sign-flip permutation test gives a p-value under the null that the sign of each difference is exchangeable. If they disagree with the t-test, believe the disagreement.

## What it assumes

- Pairs are independent and identically distributed.
- Sign-flip: under the null the differences are symmetric about zero.

## How it fails

- **Few pairs**: with fewer than about ten pairs the bootstrap has few distinct resamples and its intervals are too narrow.
- **Heavy skew**: the percentile interval can undercover; BCa corrects part of this but is not the routed default and is limited to moderate n.
- **Serial dependence** between pairs: resampling single pairs breaks it. Use the [block bootstrap](block-bootstrap.md).

## Use it

Run automatically for `dependence: {type: paired}` (percentile bootstrap and permutation, 9,999 resamples each). Memory stays O(batch x n): the bootstrap works on the vector of differences in batches and never builds an n x B index matrix.

```python
from isbetter.methods.resampling import paired_bootstrap, sign_flip_permutation
ci = paired_bootstrap(a, b, rng, method="percentile")   # or "bca" (not routed; n <= 50,000)
p = sign_flip_permutation(a, b, rng)                    # exact for n <= 16, else Monte Carlo
```

## Checked against

- `scipy.stats.permutation_test`: the exact sign-flip p-value equals SciPy's to 1e-12 and the Monte Carlo one is close.
- The percentile interval is checked against the t interval on well-behaved data. BCa is SciPy's `stats.bootstrap` itself, so it has no separate oracle.
- A `tracemalloc` test: n = 300,000 resamples in batches stays within the memory bound, never building an n x B matrix.

## References

- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics* 7.
- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap*.
