# Paired t-test

**Dependence:** `paired` · **Metric:** continuous · **Code:** `isbetter.methods.parametric.paired_t` · **Routed by default** for paired data, with the [bootstrap and permutation](bootstrap-permutation.md) cross-checks.

```text
d_i = a_i - b_i ;  t = mean(d) / (sd(d) / sqrt(n)),  df = n - 1
```

## When to use it

Both variants measured on the same units: the same users under two treatments, the same test items scored by two models. Pairing removes unit-level noise shared by both measurements, so it needs far fewer units than an unpaired test for the same precision.

## What it assumes

- Pairs are independent of each other.
- The within-pair differences are roughly normal, or there are enough pairs for the central limit theorem.

## How it fails

- **A missing partner or a duplicate row**: a declared pairing with a gap is rejected with an error rather than silently dropped.
- **Pairs that share a cluster** (items from the same passage): the differences are correlated; use `clustered` with `paired: true`.
- **Pairs ordered in time** with autocorrelated differences: use `timeseries`.

## Use it

```yaml
dependence: {type: paired}
```

The data need a `pair_id` column linking each unit's two rows (see `examples/toy_paired.csv`).

## Checked against

- `scipy.stats.ttest_rel`, to 1e-8.
- Property tests: swapping A and B flips the sign of the estimate, and adding a constant to both arms leaves the p-value unchanged.

## References

- Student (Gosset, W. S.) (1908). The probable error of a mean. *Biometrika* 6.
