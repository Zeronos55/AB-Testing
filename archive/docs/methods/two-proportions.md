# Two-proportion z, Newcombe, Fisher

**Dependence:** `iid` · **Metric:** binary (conversion, click) · **Code:** `isbetter.methods.proportions` · **Routed by default**, with Fisher's exact test as cross-check.

## When to use it

A randomised A/B test on a 0/1 outcome per user. The z-test reports the difference in rates with a Wald or Newcombe interval; Fisher's exact test (conditional on the margins) checks it when counts are small. Wilson and Jeffreys intervals for a single arm are in the same module.

## What it assumes

- One 0/1 outcome per randomised unit; units independent.
- For the z-test, enough successes and failures in each arm for the normal approximation (a common rule: expected counts of at least 5).

## How it fails

- **Rows are impressions, not users**: if one user contributes several rows the units are not independent. Aggregate to users or declare `clustered`. The Criteo case study lists this as an open question for the real data.
- **Rare events with small counts**: the z interval is unreliable; read the Fisher cross-check and the Newcombe interval.
- **Peeking** at the p-value while the test runs inflates false positives: see the [failure gallery](../../validation/failure_gallery/) and [confidence sequences](confidence-sequences.md).

## Use it

```yaml
metric: {name: conversion, kind: binary, min_effect: 0.002}
dependence: {type: iid}
```

With `adapter: ab_table`, a sample-ratio check, multiplicity across success metrics and guardrails are added automatically; see [guardrails](guardrails.md).

## Checked against

- `statsmodels.stats.proportion` (z-test, Wilson, Jeffreys, Newcombe) and SciPy's `fisher_exact`.
- An A/A simulation of the false-positive rate: see the [calibration table](../calibration.md).

## References

- Newcombe, R. G. (1998). Interval estimation for the difference between independent proportions: comparison of eleven methods. *Statistics in Medicine* 17.
- Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science* 16.
