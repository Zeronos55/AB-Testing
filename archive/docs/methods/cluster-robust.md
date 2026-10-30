# Cluster-robust inference (CR1)

**Dependence:** `clustered` · **Metric:** any mean · **Code:** `isbetter.methods.cluster` · **Routed by default** for clustered data, with a cluster bootstrap as cross-check; `paired: true` gives the paired version.

## When to use it

Rows share a cluster: sessions of one user, questions on one passage, students in one class. Rows in a cluster are correlated, so the effective sample size is closer to the number of clusters than to the number of rows. The estimator is OLS of the value on an arm dummy with a CR1 cluster-robust standard error and t(G - 1) inference.

## What it assumes

- Clusters are independent of each other, and assignment (or the comparison) is at the cluster level.
- A reasonable number of clusters: the sandwich standard error is an asymptotic result in G.

## How it fails

- **Few clusters**: the interval is mildly liberal. In the calibration table CR1 with 100 clusters per arm lands slightly above 5%, and with 30 passages the paired version is close to nominal.
- **Ignoring the clusters** is far worse: with 10 sessions per user at ICC 0.3 the row-level test rejects about 31% of true nulls, and the paired t-test on 30 passages x 5 items rejects about 26%.
- **Declaring the wrong cluster**: cluster at the level where randomisation (or correlation) happens.

## Use it

```yaml
dependence: {type: clustered, cluster_col: user_id}
dependence: {type: clustered, paired: true}     # same items scored by both models
```

```python
from isbetter.methods.cluster import cluster_robust_diff, paired_cluster_robust
```

## Checked against

- `statsmodels` OLS with `cov_type='cluster'` (t(G - 1) as statsmodels uses); the paired version to 1e-12.
- The paired version reduces exactly to the paired t-test when every item is its own cluster, and equals the form in Miller (2024) transcribed as triple loops.
- The third-party `errorbars` package agrees (a cross-check by another implementer, not the paper's numbers).
- A calibration test and [failure gallery #5](../../validation/failure_gallery/) for the false-positive rate with and without clusters.

## References

- Cameron, A. C. and Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources* 50.
- Miller, E. (2024). Adding error bars to evals: a statistical approach to language model evaluations. arXiv:2411.00640.
