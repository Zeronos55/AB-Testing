Exact coverage of nominal 95% intervals for one model's accuracy (no simulation).

| N items | true accuracy | CLT (normal) | Wilson | Jeffreys | Clopper-Pearson |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.50 | 0.957 | 0.957 | 0.957 | 0.957 |
| 30 | 0.80 | 0.946 | 0.964 | 0.930 | 0.980 |
| 30 | 0.90 | 0.809 | 0.974 | 0.932 | 0.992 |
| 30 | 0.95 | 0.782 | 0.939 | 0.984 | 0.984 |
| 50 | 0.50 | 0.935 | 0.935 | 0.935 | 0.967 |
| 50 | 0.80 | 0.938 | 0.951 | 0.951 | 0.967 |
| 50 | 0.90 | 0.879 | 0.970 | 0.942 | 0.970 |
| 50 | 0.95 | 0.920 | 0.962 | 0.885 | 0.988 |
| 100 | 0.50 | 0.943 | 0.943 | 0.943 | 0.965 |
| 100 | 0.80 | 0.933 | 0.941 | 0.955 | 0.967 |
| 100 | 0.90 | 0.932 | 0.936 | 0.956 | 0.956 |
| 100 | 0.95 | 0.877 | 0.966 | 0.935 | 0.983 |
| 200 | 0.50 | 0.944 | 0.944 | 0.944 | 0.960 |
| 200 | 0.80 | 0.941 | 0.958 | 0.948 | 0.958 |
| 200 | 0.90 | 0.927 | 0.956 | 0.941 | 0.967 |
| 200 | 0.95 | 0.926 | 0.967 | 0.950 | 0.967 |

Average coverage over true accuracies 0.02-0.98, and the worst case:

| N items | CLT (normal) (mean / min) | Wilson (mean / min) | Jeffreys (mean / min) | Clopper-Pearson (mean / min) |
|---:|---:|---:|---:|---:|
| 30 | 0.904 / 0.454 | 0.953 / 0.919 | 0.949 / 0.889 | 0.973 / 0.954 |
| 50 | 0.924 / 0.635 | 0.952 / 0.922 | 0.950 / 0.885 | 0.968 / 0.953 |
| 100 | 0.938 / 0.802 | 0.951 / 0.933 | 0.949 / 0.912 | 0.964 / 0.952 |
| 200 | 0.943 / 0.857 | 0.951 / 0.933 | 0.949 / 0.928 | 0.960 / 0.950 |

Paired difference of two models (2000 simulated evals per row, models' correctness correlated at 0.4): coverage of the nominal 95% interval for acc_A - acc_B.

| N items | acc A / acc B | normal-theory paired | Tango score (isbetter) |
|---:|---|---:|---:|
| 30 | 0.90 / 0.85 | 0.916 | 0.968 |
| 60 | 0.95 / 0.90 | 0.921 | 0.971 |
| 100 | 0.80 / 0.75 | 0.943 | 0.949 |

Coverage of a discrete count is a sawtooth, so no interval is exact at every accuracy (Jeffreys dips to about 0.89 at N = 30-50). The normal-theory interval is the one that fails badly: at N = 30 it covers 78-81% when accuracy is 0.90-0.95 and 45% at its worst accuracy, where Wilson stays above 0.91 and Clopper-Pearson above 0.95. By N = 200 the gap is small except near the extremes. The paired normal-theory interval undercovers the same way when both models are strong.
