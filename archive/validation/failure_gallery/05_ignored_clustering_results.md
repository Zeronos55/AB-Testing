False-positive rate at nominal 5% (2000 simulated experiments per row, Monte Carlo band 0.040-0.060); 100 users per arm, identical arms.

Varying sessions per user (ICC = 0.3):

| sessions/user | rows analysed | design effect | naive row-level Welch | cluster-robust CR1 |
|---:|---:|---:|---:|---:|
| 1 | 200 | 1.0 | 0.050 | 0.050 |
| 2 | 400 | 1.3 | 0.089 | 0.061 |
| 5 | 1,000 | 2.2 | 0.187 | 0.058 |
| 10 | 2,000 | 3.7 | 0.314 | 0.052 |
| 20 | 4,000 | 6.7 | 0.456 | 0.051 |

Varying the intra-class correlation (10 sessions per user):

| ICC | design effect | naive row-level Welch | cluster-robust CR1 |
|---:|---:|---:|---:|
| 0 | 1.0 | 0.049 | 0.048 |
| 0.05 | 1.4 | 0.108 | 0.056 |
| 0.2 | 2.8 | 0.219 | 0.046 |
| 0.5 | 5.5 | 0.391 | 0.052 |
