Coverage of nominal 95% Sharpe-ratio intervals (1000 simulated return series per row, Monte Carlo band 0.936-0.964); T=500 periods, AR(1) coefficient phi, GARCH(1,1) t(5) innovations, true per-period Sharpe 0.05.

| autocorrelation phi | iid bootstrap | block bootstrap (Politis-White length) |
|---:|---:|---:|
| 0.0 | 0.949 | 0.931 |
| 0.1 | 0.912 | 0.920 |
| 0.2 | 0.892 | 0.934 |
| 0.3 | 0.834 | 0.915 |
| 0.4 | 0.782 | 0.909 |

The iid bootstrap is fine only without autocorrelation (phi = 0) and degrades quickly as it grows. The block bootstrap holds up much better but is not exact: with T=500 it stays about 2-4 points short (see tests/calibration/test_sharpe_coverage.py), and at phi = 0 it gives up a little coverage for estimating a dependence that is not there.
