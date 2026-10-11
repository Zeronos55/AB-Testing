False-positive rate at nominal 5% (2000 simulated datasets per row, Monte Carlo band 0.040-0.060); T=100 forecast origins, two equally accurate forecasters, squared-error loss.

| horizon h | naive paired t | Diebold-Mariano + HLN |
|---:|---:|---:|
| 1 | 0.043 | 0.043 |
| 2 | 0.114 | 0.056 |
| 3 | 0.174 | 0.060 |
| 5 | 0.295 | 0.059 |
| 7 | 0.364 | 0.073 |
| 10 | 0.456 | 0.073 |

Diebold-Mariano + HLN is far closer to 5% but not exact: with only T=100 origins it stays slightly liberal at long horizons (see tests/calibration/test_dm_type1.py).
