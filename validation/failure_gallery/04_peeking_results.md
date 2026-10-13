False-positive rate at nominal 5% when two identical arms are read at several looks (up to 2,000 observations per arm; 2000 simulated experiments per row, Monte Carlo band 0.040-0.060).

| number of looks | fixed-horizon test, stop at first p < 0.05 | confidence sequence (isbetter) | fixed-horizon test read once at the end |
|---:|---:|---:|---:|
| 1 | 0.052 | 0.001 | 0.052 |
| 2 | 0.099 | 0.003 | 0.047 |
| 5 | 0.172 | 0.006 | 0.053 |
| 10 | 0.232 | 0.009 | 0.050 |
| 20 | 0.272 | 0.012 | 0.050 |
| 50 | 0.359 | 0.017 | 0.057 |
| 200 | 0.399 | 0.015 | 0.041 |
| 1,981 | 0.457 | 0.019 | 0.049 |

The fixed-horizon test is fine when it is read once, and its false-positive rate climbs with every extra look (a daily check over a few weeks is the 20-look row). The confidence sequence stays below 5% at every number of looks; it is conservative (about 2% when read continuously) because it pays for being valid at all sample sizes with a wider interval: about 3.0 standard errors instead of 1.96 at the sample size it is tuned for.
