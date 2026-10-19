# Six ways to fool yourself when comparing A and B

*A write-up of the failure gallery. Every number below is from a script in
[`validation/failure_gallery/`](https://github.com/Zeronos55/AB-Testing/tree/main/validation/failure_gallery),
run on simulated data where the truth is known. The full tables and figures are on the
[gallery page](../gallery.md).*

Most wrong answers to "is A better than B?" have nothing wrong with the arithmetic. The t statistic is computed correctly, the p-value is read off the right distribution, and the conclusion is still wrong, because the test assumed the observations were independent and they were not.

There is a cheap way to see this. Take two things that are exactly equally good, compare them with the method under suspicion many times, and count how often it declares a winner. A test at the 5% level should do that 5% of the time. Here are six cases where the textbook method does not, next to the method `isbetter` uses instead.

## 1. Cross-validation folds share their training data

Two learners with identical expected error, compared with repeated 10-fold cross-validation on 100 rows. The folds' training sets overlap, so the per-fold scores are positively correlated and a plain t-test treats them as more independent than they are.

| CV repeats | naive k-fold t | Nadeau-Bengio |
|---:|---:|---:|
| 1 | 4.6% | 1.0% |
| 3 | 27.9% | 3.3% |
| 5 | 41.1% | 4.3% |
| 10 | 56.8% | 4.8% |

More repeats look like more evidence, so the naive test gets *worse* the harder you work. The Nadeau-Bengio correction inflates the variance by `1/J + n_test/n_train` instead of `1/J`, and its error rate stays below 5%. It is conservative, which is the cost: no unbiased variance estimator exists. ([method card](../methods/nadeau-bengio.md))

## 2. Sessions are not users

Users are randomised to A or B and each contributes several sessions, which share a user effect. Analysing 2,000 session rows as if they were 2,000 independent units gives:

| sessions per user | design effect | naive row-level test | cluster-robust (CR1) |
|---:|---:|---:|---:|
| 1 | 1.0 | 5.0% | 5.0% |
| 5 | 2.2 | 18.7% | 5.8% |
| 10 | 3.7 | 31.4% | 5.2% |
| 20 | 6.7 | 45.6% | 5.1% |

With one session per user there is no problem. The error grows with the number of rows, which is the opposite of how data should behave. Even a small correlation matters: at an intra-class correlation of 0.05 and 10 sessions per user, the naive test already rejects 10.8% of true nulls. ([method card](../methods/cluster-robust.md))

![False-positive rate of a row-level test and a cluster-robust test as sessions per user and intra-class correlation grow](../gallery/figures/05_ignored_clustering.png)

## 3. A forecast several steps ahead overlaps its neighbours

An h-step-ahead forecast error overlaps the errors made at the next h - 1 origins, so the loss differential between two forecasters is autocorrelated. A paired t-test on it ignores that. Two equally accurate forecasters, 100 origins:

| horizon h | naive paired t | Diebold-Mariano + HLN |
|---:|---:|---:|
| 1 | 4.3% | 4.3% |
| 3 | 17.4% | 6.0% |
| 7 | 36.4% | 7.3% |
| 10 | 45.6% | 7.3% |

At h = 1 there is nothing to correct, and the two agree. The correction is far better but not exact: with 100 origins it stays slightly liberal at long horizons. ([method card](../methods/diebold-mariano.md))

## 4. Looking at the dashboard

A fixed-horizon test is valid if you look once, at the sample size you planned. Experiments are usually watched. Two identical arms, read at several points up to 2,000 observations per arm, stopping at the first p < 0.05:

| number of looks | fixed-horizon, stop at first p < 0.05 | confidence sequence | fixed-horizon, read once at the end |
|---:|---:|---:|---:|
| 1 | 5.2% | 0.1% | 5.2% |
| 5 | 17.2% | 0.6% | 5.3% |
| 20 | 27.2% | 1.2% | 5.0% |
| 1,981 (every observation) | 45.7% | 1.9% | 4.9% |

A daily check over a few weeks is the 20-look row. The same test is fine when read once, so nothing is wrong with the test; the problem is the stopping rule. The right panel of the figure below is one run of two identical arms: the difference crosses the usual 1.96 line at around 1,250 observations per arm, where a dashboard would say "significant", and the confidence sequence never does. A confidence sequence is an interval valid at every sample size at once, so it may be read after every observation. It pays for that with a wider interval: about 3.0 standard errors instead of 1.96 at the sample size it is tuned for, which must be fixed in advance. ([method card](../methods/confidence-sequences.md))

![False-positive rate against the number of looks, and one A/A run against both boundaries](../gallery/figures/04_peeking.png)

## 5. Thirty items is not enough for a normal approximation

An LLM eval with 30 to 200 items, and the error bar `mean +/- 1.96 sd / sqrt(n)`. The coverage of a nominal 95% interval for one model's accuracy is a finite sum (the count of correct answers is binomial), so there is no simulation noise:

| N = 30, true accuracy | normal | Wilson | Clopper-Pearson |
|---:|---:|---:|---:|
| 0.80 | 94.6% | 96.4% | 98.0% |
| 0.90 | 80.9% | 97.4% | 99.2% |
| 0.95 | 78.2% | 93.9% | 98.4% |

Over every true accuracy from 0.02 to 0.98 at N = 30, the normal interval's worst case is 45.4% coverage; Wilson's is 91.9% and Clopper-Pearson's is 95.4%. Nothing is exact everywhere, since the coverage of a discrete count is a sawtooth: Jeffreys dips to 88.9% at N = 30. But the normal interval fails badly where eval results often sit, with strong models near the top of the scale. The paired comparison of two models has the same problem and a score interval fixes it. ([method card](../methods/small-n.md))

## 6. An iid bootstrap on dependent returns

A bootstrap interval for a Sharpe ratio that resamples single days ignores autocorrelation in returns, so the interval is too short. 500 periods, AR(1) returns with GARCH noise, nominal 95%:

| autocorrelation | iid bootstrap | block bootstrap |
|---:|---:|---:|
| 0.0 | 94.9% | 93.1% |
| 0.2 | 89.2% | 93.4% |
| 0.4 | 78.2% | 90.9% |

The block bootstrap holds up much better, and is not exact: it stays 2 to 4 points short at this sample size, and at zero autocorrelation it gives up a little coverage for estimating a dependence that is not there. ([method card](../methods/sharpe-psr-dsr.md))

## What the six have in common

In each case the data have a structure (overlapping folds, repeated users, overlapping horizons, a stopping rule, a discrete count, serial dependence) and a method that assumes it away. The failure is not subtle and not rare: it takes 10 CV repeats, 10 sessions per user or three weeks of daily checks, which are ordinary things to do.

That is the design of `isbetter`. You declare the [dependence](../concepts/dependence.md) of your data, and the router chooses the method that is valid for it, printing the plan before it runs. If there is no valid method for your combination, it refuses instead of falling back to the naive one.

## What this does not show

- **These are simulations of one data-generating process each.** They show that the failure exists and how large it can be, not how large it is in your data. The case studies run on synthetic stand-ins with the structure of real datasets; none of them uses the real data.
- **The fixes are not free or perfect.** Nadeau-Bengio is conservative, the confidence sequence is wide, and Diebold-Mariano with the HLN correction and the block bootstrap are slightly off at small sample sizes. The [calibration table](../calibration.md) reports every recommended method as it came out, including where it falls outside its Monte Carlo band.
- **Six failures are not all of them.** Multiple comparisons, selection among many strategies and sample-ratio mismatches have their own [method cards](../methods/index.md).

## Reproduce it

```bash
git clone https://github.com/Zeronos55/AB-Testing && cd AB-Testing && uv sync
uv run python validation/failure_gallery/01_naive_cv_ttest.py
uv run python validation/failure_gallery/05_ignored_clustering.py
uv run python validation/failure_gallery/02b_dm_horizon.py
uv run python validation/failure_gallery/04_peeking.py
uv run python validation/failure_gallery/03_clt_small_n.py
uv run python validation/failure_gallery/02_iid_bootstrap_sharpe.py
```

Each script writes its table and figure. Seeds are fixed, so you should get the same numbers as above.
