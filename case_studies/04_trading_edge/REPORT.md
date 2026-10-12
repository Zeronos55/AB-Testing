# Case study 4: trading edge or luck? (moving-average crossovers)

> **Data:** SYNTHETIC daily prices -- NOT real market data.
> Real ETF prices could not be downloaded here; rerun with `--data prices.csv` for a real analysis. Nothing below is a claim about real markets.

## 1. Question

Search 272 long/flat moving-average rules, keep the best one: is its out-performance over
buy-and-hold an *edge*, or what you would expect from the best of 272 rules that have no
skill at all? A difference smaller than **0.0002 a day** (about
5% a year) is treated as practically irrelevant.

## 2. Data and dependence structure

Daily returns (2,520 days, the first 250 used to warm up the averages). Daily returns
are serially dependent in their *size* (volatility clusters) and, for some strategies, in their
direction (a trend rule holds positions for weeks, so its daily returns are autocorrelated), hence
`dependence = timeseries`. The 272 rules are evaluated on the same days, so they are
strongly correlated with each other, and every evaluation is appended to a `TrialRegistry`
(`out/trials_*.jsonl`), the source of the trial count and the Sharpe spread used by the DSR.

Every statistic below is about the **excess return over buy-and-hold** (rule return minus
buy-and-hold return, per day).

## 3. Naive analysis, and why it misleads

Pick the rule with the highest Sharpe ratio, and t-test whether its mean excess return is zero.
That test is valid for *one* rule chosen in advance. The best of 272 is a maximum, so its
t-statistic is inflated even when no rule has any edge. Disclosure: the worlds shown below are
conditioned on the naive test being fooled, so the story is visible; how often that happens
without conditioning is in section 5.

* World A: seed 20261014, the first of 5 consecutive seeds (starting at the config seed) in which the best variant's naive t exceeds 1.96.
* World B: seed 20261011, the first of 2 consecutive seeds (starting at the config seed) in which the best variant's naive t exceeds 1.96.

## 4. Correct analysis

### World A: no edge exists (returns are unpredictable)

*SYNTHETIC world 'luck'.* Best variant by Sharpe ratio: **`ma_15_30`**, scored on 2,270 days
(9.0 years); its excess return over buy-and-hold averages +8.4% a year.

| step | what it says | result |
|---|---|---|
| naive t-test | mean excess return of the best rule = 0? | t = **2.06**, p = 0.039 |
| PSR (no deflation) | P(true Sharpe > 0), given skew and fat tails | 0.976 |
| **DSR** | P(true Sharpe > what the best of 272 unskilled variants reaches) | **0.820** (threshold SR* = 0.37 a year, observed 0.69) |
| MinTRL | track record needed to clear 0 / to clear SR* | 6.2 years / 29.0 years (have 9.0 years) |
| block-bootstrap CI | 95% interval of the annual excess Sharpe ratio | [-0.01, +1.40] |
| iid-bootstrap CI | same, wrongly assuming independent days | [+0.04, +1.39] |
| SPA vs buy-and-hold | does *any* of the 272 rules beat buy-and-hold? | consistent p = 0.190 |
| StepM vs buy-and-hold | rules that significantly beat it (family-wise 5%) | 0 of 272 |
| **PBO** (CSCV, 16 blocks) | how often the in-sample best is at or below the OOS median | **0.45** |
| standard DM test, ignoring the search | verdict on `best_variant` vs buy-and-hold | `A_better_maybe_trivial` |

**Cannot be told from luck.** DSR = 0.820 (< 0.95): after accounting for the 272 variants tried, the best variant's Sharpe ratio is not distinguishable from the best of that many unskilled variants. The deflation is the difference between the PSR row and the DSR row.

*Known only because the data are simulated:* no rule has any edge here, so the best variant's apparent out-performance is luck.

### World B: an edge exists (persistent trends)

*SYNTHETIC world 'trend'.* Best variant by Sharpe ratio: **`ma_25_250`**, scored on 2,270 days
(9.0 years); its excess return over buy-and-hold averages +8.7% a year.

| step | what it says | result |
|---|---|---|
| naive t-test | mean excess return of the best rule = 0? | t = **3.17**, p = 0.002 |
| PSR (no deflation) | P(true Sharpe > 0), given skew and fat tails | 0.999 |
| **DSR** | P(true Sharpe > what the best of 272 unskilled variants reaches) | **0.896** (threshold SR* = 0.63 a year, observed 1.05) |
| MinTRL | track record needed to clear 0 / to clear SR* | 2.5 years / 15.4 years (have 9.0 years) |
| block-bootstrap CI | 95% interval of the annual excess Sharpe ratio | [+0.05, +2.05] |
| iid-bootstrap CI | same, wrongly assuming independent days | [+0.41, +1.73] |
| SPA vs buy-and-hold | does *any* of the 272 rules beat buy-and-hold? | consistent p = 0.170 |
| StepM vs buy-and-hold | rules that significantly beat it (family-wise 5%) | 0 of 272 |
| **PBO** (CSCV, 16 blocks) | how often the in-sample best is at or below the OOS median | **0.41** |
| standard DM test, ignoring the search | verdict on `best_variant` vs buy-and-hold | `A_better_maybe_trivial` |

**Cannot be told from luck.** DSR = 0.896 (< 0.95): after accounting for the 272 variants tried, the best variant's Sharpe ratio is not distinguishable from the best of that many unskilled variants. The deflation is the difference between the PSR row and the DSR row.

*Known only because the data are simulated:* an edge does exist here, but 9.0 years of data cannot confirm it once 272 variants have been tried: the price of deflating is power.

The standard verdict for world A, as `isbetter run` produces it for a single pre-chosen
comparison (it knows nothing about the search):



Estimates are **candidate − baseline** in the metric's own units. "Oriented" values flip the sign
for lower-is-better metrics so that positive always means the candidate wins.

### best_variant vs buy_and_hold — return

**Verdict: candidate is better, but the effect may be trivial** (`A_better_maybe_trivial`)

CI is above 0 but reaches into the margin: A is better, but the gain may be trivial.


| Method | Estimate | 95.0% CI | p-value |
|---|---:|---|---:|
| diebold_mariano_hln | 0.0003344 | [1.664e-05, 0.0006522] | 0.03916 |
| stationary_block_bootstrap | 0.0003344 | [1.97e-06, 0.0006651] | n/a |

Oriented effect: 0.0003344 (CI 1.664e-05 to 0.0006522), minimum effect of interest δ = 0.0002.

**Why this method:** Diebold-Mariano test of equal accuracy: dependence=timeseries, horizon 1. The 1-step loss differential is autocorrelated, so the variance uses 0 autocovariance lag(s) (rectangular) instead of assuming independent origins; Harvey-Leybourne-Newbold small-sample correction, compared with t(2269).

- *stationary_block_bootstrap:* Stationary block bootstrap of the mean loss differential (mean block length 3, 9999 resamples): resamples blocks so serial dependence is preserved; makes no normality assumption. Block length 3 (suggested by Politis-White); automatic suggestion: stationary 3.3, circular 3.8.



## 5. How often is each tool fooled?

Unconditional behaviour over 100 fresh simulated worlds of each kind (the best of the
same 272 variants in each):

| world | mean naive t of the best variant | naive t > 1.96 | DSR > 0.95 | mean PBO |
|---|---:|---:|---:|---:|
| luck | 1.16 | 13% | 1% | 0.48 |
| trend | 2.71 | 75% | 28% | 0.38 |

In the *luck* world every flagged result is a false positive: the naive test raises one in
13% of worlds (a single pre-registered test would, by design, raise 5%), DSR
1%. In the *trend* world the edge is real and DSR finds it less often than
the naive test does (28% against 75%): the price of not
being fooled is power. PBO is 0.48 in the luck world (tuning the parameters
does not help, as it should be) and 0.38 in the trend world, only modestly
lower although the edge is real: neighbouring moving averages are near-duplicates, so *which* one
ranks first in-sample is largely chance even when they share a real edge. PBO answers "does
picking the best parameters help?", not "is there an edge?".

## 6. What would change the conclusion

* **Real data.** Equity indices trend only weakly and have positive drift; run with `--data`.
* **Transaction costs and slippage** are not modelled; they only make every rule worse.
* **Number of trials.** The DSR assumes independent trials; 272 neighbouring averages are
  far fewer independent bets, but their Sharpe spread is correspondingly small. The registry
  count and spread are used as logged. Trials run outside the registry would raise the bar:
  pass an explicit `n_trials`.
* **A longer track record**: MinTRL above says how many years the best variant would need.
* Peak memory of this run: 0.46 GB.
