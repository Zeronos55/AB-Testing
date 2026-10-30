# Case study 3: forecasting daily sales (M5-style)

> **Data:** SYNTHETIC M5-style stand-in -- NOT real retail data.
> The real M5 file could not be downloaded here; rerun with `--data sales_train_evaluation.csv` for the real analysis. Nothing below is a claim about real retail demand.

## 1. Question

Is ARIMA or a global LightGBM model *really* more accurate than the seasonal-naive forecast
("same weekday last week") for daily sales 14 days ahead, and which of the three can we
rule out? Accuracy is the mean absolute scaled error (MASE: 1.0 means as accurate as seasonal
naive on the first 730 days). A difference smaller than **0.02 MASE**
is treated as practically irrelevant.

## 2. Data and dependence structure

7 daily series (1 store(s) x 7 departments, 1096 days). Every model is
re-run from 353 consecutive daily forecast origins (days 730 to
1082), forecasting up to 14 days ahead; ARIMA and LightGBM are refitted every
28 origins. Per-series scaled errors are averaged over the series, giving **one loss per
model, origin and horizon**. We test over origins, not series, because series in a store move
together (a shared store-level shock), so they are not independent replicates.

The origins are not independent either. A forecast made on day *t* for day *t+h* and one made on
day *t+1* for day *t+1+h* are built from almost the same information, so the loss differential
of **h-step** forecasts is autocorrelated at lags 1 to h-1. That is `dependence = timeseries`
with `horizon = h`, and it picks the Diebold-Mariano test with the Harvey-Leybourne-Newbold
correction (`h - 1` autocovariance lags).

## 3. Naive analysis, and why it misleads

A paired t-test on the loss differential treats the 353 origins as independent
observations. At horizon 1 that is exactly right (the two p-values below are identical). At longer
horizons the differential is autocorrelated, and the naive test can be badly overconfident: in
the failure gallery (#2b, 100 origins) it falsely flags two equally good forecasters 17% of the
time at h=3 and 46% at h=10. How much it matters depends on the data: the last column is the
standard error of the Diebold-Mariano test relative to the naive one (above 1: the naive test was
too confident; below 1: too cautious).

| horizon | candidate | mean MASE diff vs seasonal_naive | naive p | DM p | SE ratio DM/naive |
|---:|---|---:|---:|---:|---:|
| 1 | arima | -0.2564 | 5.64e-46 | 5.64e-46 | 1.00 |
| 1 | lightgbm | -0.3013 | 2.07e-55 | 2.07e-55 | 1.00 |
| 7 | arima | -0.2392 | 8.1e-41 | 2.66e-32 | 1.16 |
| 7 | lightgbm | -0.2966 | 1.94e-55 | 2.56e-48 | 1.10 |
| 14 | arima | -0.2218 | 8.37e-32 | 1.75e-26 | 1.12 |
| 14 | lightgbm | -0.2903 | 6.14e-55 | 4.3e-59 | 0.95 |

The two candidates are both far ahead of seasonal-naive, so the naive test and the correct test
agree there. The informative comparison is the close one, LightGBM against ARIMA (positive effect:
LightGBM is more accurate, in MASE units; 95% CIs):

| horizon | LightGBM minus ARIMA accuracy gain | 95% CI (Diebold-Mariano) | naive p | DM p |
|---:|---:|---|---:|---:|
| 1 | +0.0449 | [+0.0324, +0.0574] | 8.91e-12 | 8.91e-12 |
| 7 | +0.0574 | [+0.0450, +0.0698] | 5.38e-18 | 6.14e-18 |
| 14 | +0.0686 | [+0.0502, +0.0869] | 9.14e-20 | 1.53e-12 |

## 4. Correct analysis (Diebold-Mariano + HLN, Holm across the 2 candidates)

Headline at horizon 7 (one-week-ahead); the same analysis at horizons 1 and 14 is summarised
below the report.



Estimates are **candidate − baseline** in the metric's own units. "Oriented" values flip the sign
for lower-is-better metrics so that positive always means the candidate wins.

### arima vs seasonal_naive — mase_h7

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.02.


| Method | Estimate | 97.5% CI | p-value |
|---|---:|---|---:|
| diebold_mariano_hln | -0.2392 | [-0.2802, -0.1982] | 2.664e-32 |
| stationary_block_bootstrap | -0.2392 | [-0.2781, -0.2013] | n/a |

Multiplicity (holm across 2 candidates): adjusted p = 2.664e-32; CIs use per-comparison alpha = 0.025.

Oriented effect: 0.2392 (CI 0.1982 to 0.2802), minimum effect of interest δ = 0.02.

**Why this method:** Diebold-Mariano test of equal accuracy: dependence=timeseries, horizon 7. The 7-step loss differential is autocorrelated, so the variance uses 6 autocovariance lag(s) (rectangular) instead of assuming independent origins; Harvey-Leybourne-Newbold small-sample correction, compared with t(352).

- *stationary_block_bootstrap:* Stationary block bootstrap of the mean loss differential (mean block length 2, 9999 resamples): resamples blocks so serial dependence is preserved; makes no normality assumption. Block length 2 (suggested by Politis-White); automatic suggestion: stationary 2.5, circular 2.8.

### lightgbm vs seasonal_naive — mase_h7

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.02.


| Method | Estimate | 97.5% CI | p-value |
|---|---:|---|---:|
| diebold_mariano_hln | -0.2966 | [-0.3356, -0.2577] | 2.556e-48 |
| stationary_block_bootstrap | -0.2966 | [-0.3343, -0.2598] | n/a |

Multiplicity (holm across 2 candidates): adjusted p = 5.112e-48; CIs use per-comparison alpha = 0.025.

Oriented effect: 0.2966 (CI 0.2577 to 0.3356), minimum effect of interest δ = 0.02.

**Why this method:** Diebold-Mariano test of equal accuracy: dependence=timeseries, horizon 7. The 7-step loss differential is autocorrelated, so the variance uses 6 autocovariance lag(s) (rectangular) instead of assuming independent origins; Harvey-Leybourne-Newbold small-sample correction, compared with t(352).

- *stationary_block_bootstrap:* Stationary block bootstrap of the mean loss differential (mean block length 2, 9999 resamples): resamples blocks so serial dependence is preserved; makes no normality assumption. Block length 2 (suggested by Politis-White); automatic suggestion: stationary 2.5, circular 2.8.


**All three horizons** (oriented effect > 0 means the candidate is more accurate):

| horizon | candidate | effect in MASE | 95% CI (Holm level) | verdict |
|---:|---|---:|---|---|
| 1 | arima | +0.2564 | [+0.2215, +0.2912] | `A_better_practical` |
| 1 | lightgbm | +0.3013 | [+0.2653, +0.3372] | `A_better_practical` |
| 7 | arima | +0.2392 | [+0.1982, +0.2802] | `A_better_practical` |
| 7 | lightgbm | +0.2966 | [+0.2577, +0.3356] | `A_better_practical` |
| 14 | arima | +0.2218 | [+0.1786, +0.2649] | `A_better_practical` |
| 14 | lightgbm | +0.2903 | [+0.2573, +0.3234] | `A_better_practical` |

## 5. Which models can we rule out? (SPA, StepM and the Model Confidence Set at horizon 7)

Mean MASE at horizon 7: `seasonal_naive` 1.0344, `arima` 0.7952, `lightgbm` 0.7378.

* **Hansen's SPA** (is *any* of ARIMA / LightGBM better than seasonal naive, accounting for having
  looked at both?): consistent p < 0.001
  (bootstrap with 1,000 resamples).
* **Romano-Wolf StepM** (which of them are significantly better, family-wise error 5%):
  `arima` and `lightgbm`.
* **Model Confidence Set** (90%; block bootstrap with Politis-White block length
  2): The 90% Model Confidence Set contains **only `lightgbm`**: it is significantly better than `arima` and `seasonal_naive`, and the data single it out as the best model.
  MCS p-values: `seasonal_naive` 0.000, `arima` 0.000, `lightgbm` 1.000.

*In plain English:* the Model Confidence Set is the list of models that are still in the running.
A model is dropped only if it is clearly worse than the best one; models that stay in are not
"equally good", they are simply not distinguishable with this much data. Being outside the set is
evidence of being worse; being inside is not evidence of being best. (Politis-White suggests a mean
block length of 0.05 for the ARIMA-vs-LightGBM loss differential; a value below 1
means the estimated autocorrelation is negligible, so blocks of length 1 are used.)

## 6. What would change the conclusion

* More series or more years of evaluation: the verdicts above rest on 353
  strongly overlapping origins, so the `n_needed` note in the report says how much more is needed
  for the inconclusive comparisons.
* A different margin: with a larger minimum effect more comparisons become "equivalent".
* Real data: promotions, SNAP days, zero-inflated items and price changes are all absent from the
  synthetic generator and could favour LightGBM (which can use them) over ARIMA.
* Run time and memory: peak RSS of this run was 0.26 GB; 0 warning(s) were raised
  by ARIMA fits (all fits were kept).
