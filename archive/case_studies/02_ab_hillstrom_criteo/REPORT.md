# Case study 2: an email A/B test and a Criteo-scale ad test

> **Data:** both datasets are SYNTHETIC stand-ins here (the real files could not be downloaded
> in the build environment). Rerun with `--hillstrom` and `--criteo` for the real analysis.
> Nothing below is a claim about real campaigns.

## 1. Question

Should either email variant ("Mens E-Mail", "Womens E-Mail") replace sending no email? Success is
a lift in site visits or purchase conversion (a lift smaller than 1 percentage point of visits or
0.2 points of conversion is treated as trivial). The guardrail is revenue: spend per customer must
not fall by more than $0.10.

## 2. Data and dependence structure

**SYNTHETIC stand-in** shaped like the Hillstrom email challenge (64,000 customers, three arms of 21,306 / 21,307 / 21,387, zero-inflated spend). Effect sizes are assumed. One row per customer, randomised once to one of three arms, so units are independent:
`dependence = iid`. Arm counts: No E-Mail: 21,306, Mens E-Mail: 21,307, Womens E-Mail: 21,387.
Last year's spend (`history`) is measured before the experiment and is the CUPED covariate.

## 3. Naive analysis, and why it misleads

The naive recipe runs each metric and arm as its own 95% test, ignores the pre-experiment
covariate and never checks that randomisation delivered the planned split:

| metric | candidate | naive diff | naive 95% CI | naive p | corrected CI | Holm-adjusted p |
|---|---|---:|---|---:|---|---:|
| visit | Mens E-Mail | +0.0752 | [+0.0684, +0.0820] | 9.01e-104 | [+0.0665, +0.0838] (98.8%) | 3.6e-103 |
| visit | Womens E-Mail | +0.0484 | [+0.0419, +0.0549] | 1.68e-47 | [+0.0401, +0.0567] (98.8%) | 5.03e-47 |
| conversion | Mens E-Mail | +0.0088 | [+0.0066, +0.0109] | 6.69e-16 | [+0.0062, +0.0116] (98.8%) | 3.44e-16 |
| conversion | Womens E-Mail | +0.0045 | [+0.0026, +0.0064] | 5.59e-06 | [+0.0022, +0.0071] (98.8%) | 2.4e-06 |
| spend | Mens E-Mail | +0.7818 | [+0.5232, +1.0404] | 3.12e-09 | [+0.5838, +1.0141] (90.0%) | n/a (guardrail) |
| spend | Womens E-Mail | +0.3850 | [+0.1539, +0.6161] | 0.0011 | [+0.2132, +0.5975] (90.0%) | n/a (guardrail) |

* **Multiplicity.** 4 of 4 success hypotheses are naively significant;
  4 of 4 remain wins after Holm. When effects are large the correction costs
  wider intervals but changes no conclusion; with several weak metrics it would. The correction
  spans success hypotheses only: guardrails are tested one-sided at the unadjusted level so that
  a regression is not made easier to miss.
* **Ignored covariate.** Leaving `history` out of the analysis forgoes some precision; section 5
  measures how much.
* **No SRM check.** Effect tests cannot detect broken assignment. If 3,000 customers vanished from
  `Womens E-Mail` through a logging bug the SRM check would report p = 2.1e-61
  (flagged) while every effect estimate would still
  look fine.

## 4. Correct analysis

Automatic sample-ratio check, Holm across the 4 success hypotheses, one-sided
non-inferiority for the guardrail, then one decision per candidate.

### Sample-ratio check

**Passed.** Chi-square goodness of fit of arm counts (No E-Mail: 21,306, Mens E-Mail: 21,307, Womens E-Mail: 21,387) against the planned split (No E-Mail: 33.3%, Mens E-Mail: 33.3%, Womens E-Mail: 33.3%): chi2 = 0.2025, df = 2, p = 0.904 vs threshold 0.001; no sample-ratio mismatch detected.

| Arm | Observed | Expected | Planned share |
|---|---:|---:|---:|
| No E-Mail | 21,306 | 21,333.3 | 33.3% |
| Mens E-Mail | 21,307 | 21,333.3 | 33.3% |
| Womens E-Mail | 21,387 | 21,333.3 | 33.3% |

### Decision

| Candidate | Decision | Why |
|---|---|---|
| Mens E-Mail | **SHIP** | success metric(s) won: visit, conversion; all 1 guardrail(s) passed |
| Womens E-Mail | **SHIP** | success metric(s) won: visit, conversion; all 1 guardrail(s) passed |

#### Mens E-Mail vs No E-Mail

| Metric | Role | Result | Effect | CI | Note |
|---|---|---|---:|---|---|
| visit | success | win | 0.07518 | [0.06655, 0.08381] | improvement is statistically and practically significant |
| conversion | success | win | 0.008897 | [0.006201, 0.01159] | improvement is statistically and practically significant |
| spend | guardrail | pass | 0.799 | [0.5838, 1.014] | non-inferior: regression larger than 0.1 is ruled out |

#### Womens E-Mail vs No E-Mail

| Metric | Role | Result | Effect | CI | Note |
|---|---|---|---:|---|---|
| visit | success | win | 0.04838 | [0.04006, 0.05671] | improvement is statistically and practically significant |
| conversion | success | win | 0.004645 | [0.002185, 0.007105] | improvement is statistically and practically significant |
| spend | guardrail | pass | 0.4053 | [0.2132, 0.5975] | non-inferior: regression larger than 0.1 is ruled out |


Effects are **candidate − baseline**, oriented so positive means the candidate wins. Success-metric
intervals carry the multiplicity correction (holm across all success
hypotheses). Guardrail intervals are two-sided 90% intervals, i.e. a one-sided
test at the unadjusted α = 0.05; a guardrail passes only if the lower limit is above minus its margin.

### Results by metric

#### visit (success)

##### Mens E-Mail vs No E-Mail — visit

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.01.


| Method | Estimate | 98.8% CI | p-value |
|---|---:|---|---:|
| two_proportion_z | 0.07518 | [0.06655, 0.08381] | 9.01e-104 |
| fisher_exact | 0.07518 | n/a | 1.43e-104 |

Multiplicity (holm across 4 hypotheses): adjusted p = 3.604e-103; CIs use per-comparison alpha = 0.0125.

Oriented effect: 0.07518 (CI 0.06655 to 0.08381), minimum effect of interest δ = 0.01.

**Why this method:** Two-proportion z-test (binary metric, independent units): the p-value uses the pooled variance under H0; the CI is a Wald interval with unpooled variance.

- *fisher_exact:* Fisher's exact test: exact p-value conditional on the table margins (cross-check; valid for any counts).

##### Womens E-Mail vs No E-Mail — visit

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.01.


| Method | Estimate | 98.8% CI | p-value |
|---|---:|---|---:|
| two_proportion_z | 0.04838 | [0.04006, 0.05671] | 1.676e-47 |
| fisher_exact | 0.04838 | n/a | 1.237e-47 |

Multiplicity (holm across 4 hypotheses): adjusted p = 5.029e-47; CIs use per-comparison alpha = 0.0125.

Oriented effect: 0.04838 (CI 0.04006 to 0.05671), minimum effect of interest δ = 0.01.

**Why this method:** Two-proportion z-test (binary metric, independent units): the p-value uses the pooled variance under H0; the CI is a Wald interval with unpooled variance.

- *fisher_exact:* Fisher's exact test: exact p-value conditional on the table margins (cross-check; valid for any counts).

#### conversion (success)

##### Mens E-Mail vs No E-Mail — conversion

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.002.


| Method | Estimate | 98.8% CI | p-value |
|---|---:|---|---:|
| cuped | 0.008897 | [0.006201, 0.01159] | 1.719e-16 |
| two_proportion_z | 0.008776 | [0.006064, 0.01149] | 6.692e-16 |

Multiplicity (holm across 4 hypotheses): adjusted p = 3.438e-16; CIs use per-comparison alpha = 0.0125.

Oriented effect: 0.008897 (CI 0.006201 to 0.01159), minimum effect of interest δ = 0.002.

**Why this method:** CUPED: outcome adjusted with a pre-experiment covariate (theta = 7.305e-05, within-arm correlation rho = 0.110); randomisation keeps the estimate unbiased while the variance shrinks by about 1 - rho^2.

- *two_proportion_z:* Unadjusted comparison, for reference. Two-proportion z-test (binary metric, independent units): the p-value uses the pooled variance under H0; the CI is a Wald interval with unpooled variance.

##### Womens E-Mail vs No E-Mail — conversion

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.002.


| Method | Estimate | 98.8% CI | p-value |
|---|---:|---|---:|
| cuped | 0.004645 | [0.002185, 0.007105] | 2.404e-06 |
| two_proportion_z | 0.004504 | [0.002028, 0.006979] | 5.592e-06 |

Multiplicity (holm across 4 hypotheses): adjusted p = 2.404e-06; CIs use per-comparison alpha = 0.0125.

Oriented effect: 0.004645 (CI 0.002185 to 0.007105), minimum effect of interest δ = 0.002.

**Why this method:** CUPED: outcome adjusted with a pre-experiment covariate (theta = 6.984e-05, within-arm correlation rho = 0.113); randomisation keeps the estimate unbiased while the variance shrinks by about 1 - rho^2.

- *two_proportion_z:* Unadjusted comparison, for reference. Two-proportion z-test (binary metric, independent units): the p-value uses the pooled variance under H0; the CI is a Wald interval with unpooled variance.

#### spend (guardrail)

*Guardrail: the margin below is the non-inferiority margin, not an equivalence band; read the decision above rather than the verdict label.*

##### Mens E-Mail vs No E-Mail — spend

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.1.


| Method | Estimate | 90.0% CI | p-value |
|---|---:|---|---:|
| cuped | 0.799 | [0.5838, 1.014] | 1.023e-09 |
| welch_t | 0.7818 | [0.5648, 0.9988] | 3.125e-09 |

Oriented effect: 0.799 (CI 0.5838 to 1.014), minimum effect of interest δ = 0.1.

**Why this method:** CUPED: outcome adjusted with a pre-experiment covariate (theta = 0.01036, within-arm correlation rho = 0.129); randomisation keeps the estimate unbiased while the variance shrinks by about 1 - rho^2.

- *welch_t:* Unadjusted comparison, for reference. Welch t-test: independent units (dependence=iid); does not assume equal variances.

##### Womens E-Mail vs No E-Mail — spend

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.1.


| Method | Estimate | 90.0% CI | p-value |
|---|---:|---|---:|
| cuped | 0.4053 | [0.2132, 0.5975] | 0.0005199 |
| welch_t | 0.385 | [0.191, 0.5789] | 0.001095 |

Oriented effect: 0.4053 (CI 0.2132 to 0.5975), minimum effect of interest δ = 0.1.

**Why this method:** CUPED: outcome adjusted with a pre-experiment covariate (theta = 0.01007, within-arm correlation rho = 0.137); randomisation keeps the estimate unbiased while the variance shrinks by about 1 - rho^2.

- *welch_t:* Unadjusted comparison, for reference. Welch t-test: independent units (dependence=iid); does not assume equal variances.


## 5. Verdict, effect size, CI, CUPED and `n_needed`

The decisions and effect sizes with confidence intervals are in section 4. `conversion` and
`spend` are analysed with CUPED using `history`; widths below are at the same confidence level,
with the unadjusted interval as the comparison:

| metric | candidate | CI width, no CUPED | CI width, CUPED | narrower by | variance cut | rho^2 |
|---|---|---:|---:|---:|---:|---:|
| conversion | Mens E-Mail | 0.0054 | 0.0054 | 0.6% | 1.2% | 1.2% |
| conversion | Womens E-Mail | 0.0050 | 0.0049 | 0.6% | 1.3% | 1.3% |
| spend | Mens E-Mail | 0.4340 | 0.4304 | 0.8% | 1.7% | 1.7% |
| spend | Womens E-Mail | 0.3879 | 0.3842 | 0.9% | 1.9% | 1.9% |

CUPED narrows the intervals by 0.6% to 0.9% here. The gain is capped by rho^2, the squared within-arm correlation between `history` and the outcome (|rho| is 0.11 to 0.14): a covariate that barely predicts a zero-inflated outcome cannot buy much precision, and a stronger pre-period metric would.

More data needed where a comparison is inconclusive:

No comparison is inconclusive at this sample size, so no extra data is needed.

Planning: with the control conversion rate of 0.0084 observed here, detecting a +0.2 point lift
at 80% power needs about **36,264 customers per arm** (`isbetter plan-size --kind
proportions --baseline 0.0084 --mde 0.002`).

## 6. What would change the conclusion

* A failed SRM check (a real one, not the planted example above) invalidates every number here.
* A real guardrail such as unsubscribes or complaints. The data have none; spend per customer
  stands in to show the rule, and a guardrail with a narrower margin needs more data to pass.
* Customers who receive several emails or appear in several rows would make `dependence = iid`
  wrong; declare `clustered` with a customer id (see failure gallery #5 for the damage ignoring
  that does).
* Real data: the stand-in effect sizes are assumptions.

## Appendix: scale test, a Criteo-style table aggregated without loading it

**SYNTHETIC stand-in**: 14,000,000 rows with columns `treatment, visit, conversion`, about 85% treated and rare binary outcomes. Rates are assumptions, not measurements.

The file is never loaded into memory: DuckDB streams it and keeps only n, sum and sum of squares
per arm (`table.mode: stats`), and the same two-proportion z-tests, Holm family and decision rule
run from those 12 numbers.

* rows aggregated: **14,000,000**
* aggregation + analysis time: **1.6 s**
* peak resident memory of the analysis process: **276 MB** (includes the Python and
  library imports; the target is under 1,024 MB)

### Sample-ratio check

**Passed.** Chi-square goodness of fit of arm counts (0: 2,101,619, 1: 11,898,381) against the planned split (0: 15.0%, 1: 85.0%): chi2 = 1.468, df = 1, p = 0.226 vs threshold 0.001; no sample-ratio mismatch detected.

| Arm | Observed | Expected | Planned share |
|---|---:|---:|---:|
| 0 | 2,101,619 | 2,100,000.0 | 15.0% |
| 1 | 11,898,381 | 11,900,000.0 | 85.0% |

### Decision

| Candidate | Decision | Why |
|---|---|---|
| 1 | **SHIP** | success metric(s) won: visit, conversion |

#### 1 vs 0

| Metric | Role | Result | Effect | CI | Note |
|---|---|---|---:|---|---|
| visit | success | win | 0.006685 | [0.006351, 0.007018] | improvement is statistically and practically significant |
| conversion | success | win | 0.001039 | [0.0009652, 0.001113] | improvement is statistically and practically significant |


Effects are **candidate − baseline**, oriented so positive means the candidate wins. Success-metric
intervals carry the multiplicity correction (holm across all success
hypotheses). Guardrail intervals are two-sided 90% intervals, i.e. a one-sided
test at the unadjusted α = 0.05; a guardrail passes only if the lower limit is above minus its margin.

### Results by metric

#### visit (success)

##### 1 vs 0 — visit

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.002.


| Method | Estimate | 97.5% CI | p-value |
|---|---:|---|---:|
| two_proportion_z | 0.006685 | [0.006351, 0.007018] | <1e-300 |
| fisher_exact | 0.006685 | n/a | <1e-300 |

Multiplicity (holm across 2 hypotheses): adjusted p = <1e-300; CIs use per-comparison alpha = 0.025.

Oriented effect: 0.006685 (CI 0.006351 to 0.007018), minimum effect of interest δ = 0.002.

**Why this method:** Two-proportion z-test (binary metric, independent units): the p-value uses the pooled variance under H0; the CI is a Wald interval with unpooled variance.

- *fisher_exact:* Fisher's exact test: exact p-value conditional on the table margins (cross-check; valid for any counts).

#### conversion (success)

##### 1 vs 0 — conversion

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.0005.


| Method | Estimate | 97.5% CI | p-value |
|---|---:|---|---:|
| two_proportion_z | 0.001039 | [0.0009652, 0.001113] | 3.846e-159 |
| fisher_exact | 0.001039 | n/a | 2.871e-176 |

Multiplicity (holm across 2 hypotheses): adjusted p = 3.846e-159; CIs use per-comparison alpha = 0.025.

Oriented effect: 0.001039 (CI 0.0009652 to 0.001113), minimum effect of interest δ = 0.0005.

**Why this method:** Two-proportion z-test (binary metric, independent units): the p-value uses the pooled variance under H0; the CI is a Wald interval with unpooled variance.

- *fisher_exact:* Fisher's exact test: exact p-value conditional on the table margins (cross-check; valid for any counts).


### Planted logging bug

The same analysis on 2,000,000 rows where about 2% of treated rows were lost
(treated share 84.7% instead of the planned 85%) with the planned split left unchanged:
SRM **FAILED** (p = 1.1e-33), so every decision is blocked: 1: DO NOT SHIP.
The effect estimates alone would not have revealed the problem.

### Verify before trusting real Criteo numbers

The build environment could not reach the Criteo host, so the dataset URL and layout were **not
verified here**. Before using `--criteo`, check (a) the download URL on the Criteo AI Lab uplift
dataset page, (b) that the file has `treatment`, `visit` and `conversion` columns, and (c)
whether rows are users or impressions. If a user can appear in several rows, the units are not
independent: aggregate to user level or declare `dependence: clustered` with a user id. As far as
I recall the public file has no user id column, in which case user-level clustering cannot be
corrected from the file alone and the intervals should be read as optimistic. This is
unverified.
