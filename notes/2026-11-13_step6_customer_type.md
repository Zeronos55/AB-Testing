# Step 6: who does the email work for? (2026-11-13)

Pre-registered in `notes/2026-11-13_step6_customer_type_plan.md` (committed before running).
Script `analysis/06_customer_type.py`, seed 20261113. Results: `results/06_lifts_by_type.csv`,
`results/06_difference_of_lifts.csv`, `results/06_trust_and_power.csv`, chart `results/06_customer_type.png`.
Customer types come from past-purchase flags (not gender). They are measured before the email, so
comparing within a type is still a randomized comparison. Difference of lifts, checked against a
statsmodels interaction model (HC0): identical to 1e-8. Holm across the two pre-registered tests.

## Lift in purchases within each type (points vs no email in the same type, 95% Newcombe)
| Email | Men's-only | Women's-only | Both (context only) |
|---|---|---|---|
| Men's | +0.60 (0.34 to 0.87) | +0.57 (0.32 to 0.83) | +1.50 (0.74 to 2.30) |
| Women's | +0.06 (-0.15 to 0.28) | +0.51 (0.27 to 0.77) | +0.53 (-0.13 to 1.22) |
Cell sizes: about 9,600 per email group for the two main types, about 2,150 for "both".

## Pre-registered tests (difference of lifts)
| Email | Contrast | Difference (points) | 95% interval | p | Holm p |
|---|---|---|---|---|---|
| Men's | men's-only minus women's-only | +0.03 | -0.33 to 0.39 | 0.88 | 0.88 |
| Women's | women's-only minus men's-only | +0.45 | 0.12 to 0.78 | 0.0073 | 0.0146 |
Logistic interaction (odds scale, second look): p = 0.97 and 0.023 (about 0.047 after Holm: borderline).

## Trust check and power for this method (real cell sizes, 10,000 fake experiments)
- No real gap: claims one 4.8% (men's email) and 4.9% (women's email) of the time. Calibrated.
- Smallest gap detectable with 80% power: 0.56 and 0.51 points (simulation confirms: 80.6%, 80.1%).

## Predictions vs result
1. Men's email lifts men's-only more: numerically 0.60 vs 0.57 but the gap is 0.03 points, p = 0.88.
   Hypothesis not supported; the lifts are about equal.
2. Difference after Holm "borderline": wrong for the men's email (p = 0.88, nowhere near 0.05).

## Reading
- Men's email: works about equally for both types. Interval of the gap (-0.33 to 0.39) rules out
  large differences but, with a detectable gap of 0.56, not modest ones.
- Women's email: works for women's-only buyers, with no detectable lift among men's-only buyers. The
  difference survives correction (Holm 0.015), but the observed 0.45 is below the 0.51 gap that
  80% power needs, so winner's-curse exaggeration is likely; the true gap is probably smaller.
- "Both" buyers: large men's-email lift, exploratory only, not a conclusion.
- Not tested: why; types are purchase histories, not gender.

## User's reading
(to fill in after discussion)
