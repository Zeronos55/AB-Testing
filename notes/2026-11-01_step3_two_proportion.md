# Step 3: is the lift real or luck? Two-proportion test (2026-11-01)

Method: pooled two-proportion z-test for the p-value, Wald 95% interval for the difference.
Written by hand in `analysis/02_two_proportion.py`; p-values, z and intervals match statsmodels
(`proportions_ztest`, `confint_proportions_2indep`) and SciPy (`chi2_contingency`, no correction)
to 8+ digits. Results: `results/02_two_proportion.csv`, chart `results/02_lift_ci.png`.

## Results (each email vs no email; differences in percentage points)
| Outcome | Group | Lift | 95% interval | p-value |
|---|---|---|---|---|
| Bought | Men's | +0.68 | 0.50 to 0.86 | 1.5e-13 |
| Bought | Women's | +0.31 | 0.15 to 0.47 | 0.00016 |
| Visited | Men's | +7.66 | 7.00 to 8.32 | 5.7e-112 |
| Visited | Women's | +4.52 | 3.89 to 5.16 | 3.2e-44 |

## Predictions vs result
- User: men's lift is real. Correct. User: women's lift also not luck. Correct on the conclusion.
- Both emails lifted purchases and visits well beyond what chance gives for equal groups.

## Limits (not yet tested)
- "Men's beat women's" was not tested directly (men's vs women's, purchases). Non-overlapping
  intervals are only a hint.
- A small p-value says "not luck", not "why" and not "big enough to matter for the business".
- The test assumes each person's outcome is independent and the split was random (the balance
  check supports the second).
- Four tests were run; with p-values this small, correcting for that would not change the answers.

## User's reading
(to fill in after discussion)
