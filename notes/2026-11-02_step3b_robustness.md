# Step 3b: do other tests agree? (2026-11-02)

Purchases only; each email vs no email. Script `analysis/02b_robustness.py` (seed 20261102 for the
permutation test), results `results/02b_robustness.csv`, chart `results/02b_intervals.png`.
Newcombe interval written by hand from Wilson intervals; matches statsmodels to 8+ digits.

## p-values for "the email does nothing"
| Method | Men's | Women's |
|---|---|---|
| Two-proportion z (step 2) | 1.5e-13 | 0.00016 |
| Fisher's exact | 1.2e-13 | 0.00017 |
| Logistic regression (Wald) | 6.0e-13 | 0.00018 |
| Logistic regression (likelihood ratio) | 7.8e-14 | 0.00015 |
| Permutation, 10,000 shuffles | 0.0001 (floor) | 0.0003 |

## Intervals for the lift (percentage points)
| | Wald | Newcombe |
|---|---|---|
| Men's | 0.500 to 0.861 | 0.501 to 0.864 |
| Women's | 0.150 to 0.472 | 0.150 to 0.475 |

## Reading
- 10 of 10 method results are significant at 5%; none flipped. The user predicted that some tool
  would flip the conclusion; it did not.
- The permutation p-value cannot go below 1/10,001 = 0.0001. That is a resolution limit of the
  number of shuffles, not a different answer. Women's 0.0003 vs 0.00016 is shuffling noise.
- Fisher's exact test equals the exact permutation test for a 2x2 table, so it gives the permutation
  answer without shuffling.
- Wald and Newcombe intervals differ by about 0.003 points at these counts.
- Process note: a first draft printed "any flip: True" because `~True` is -2 in Python. Caught before
  reporting; fixed by converting to bool.

## User's reading (2026-11-03)
1. Why predict a flip: "different tests serve different purposes, so some wouldn't tally."
   - Claude's response: half right. Tests that answer *different* questions (e.g. t-test on means vs
     Mann-Whitney, or a rate test vs a spend test) can disagree. Tests answering the *same*
     question (here: are the buying rates different?) should agree on a clear effect, and
     disagreement among them is a warning. They can still split near the edge (p around 0.05) or
     with small samples. Ask first: do these tests answer the same question?
2. Shuffles needed for a smaller p: "1x10^n shuffles gives about 1x10^-(n+1)".
   - Claude's response: right pattern, off by one. 10,000 = 10^4 shuffles gives a floor of
     about 1/10,001 = 10^-4, so B shuffles resolve p down to about 1/B. For p = 1.5e-13 you need
     about 10^13 shuffles (centuries at 1,000 per second), which is why Fisher's exact test counts
     all shuffles with a formula instead.
3. Next step chosen: test men's email vs women's email directly. Purchases only.
   - Prediction: the p-value will be about the same size as women's vs no email (0.00016).
