# Step 6 plan (written BEFORE running): who does the email work for? (2026-11-13)

## Pre-registered hypothesis (the user's audience idea, stated before looking)
The men's email lifts purchases more among customers whose past purchases were men's items than
among customers who bought only women's items.

## Pre-registered tests (only these two count as conclusions)
Customer types from past-purchase flags (not gender): men's-only (mens=1, womens=0),
women's-only (mens=0, womens=1), both (context only, not tested).
1. Men's email: lift among men's-only minus lift among women's-only.
2. Women's email (mirror): lift among women's-only minus lift among men's-only.
Holm correction across the two. Any other slicing (region, channel, newbie, recency) is exploratory
and cannot be reported as a conclusion unless pre-stated.

## Method choice
- Chosen: difference of lifts (each lift = rate with the email minus the no-email rate within the
  same customer type), with a z-test and interval from the four cell variances. Checked against a
  statsmodels linear model with an interaction and HC0 robust standard errors (identical in a saturated
  2x2), plus a logistic interaction as a second look on the odds scale.
- Rejected: testing each type separately and comparing "significant vs not" (the difference between
  significant and not significant is not itself significant); uplift machine-learning (overkill, and
  automatic segment-searching is the multiple-comparisons problem).
- Power warning: each type has about 9,500 people per email group; detecting a difference between
  subgroup lifts needs roughly four times the sample of detecting the overall effect. A non-significant
  result would not show the email works equally for everyone. Trust check (A/A with equal lifts) and
  minimum detectable gap are part of this step.

## Predictions (user, 2026-11-13)
1. Men's email: bigger lift among men's-only buyers (the hypothesis).
2. The difference of lifts, after Holm: borderline (around the 5% line).
