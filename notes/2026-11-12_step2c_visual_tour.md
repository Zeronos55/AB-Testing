# Step 2c: visual tour of the data (2026-11-12)

Script `analysis/01c_visual_tour.py`. Charts: `results/01c_distributions.png`,
`results/01c_balance.png`; table `results/01c_balance_smd.csv`. No outcomes by customer segment
were drawn on purpose (hypothesis first: the men's email lifts purchases more among men's-history
buyers than among women's-only buyers).

## What the charts show
- Recency (months since last purchase): 1 to 12; most customers at 1 month (8,952), a second bump at
  months 9 and 10 (6,441 and 7,565) with a dip at 8. Cause unknown.
- Past-year spend: right-skewed (median $158, mean $242). 7,947 customers (12%) have exactly
  $29.99, the minimum. Top 20% of customers account for 54% of past-year spend.
- Spend after the email: 63,422 customers (99.1%) spent $0; 578 spent. Among spenders, 136 (24%)
  spent exactly $29.99 and 12 spent exactly $499 (the maximum). Top 20% of spenders account for 51%
  of post-email spend. (Earlier note said "over 98%" zero; 99.1% is the exact figure.)
- Balance: largest standardized difference between any email group and the no-email group is 0.014
  (rural, men's group); the usual threshold is 0.1. The random split left the groups alike.

## Open questions (not explained by the data)
- Is $29.99 one standard-priced item? No product column, so unverified.
- The 12 spenders at exactly $499: a cap, a bundle, or a data quirk?

## Predictions for step 4 (trust check), recorded 2026-11-12
- 21,000 per group: about 5% false alarms (calibrated).
- 500 per group (about 3 buyers each): the test breaks, far from 5%.

## User's reading
(to fill in after discussion)
