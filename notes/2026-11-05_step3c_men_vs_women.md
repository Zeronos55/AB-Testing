# Step 3c: men's email vs women's email, purchases (2026-11-05)

Script `analysis/03_men_vs_women.py`; results `results/03_pairwise_purchases.csv`, chart
`results/03_pairwise_lifts.png`. Two-proportion z-test (by hand, checked against statsmodels),
Fisher's exact, Newcombe interval (by hand, checked against statsmodels), Holm adjustment
(by hand, checked against statsmodels `multipletests`). Purchases only (user's choice).

Overall test (3 groups x bought/not): chi-square 55.3, 2 degrees of freedom, p = 1e-12.

| Comparison | Rates | Difference (points) | 95% Newcombe | Raw p | Fisher p | Holm p |
|---|---|---|---|---|---|---|
| Men's vs none | 1.253% vs 0.573% | +0.68 | 0.50 to 0.86 | 1.5e-13 | 1.2e-13 | 4.6e-13 |
| Women's vs none | 0.884% vs 0.573% | +0.31 | 0.15 to 0.47 | 0.00016 | 0.00017 | 0.00031 |
| Men's vs women's | 1.253% vs 0.884% | +0.37 | 0.17 to 0.57 | 0.00021 | 0.00023 | 0.00031 |

## Prediction vs result
- User predicted the men's-vs-women's p-value would be about the same size as women's-vs-none
  (0.00016). Correct: 0.00021 raw.
- Why not smaller despite a bigger gap (0.37 vs 0.31 points): z = gap / noise, and the noise is
  larger when both groups have higher buying rates (z 3.71 vs 3.78).

## Reading
- Men's beats women's: the lift is real after correcting for three comparisons (Holm 0.00031).
- The plausible gap is wide: 0.17 to 0.57 points, a factor of more than 3 between ends.
- Holm gives women's-vs-none and men's-vs-women's the same adjusted p (0.00031) because the
  adjustment never lets a larger raw p get a smaller adjusted one.
- Not answered: why. The two emails differ in merchandise, and probably also in offer, wording and
  design; we cannot separate these.

## User's reading
(to fill in after discussion)
