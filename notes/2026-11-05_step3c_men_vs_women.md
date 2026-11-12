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

## User's reading (2026-11-12)
1. Question: how many non-males clicked into the men's merchandise email?
   - Claude's response: the data cannot say. There is no gender column and no click column. `visit`
     means visited the website within two weeks, not clicked the email. Closest proxy: customers
     whose only past purchases were women's items (`womens`=1, `mens`=0), about 45% of customers.
     Leaky, because gifts blur it. Segment slices add comparisons (more lottery tickets), so state the
     hypothesis first: the men's email lifts purchases more among men's-history buyers than among
     women's-only buyers. Candidate follow-up step, not yet run.
2. Confidence in the men's-vs-women's result: the wide range (0.17 to 0.57 points) does not change it.
   - Claude's response: consistent. The range excludes zero, so confidence that the lift is real is
     unaffected; the range only limits how precisely we know its size.
3. Ready for step 4 (the trust check).
4. Question: when do we visualize the data?
   - Claude's response: so far charts were reading aids for results (group rates, interval plots). We
     have not yet drawn the data itself (distributions of past spend and recency, spend with its
     many zeros, balance across groups). Fair gap; options offered for where to place a visual tour.
