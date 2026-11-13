# Which test? A decision guide

Rule of thumb: choose the test that matches **how the data was collected**, not the one you
remember. Ask these five questions in order.

1. **What is the outcome?** Yes/no, a number, a count, time until an event, or a ratio of two numbers.
2. **How are the groups related?** Different people (independent), the same people measured twice
   or matched (paired), people with many rows each (clustered), or ordered in time.
3. **How much data?** Large samples allow shortcut formulas; small ones need exact methods.
4. **What do you want to report?** A p-value, the size of the effect with a range, or a probability.
5. **How many comparisons are you making?** More looks means more chances for a false alarm.

| Situation | Good options | Watch out for |
|---|---|---|
| Yes/no, two independent groups, large samples | Two-proportion z-test (same as chi-square on a 2x2 table). Interval: Newcombe or Wilson-based | Needs roughly 10+ successes and failures in each group |
| Yes/no, small samples or tiny counts | Fisher's exact test; exact or Wilson-type intervals | The textbook (Wald) interval is poor with few events |
| Yes/no, same people twice or matched pairs | McNemar's test | Ignoring the pairing throws away power |
| Yes/no, want to adjust for other columns, or compare many groups at once | Logistic regression; chi-square on the full table first | Gives odds ratios, not point differences, unless converted |
| A number, two independent groups, well behaved | Welch t-test (does not assume equal spread) | Heavy tails or outliers |
| A number that is skewed or mostly zeros (like spend) | Bootstrap interval for the difference in means; permutation test; or split into "bought or not" and "how much" | Mann-Whitney answers a different question (which group tends to be higher), not the difference in means |
| A number, same people twice | Paired t-test; paired bootstrap; sign-flip permutation | |
| Counts per person | Poisson or negative-binomial regression; mean difference if samples are large | Counts with a lot of spread |
| Ratio of two totals (revenue per session) | Delta method or bootstrap by user | Treating each row as independent |
| Many rows per user or per group | Analyze per cluster, or cluster-robust errors | The naive test is far too confident |
| Several comparisons | Holm (controls any false alarm) or Benjamini-Hochberg (controls the share of false alarms among the wins) | Picking the best-looking result after the fact |
| Checking results while data comes in | Sequential methods (confidence sequences) | Repeated fixed-sample tests inflate false alarms |
| Time series, forecasts | Diebold-Mariano; block bootstrap | Treating autocorrelated points as independent |
| Comparing ML models with cross-validation | Corrected resampled t (Nadeau-Bengio) | The naive k-fold t-test is badly overconfident |
