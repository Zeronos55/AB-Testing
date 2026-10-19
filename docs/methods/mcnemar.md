# McNemar's test

**Dependence:** `paired` · **Metric:** per-item right/wrong · **Code:** `isbetter.methods.classification.mcnemar` · **Inside** the paired right/wrong route (see [small N](small-n.md)).

```text
Exact: p = min(1, 2 * BinomCDF(min(b, c); b + c, 0.5))
Chi-square: (|b - c| - 1)^2 / (b + c)
```

## When to use it

Two classifiers scored right or wrong on the same test items. Only the items on which they disagree carry information: b = A right and B wrong, c = A wrong and B right. The test asks whether b and c are equally likely.

## What it assumes

- Items are independent, and the test set is fixed.
- Exact binomial when b + c < 25, otherwise chi-square with continuity correction (override with `exact=`).

## How it fails

- **It sees test-set variation only**, not the randomness of training. Dietterich (1998) recommends it for one train/test split and cross-validation based tests when you can retrain.
- **Few discordant pairs**: the test has little power, and the exact version is conservative.
- **Items from the same source** (passages, users) are not independent: use the paired cluster-robust route.
- It gives a p-value for a difference, not an interval. For an interval use the [paired score interval](small-n.md).

## Use it

```python
from isbetter.methods.classification import mcnemar
result = mcnemar(correct_a, correct_b)       # boolean arrays over the same items
```

## Checked against

- `statsmodels.stats.contingency_tables.mcnemar`.

## References

- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12.
- Dietterich, T. G. (1998), as above.
