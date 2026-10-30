# Multiple comparisons: Bonferroni, Holm, Benjamini-Hochberg

**Applies to:** several candidates or several metrics · **Code:** `isbetter.methods.multiplicity` · **Set by** `decision.correction`.

```text
Holm: sort p; reject p_(i) while p_(i) <= alpha / (m - i + 1); stop at the first failure
BH:   largest k with p_(k) <= k alpha / m; reject hypotheses 1..k
```

## When to use it

More than one hypothesis is tested and you will act on any that look significant: three models against one baseline, or several success metrics. Each method returns adjusted p-values, and the report shows them with the per-comparison alpha used for the intervals.

## What it assumes

- **Bonferroni and Holm** control the probability of any false positive (family-wise error) under any dependence between the tests. Holm is never less powerful than Bonferroni.
- **Benjamini-Hochberg** controls the expected share of false discoveries, and is valid when the tests are independent or positively dependent.

## How it fails

- **Choosing the family after looking**: the family is the set of hypotheses you commit to in advance. Dropping inconvenient comparisons afterwards defeats the correction.
- **Reading BH as family-wise control**: it allows some false positives by design.
- **Guardrails are deliberately not corrected**: adjusting them would make a regression easier to miss. See [guardrails](guardrails.md).
- **Trading strategies**: Holm and BH are refused there, because the [Deflated Sharpe Ratio](sharpe-psr-dsr.md) is what accounts for the search over variants.

## Use it

```yaml
decision: {alpha: 0.05, correction: holm}   # none | holm | bonferroni | bh
```

In an A/B experiment the family is every (candidate x success metric) hypothesis.

## Checked against

- `statsmodels.stats.multitest.multipletests` on 1,000 random p-value vectors (fewer for Holm, which is slow in statsmodels).

## References

- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B* 57.
