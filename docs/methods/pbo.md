# Probability of Backtest Overfitting (PBO-CSCV)

**Dependence:** `timeseries` with `metric.kind: returns` · **Code:** `isbetter.methods.pbo` · **Reported beside** the strategy comparison (never changes the verdict).

## When to use it

You tried many strategy configurations and picked the best in-sample. PBO asks how often the in-sample winner ends up at or below the median out-of-sample. Split the periods into S blocks; for each of the C(S, S/2) ways of choosing half as in-sample, find the in-sample best configuration, take its relative out-of-sample rank, and count the share of splits where it falls at or below the median. A PBO near 0.5 on pure noise is what you expect; a high PBO is a warning.

## What it assumes

- Every configuration that was tried is in the matrix, not only the survivors.
- Configurations are evaluated over the same periods.

## How it fails

- **A pre-filtered set of survivors** says nothing about the search that produced them.
- **One dataset's PBO is noisy**: on pure noise its standard deviation is about 0.14 because the splits overlap; only its mean over datasets is 0.5.
- **Near-duplicate configurations** inflate the grid without adding independent trials.

## Use it

```yaml
finance: {registry: out/trials.jsonl, pbo_blocks: 16, pbo_metric: sharpe}
```

With S = 16 there are 12,870 splits, computed in about a second for 2,500 periods x 400 configurations: the per-block sums are combined with a 0/1 split matrix, so one matrix product gives every split's in-sample and out-of-sample Sharpe ratios.

## Checked against

- A slow, literal transcription of the algorithm in the paper.
- On pure noise PBO is about 0.5; with one planted strong edge it is low.

## References

- Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance* 20.
