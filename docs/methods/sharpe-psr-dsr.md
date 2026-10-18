# Sharpe ratio: PSR, DSR, MinTRL and the block-bootstrap interval

**Dependence:** `timeseries` with `metric.kind: returns` · **Code:** `isbetter.methods.finance` · **Routed** through `adapter: strategy_returns` with a `finance:` block.

```text
PSR(SR*) = Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2))
DSR = PSR at SR* = sqrt(V[SR_trials]) * ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))
MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (z_alpha / (SR - SR*))^2
```

## When to use it

Is a strategy's edge real or luck? The primary result is a block-bootstrap interval for the Sharpe-ratio difference against a baseline such as buy-and-hold. Beside it: the **Probabilistic Sharpe Ratio** (probability that the true Sharpe ratio exceeds a benchmark, allowing for sample length, skewness and kurtosis), the **Deflated Sharpe Ratio** (the PSR against the Sharpe ratio the best of N unskilled trials would reach by luck) and the **minimum track record length**. They add warnings but never change the verdict.

## What it assumes

- Sharpe ratios are **per period**, not annualised: annualising multiplies by sqrt(periods per year) and must be undone first, or the probabilities saturate at 1. `min_effect` is in the same units.
- DSR needs the number of strategies that were tried, from a trial registry or an explicit `n_trials` with `sr_variance`. There is no default: `isbetter` raises `MissingTrialsError` rather than assume a count and hide the selection bias.

## How it fails

- **Treating the trials as independent**: DSR does, although neighbouring variants (two moving averages with nearby windows) are near-duplicates, which overstates the number of independent trials.
- **The closed-form expected maximum Sharpe is an approximation**: about 8% low at N = 2 and 0.5 to 3% high for N >= 3.
- **The block-bootstrap interval stays 2 to 4 points short of nominal coverage** at T = 500, and the iid bootstrap is worse: [failure gallery #2](../gallery.md).
- **A survivor-only registry**: logging only the winners makes DSR and PBO say nothing about the search.

## Use it

```yaml
adapter: strategy_returns
metric: {name: return, kind: returns}
dependence: {type: timeseries}
finance: {registry: out/trials.jsonl}          # or {n_trials: 272, sr_variance: 0.0016}
```

```python
from isbetter.methods.finance import probabilistic_sharpe, deflated_sharpe, min_track_record
deflated_sharpe(returns, n_trials=272, sr_variance=0.0016)
```

## Checked against

- The Bailey and Lopez de Prado papers' worked examples could not be fetched where this was built, so the oracles are independent of the implementation: quadrature, simulation and the tabulated expected maximum of N normals; a Monte Carlo check of the Sharpe-variance formula; PSR coverage; and the exact identity PSR(MinTRL) = 1 - alpha. Paper fixtures should be added when they can be transcribed.
- On iid returns the block bootstrap with block length 1 reproduces the analytic interval.

## References

- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk* 15.
- Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management* 40.
