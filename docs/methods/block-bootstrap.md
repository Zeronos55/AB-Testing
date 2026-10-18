# Block bootstrap and block-length choice

**Dependence:** `timeseries` · **Code:** `isbetter.methods.resampling.block_bootstrap`, `isbetter.methods.forecasting.suggest_block_length` · **Routed as cross-check** to Diebold-Mariano, and used as the primary interval for Sharpe-ratio differences.

## When to use it

A resampling interval for serially dependent data. Single observations are resampled in blocks (stationary or circular) so the dependence inside each block survives. The block length matters: too short and the dependence is lost, too long and there are too few blocks. The report prints the block length used, where it came from (set in the config, Politis-White, or the T^(1/3) rule of thumb), and the automatic suggestions for both kinds.

## What it assumes

- The series is stationary and weakly dependent.
- Enough observations for several blocks.

## How it fails

- **Short series**: the block bootstrap is better than the iid one, not exact. For a Sharpe ratio at T = 500 it covers about 91 to 93% at nominal 95%.
- **Long memory or trends** break the weak-dependence assumption.
- **The iid bootstrap on dependent data** gives intervals that are too short: see [failure gallery #2](../gallery.md).

## Use it

```yaml
dependence: {type: timeseries, horizon: 1, block_len: 10}   # block_len optional
```

```python
from isbetter.methods.resampling import block_bootstrap
block_bootstrap(a, b, rng, block_len=10, kind="stationary")
```

## Checked against

- The Politis-Romano closed form for the block bootstrap, and `arch`'s bootstrap classes.
- On iid returns with block length 1 it reproduces the analytic normal-approximation interval and an independent plain-NumPy bootstrap.

## References

- Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *JASA* 89.
- Politis, D. N. and White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews* 23.
- Patton, A., Politis, D. N. and White, H. (2009). Correction to “Automatic block-length selection for the dependent bootstrap”. *Econometric Reviews* 28.
