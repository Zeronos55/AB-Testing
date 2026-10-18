# Sample size, power and n_needed

**Applies to:** planning and inconclusive results · **Code:** `isbetter.methods.power` · **CLI:** `isbetter plan-size`.

## When to use it

Before an experiment: how many units do I need to detect a lift? After an inconclusive one: how much more data would settle it? The planner covers means and proportions, reports power and the minimum detectable effect for a fixed size, and accounts for a CUPED covariate through its correlation.

## What it assumes

- Your inputs are right: the baseline rate or standard deviation, and the effect worth detecting.
- For `n_needed`, that the interval keeps its current estimate and shrinks like 1/sqrt(n).

## How it fails

- **Planning on the observed effect** after seeing it is circular. Plan on the smallest effect that matters.
- **`n_needed` is an extrapolation** and the report says so; a different true effect changes it a lot.
- **For cross-validation, repeats do not help once the n_test / n_train term dominates**: the report says to get more data instead.

## Use it

```console
$ uv run isbetter plan-size --kind proportions --baseline 0.10 --mde 0.01
$ uv run isbetter plan-size --kind means --sd 10 --mde 1 --rho 0.7      # with a CUPED covariate
$ uv run isbetter plan-size --kind proportions --baseline 0.10 --mde 0.01 --n-control 20000
```

The last form reports the power and the minimum detectable lift of a fixed size.

## Checked against

- `statsmodels.stats.power`, within 1%.
- A unit test with a known half-width ratio for `n_needed`.

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.
