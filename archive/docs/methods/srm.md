# Sample-ratio mismatch (SRM) check

**Applies to:** A/B experiments (`iid` and `clustered`) · **Code:** `isbetter.methods.srm.srm_check` · **Runs automatically** on every A/B plan.

## When to use it

Always, before reading any effect. A chi-square goodness-of-fit test compares the number of randomisation units in each arm with the planned split. A tiny p-value means assignment, logging or filtering is broken, so the arms are no longer comparable and the effect estimate cannot be trusted whatever it says. A failed check overrides the ship decision.

## What it assumes

- The counts are of randomisation units (users, or clusters when randomised by cluster), not rows.
- The planned split is stated: equal by default, or `design: {expected_split: {...}}`.

## How it fails

- **It detects that something is wrong, not what.** Bots, redirects, crashes in one arm and a bad filter all look the same.
- **Small expected counts**: below 5 in a cell the chi-square approximation is unreliable, and the report warns.
- **The threshold is strict on purpose** (p < 0.001, `design.srm_alpha`) because the check runs on every experiment and a false alarm is costly. A mild mismatch can slip under it.

## Use it

```yaml
design: {expected_split: {control: 0.5, treatment: 0.5}, srm_alpha: 0.001}
```

```python
from isbetter.methods.srm import srm_check
srm_check({"No E-Mail": 21_306, "Mens E-Mail": 21_307, "Womens E-Mail": 21_387})
```

## Checked against

- `scipy.stats.chisquare`.
- The Hillstrom arm counts (21,307 / 21,387 / 21,306) pass; a planted 50.5 / 49.5 split at n = 1,000,000 fails.
- A calibration test of the false-alarm rate: see the [calibration table](../calibration.md).

## References

- Fabijan, A. et al. (2019). Diagnosing sample ratio mismatch in online controlled experiments. *KDD*.
