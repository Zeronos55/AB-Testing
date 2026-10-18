# Calibration

A method is *calibrated* if it makes errors at the rate it claims: a 5% test rejects a true null
about 5% of the time, a 95% interval contains the truth about 95% of the time. Every method
here is checked on simulated worlds where the truth is known, and the recommended method is shown
next to the naive method it replaces.

The table is generated, not written. `validation/calibration_table.py` reruns each scenario
with the same harness and seeds as the slow tests (`uv run pytest -m slow`) and writes the full
table, with Monte Carlo bands, to
[`validation/calibration_table.md`](https://github.com/Zeronos55/AB-Testing/blob/main/validation/calibration_table.md).
A compact copy sits in the [README](https://github.com/Zeronos55/AB-Testing#is-it-calibrated),
and the nightly build regenerates it.

```bash
uv run python validation/calibration_table.py --readme     # a few minutes at 10,000 replications
uv run pytest -m slow                                      # the calibration tests themselves
```

## How to read it

- **Naive rows are foils.** They show what goes wrong without the right dependence structure; they
  are not recommendations.
- **The band** is two Monte Carlo standard errors around the nominal rate. A calibrated method
  still falls outside it about one time in twenty, so a single slightly-outside row is not by
  itself a failure. The nightly runs are the check.
- **Conservative** means below nominal: valid, but giving up power. Confidence sequences are
  conservative by construction.
- **Exact** rows (small-N interval coverage) have no simulation noise, because the count of
  correct answers is binomial.
- **Known limits are not hidden.** Where a recommended method is slightly off (cluster-robust
  standard errors with few clusters, Diebold-Mariano at long horizons with few origins, the
  block-bootstrap Sharpe interval at T = 500), the table reports it as it came out and the
  [method card](methods/index.md) says why.

See also the [failure gallery](gallery.md), which shows the same failures as figures.
