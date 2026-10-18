# Declaring dependence

The one decision `isbetter` asks of you is **how your observations depend on each other**. That
choice, together with the kind of metric, picks the test. The same scores analysed with the wrong
dependence give a confident wrong answer, which is what the [failure gallery](../gallery.md) shows.

| `dependence` | Use it when | Primary method | Cross-checks |
|---|---|---|---|
| `{type: iid}` | Independent units, one per arm (a randomised A/B test by user) | [Welch t](../methods/welch-t.md); binary: [two-proportion z](../methods/two-proportions.md); ratio: [delta method](../methods/delta-method.md); with a covariate: [CUPED](../methods/cuped.md) | binary: Fisher exact; ratio: bootstrap |
| `{type: paired}` | Both variants measured on the same units | [paired t](../methods/paired-t.md); binary right/wrong: [paired score interval](../methods/small-n.md) | [paired bootstrap, sign-flip permutation](../methods/bootstrap-permutation.md) |
| `{type: clustered}` | Rows grouped by user, passage or similar | [cluster-robust CR1](../methods/cluster-robust.md) | cluster bootstrap |
| `{type: clustered, paired: true}` | The same items scored for both models, items grouped in clusters | paired cluster-robust | paired cluster bootstrap |
| `{type: cv_overlap, n_train, n_test, repeats, folds}` | Scores from resampled train/test splits | [Nadeau-Bengio](../methods/nadeau-bengio.md) | |
| `{type: timeseries, horizon: h}` | Per-origin losses of forecasts | [Diebold-Mariano with HLN](../methods/diebold-mariano.md) | [stationary block bootstrap](../methods/block-bootstrap.md) |
| `{type: timeseries}` with `metric.kind: returns` | Strategy returns | [block-bootstrap Sharpe difference](../methods/sharpe-psr-dsr.md) | PSR, DSR, MinTRL, [PBO](../methods/pbo.md) |
| any of `iid`/`paired` plus `sequential:` | Data read while it is still arriving | [confidence sequence](../methods/confidence-sequences.md) | the fixed-horizon test, as the foil |

## It refuses rather than guessing

When a combination is not in the table the router raises an error that says what is missing, for
example a ratio metric under `cv_overlap`, or `timeseries` with a binary metric. It never
substitutes a test that would give invalid inference. The plan is printed before anything runs, so
you can see which method was chosen and check it matches what you meant.

## The data contract

Every adapter produces one long table: `unit_id`, `variant`, `metric`, `value`, plus `pair_id`
(paired), `cluster_id` (clustered), `time` (time series, strictly increasing within a variant) and
`fold`/`repeat` (CV). Validation fails loudly on a missing partner in a declared pairing,
duplicate rows, a non-monotone `time`, fewer than two clusters per variant, and NaN values. Large
A/B tables can be aggregated to sufficient statistics (`n`, `sum`, `sum_sq`, and cross-products
for CUPED) with `table: {mode: stats}`, so a 14-million-row table never has to fit in memory.
