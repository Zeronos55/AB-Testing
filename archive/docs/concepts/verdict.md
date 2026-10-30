# The verdict

A p-value answers "is the difference exactly zero?", which is rarely the question. `isbetter`
asks instead how the confidence interval for the difference sits against the **smallest effect
that matters to you**, the margin δ (`decision.min_effect`).

For a difference Δ = candidate − baseline, oriented so that positive means the candidate is
better, with interval [L, U]:

| Condition | Verdict |
|---|---|
| L > +δ | `A_better_practical` |
| 0 < L ≤ δ | `A_better_maybe_trivial`: better, but the gain may be below the margin |
| −δ < L and U < +δ | `equivalent` (a TOST-style check with the 1 − 2α interval) |
| U < −δ | `B_better_practical` |
| −δ ≤ U < 0 | `B_better_maybe_trivial` |
| anything else | `inconclusive` |

Swapping the two variants flips the sign of the estimate and mirrors the label; this is a
property test, not a convention.

## How much more data?

An inconclusive result reports `n_needed`. The half-width h of an interval shrinks like 1/√n.
If h* is the half-width at which the interval would clear the nearest boundary (0, +δ or −δ),
then about n·(h/h*)² observations are needed in total. The report states that this is an
extrapolation from the current estimate.

For cross-validation the variance has the factor 1/(rk) + n_test/n_train. When the second term
dominates, adding CV repeats cannot help, and the report says *more CV repeats won't help: you
need more data* instead of a number. With 10-fold CV the second term is about 0.111 while the
first is only 0.01 at 10 repeats.

## Several candidates or metrics

`decision.correction` takes `none`, `holm`, `bonferroni` or `bh`; the report shows adjusted
p-values and the per-comparison α used for the intervals. See
[multiplicity](../methods/multiplicity.md) and, for A/B experiments with success metrics and
guardrails, [guardrails and the ship decision](../methods/guardrails.md).
