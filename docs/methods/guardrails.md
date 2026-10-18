# Guardrails and the ship decision

**Applies to:** A/B experiments with a `metrics:` list · **Code:** `isbetter.decision.guardrails` · **Runs automatically** with `adapter: ab_table` or a `metrics` list.

## When to use it

An experiment with metrics you want to improve (success) and metrics that must not get worse (guardrails: latency, unsubscribe rate, spend). The decision follows the structure of the decision rules Spotify's experimentation team has described: success metrics are tested for superiority, guardrails for non-inferiority, and a candidate ships only if at least one success metric wins and no guardrail fails.

## What it assumes

- Success metrics: superiority tests with multiplicity correction across the success hypotheses only.
- Guardrails: a one-sided non-inferiority test at the unadjusted alpha, implemented as the two-sided 1 - 2 alpha interval. A guardrail passes only if the lower limit is above minus its margin.
- A sample-ratio mismatch overrides everything.

## How it fails

- **A margin that is too loose or too tight**: the guardrail's `min_effect` is the non-inferiority margin, so it encodes how much harm you accept. A guardrail with a wide interval fails as *inconclusive* rather than passing: absence of evidence of harm is not a pass.
- **`trivial_win`**: significant but possibly below the minimum effect. By default it still counts toward shipping, and the reason says when it was counted; set `ship_requires_practical: true` to require a practical win.
- Sequential monitoring is not yet wired into experiments.

## Use it

```yaml
metrics:
  - {name: conversion, kind: binary, min_effect: 0.002}
  - {name: spend, role: guardrail, min_effect: 0.10}     # may not drop by more than 0.10
decision: {alpha: 0.05, correction: holm, ship_requires_practical: false}
```

Each candidate gets a SHIP or DO NOT SHIP with the reason, and a per-metric table of results. A success metric is `win`, `trivial_win`, `no_win` or `worse`; a guardrail is `pass`, `fail_harm` or `fail_inconclusive`.

## Checked against

- Table-driven tests cover the ship / no-ship combinations of success and guardrail outcomes and the SRM override.

## References

- The case study [A/B test: email campaign](../case-studies/02_ab_hillstrom_criteo.md) works through a decision with two success metrics and a spend guardrail.
