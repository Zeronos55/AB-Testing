# Case study 2: an email A/B test and a Criteo-scale ad test

**Question:** should either email variant replace sending no email (success: more visits or
conversions; guardrail: revenue per customer must not drop), and can the same machinery analyse a
14-million-row ad test without loading it?

**Run it**

```bash
uv run python case_studies/02_ab_hillstrom_criteo/run.py                        # synthetic stand-ins
uv run python case_studies/02_ab_hillstrom_criteo/run.py \
    --hillstrom hillstrom.csv --criteo criteo-uplift-v2.1.csv                   # real files
uv run isbetter run case_studies/02_ab_hillstrom_criteo/config.yaml             # after run.py wrote out/
uv run isbetter plan-size --kind proportions --baseline 0.0084 --mde 0.002      # how big a test?
```

Output: [`REPORT.md`](REPORT.md) (full write-up). `config.yaml` is the email experiment and
`config_criteo.yaml` the aggregated large one; both run with `isbetter run`.

**Data caveat.** Neither real dataset could be downloaded where this was built (outbound access
to their hosts was blocked), so the committed report uses **synthetic stand-ins** with the same
shape (64,000 customers in three arms; 14,000,000 rows with `treatment, visit, conversion`). Their
effect sizes are assumptions: nothing in the report is a claim about the real campaigns. The real
Hillstrom file is expected to have columns `segment`, `history`, `visit`, `conversion`, `spend`,
and the Criteo file `treatment`, `visit`, `conversion`; neither the URLs nor those layouts were
verified here. The Criteo "rows are impressions, not users" question is likewise **unverified**;
see the last section of the report for what to check before trusting real numbers.

**What the report shows on the synthetic data**

| step | result |
|---|---|
| SRM on 21,306 / 21,307 / 21,387 | passes (p = 0.90) |
| Holm over 4 success hypotheses | all 4 are wins; correction widens CIs but changes no verdict |
| Spend guardrail (one-sided, margin $0.10) | passes for both emails |
| CUPED on `history` | CIs only 0.6-0.9% narrower: correlation with the outcome is low (about 0.12), and the gain is capped by rho^2 |
| 14M-row Criteo-style table via DuckDB | about 1.6 s, 276 MB peak resident |
| Planted logging bug (2% of treated rows lost) | SRM fails (p about 1e-33); every decision blocked |

The CUPED gain is small because the synthetic `history` is only weakly related to the outcomes.
The report says so rather than tuning the data to flatter the method.
