# Case study 5: is the candidate model really better? (small LLM eval)

**Question:** two models answered the same 200 questions, which come in 50 passages of 4. Is the
candidate better, by at least 3 accuracy points, and how many more questions would settle it?

**Run it**

```bash
uv run python case_studies/05_llm_eval/run.py                    # synthetic stand-in data, ~10 s
uv run python case_studies/05_llm_eval/run.py --data items.jsonl --baseline A --candidate B
uv run isbetter run case_studies/05_llm_eval/config.yaml         # the standard analysis, alone
```

Output: [`REPORT.md`](REPORT.md) (full write-up) and `out/items_*.jsonl` (per-item results in the
format the `llm_eval` adapter reads, which the `isbetter run` command above analyses). Peak
memory is about 0.1 GB.

**Data caveat.** Real per-item results could not be downloaded from the build environment, so
the committed report uses a **synthetic** eval: passages are easy or hard for both models, their
mistakes are partly shared, and the candidate is better on some passages and worse on others.
World A has no true difference; World B has a true 6-point advantage. They demonstrate the
methods, not the behaviour of real models. `--data` takes per-item JSON Lines (`model`,
`item_id`, `score` of 0 or 1, `cluster_id`); that path is tested only on a tiny file.

**What the report compares**

| tool | what goes wrong without it |
|---|---|
| Wilson / Jeffreys / Clopper-Pearson for one model's accuracy | the normal interval is too narrow on a 40-question pilot |
| paired difference (`paired: true`) | comparing two accuracies as if from independent samples loses power |
| cluster-robust SE over passages (`clustered`) | the paired t-test treats 200 correlated questions as 200 independent ones |
| paired score interval + McNemar (`paired_binary_score`) | exact small-sample treatment of right/wrong pairs |
| Beta/Dirichlet posterior | Bayesian cross-check |

The report also repeats the experiment over 2,000 fresh evals of each kind to show how often
each method is fooled (false positives with no true difference) and how often it finds the real
gap (power).

**Not covered:** several answers per question (sampling runs are a further level of clustering),
graded or multi-valued scores, sequential reading of a clustered eval.
