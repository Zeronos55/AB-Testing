# isbetter

**Is A really better than B?** You tell `isbetter` how your data are dependent. It picks the test that is valid for that dependence, returns a three-way verdict (*better*, *equivalent* or *inconclusive*), and shows that the test is calibrated.

Most wrong answers to "is A better than B?" are not arithmetic mistakes. They come from using a test whose independence assumption the data break:

| You have | The textbook test assumes | What happens |
|---|---|---|
| 10 repeats of 10-fold CV scores | independent scores | rejects a true null 57% of the time at a nominal 5% |
| Many sessions per user | independent rows | 31% false positives with 10 sessions per user |
| A multi-step forecast | uncorrelated loss differences | 17% false positives at horizon 3 |
| A dashboard checked every day | one look at a fixed sample size | nearly half (46 to 49% across our runs) false positives when read after every observation |
| 30 to 200 eval items | the central limit theorem | the 95% interval covers 78% at accuracy 0.95, N = 30 |

These numbers come from simulations where the truth is known; see the [calibration table](calibration.md) and the [failure gallery](../validation/failure_gallery/).

## The idea

1. **Declare the dependence** of your data: `iid`, `paired`, `clustered`, `cv_overlap` or `timeseries`. See [declaring dependence](concepts/dependence.md).
2. **The router picks the valid method** and prints the plan before running. If no valid method exists for your combination it refuses with an explanation. It never falls back to a naive test.
3. **A three-way verdict** compares the confidence interval with the smallest effect that matters to you ([the verdict](concepts/verdict.md)), and says how much more data would settle an inconclusive result.
4. **Every method is checked** against a trusted reference implementation (SciPy, statsmodels, mlxtend, arch, R's `forecast`) and against simulations of its own error rates.

## Where to go next

- [Quickstart](quickstart.md): install, run an example, read the report.
- [Method cards](methods/index.md): one page per method with *when to use it*, *what it assumes*, *how it fails* and *references*.
- [Case studies](../case_studies/): five worked comparisons (fraud classifiers, an email A/B test, forecasting, trading, an LLM eval).
- [Blueprint](BLUEPRINT.md): the full design, formulas and the build plan.

> **Status.** The case studies run on synthetic stand-in data with the structure of the real datasets; each report says so.
