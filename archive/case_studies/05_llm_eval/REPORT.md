# Case study 5: is the candidate model really better? (small LLM eval)

> **Data:** SYNTHETIC (200 questions in 50 passages, two simulated models).
> Real per-item results could not be downloaded here; rerun with `--data` for a real analysis. Nothing below is a claim about real models.

## 1. Question

Two models answered the same 200 questions. Is the candidate better, and
by enough to matter (a margin of 3 accuracy points)? How many more questions would settle it?

## 2. Data and dependence structure

One 0/1 score per model and question, in per-item JSON Lines (`out/items_*.jsonl`). Two
dependences: both models answered **the same questions**, so scores are *paired* by question
(shared difficulty cancels in the difference), and questions come in **passages** of
4 that share their source, so the per-question differences within a passage are
*correlated* and the effective sample is closer to the number of passages than of questions.
Hence `dependence: {type: clustered, paired: true}`.

## 3. Naive analysis, and why it misleads

Treat each model's questions as an independent sample and compare two accuracies with a normal
interval. Three things go wrong (section 4 shows them): the normal interval is too narrow for a
single model when the sample is small and accuracy is far from 50%; ignoring that both models
answered the same questions wastes the pairing and loses power; and ignoring passages makes the
paired test too confident.

World A is seed 20261050, the first of 41 consecutive seeds in which the paired t-test (which ignores passages) declares the candidate better, so the problem is visible; how often that happens without conditioning is in section 5.

## 4. Correct analysis

### World A: the models are equally good

Baseline 64.0%, candidate 70.5% on 200 questions in 50 passages.
Truth (known because the data are simulated): the models are equally good, so any observed difference is chance.

**One model's accuracy** (the baseline, all 200 questions, and its first 40
questions as a pilot):

| interval | accuracy | 95% interval |
|---|---:|---|
| normal (CLT) | 64.0% | [57.3%, 70.7%] |
| Wilson | 64.0% | [57.1%, 70.3%] |
| Jeffreys | 64.0% | [57.2%, 70.4%] |
| Clopper-Pearson | 64.0% | [56.9%, 70.6%] |

Pilot (40 questions):

| interval | accuracy | 95% interval |
|---|---:|---|
| normal (CLT) | 72.5% | [58.5%, 86.5%] |
| Wilson | 72.5% | [57.2%, 83.9%] |
| Jeffreys | 72.5% | [57.5%, 84.4%] |
| Clopper-Pearson | 72.5% | [56.1%, 85.4%] |

**Candidate minus baseline**, by the naive and the matching methods:

| method | accuracy difference (points) | 95% interval | p-value |
|---|---:|---|---:|
| unpaired normal (Welch) | +6.5 | [-2.7, +15.7] | 0.167 |
| paired normal (paired t) | +6.5 | [+0.6, +12.4] | 0.032 |
| paired + clustered (CR1) | +6.5 | [-0.4, +13.4] | 0.063 |
| paired score interval + McNemar | +6.5 | [+0.6, +12.7] | 0.049 |
| paired Beta posterior (credible) | +6.4 | [+0.6, +12.4] | n/a |

The verdict `isbetter run` gives from `config.yaml` (`dependence: clustered, paired: true`),
margin 3 points:

#### candidate vs baseline — correct

**Verdict: inconclusive** (`inconclusive`)

CI straddles 0 and extends beyond the margin.

**How much more data?** About 222 observations in total (currently 200). assumes CI half-width shrinks like 1/sqrt(n) at the current estimate.


| Method | Estimate | 95.0% CI | p-value |
|---|---:|---|---:|
| paired_cluster_robust | 0.065 | [-0.003555, 0.1336] | 0.06261 |
| paired_cluster_bootstrap | 0.065 | [0, 0.135] | n/a |

Oriented effect: 0.065 (CI -0.003555 to 0.1336), minimum effect of interest δ = 0.03.

**Why this method:** Paired difference over 200 items with a cluster-robust (CR1) standard error (50 clusters, t(49) inference): both models answer the same items, so the per-item differences are analysed, and items in a cluster are correlated, so the error bar counts clusters rather than items.

- *paired_cluster_bootstrap:* Percentile bootstrap of the paired difference resampling whole clusters (50 clusters); a cross-check of the cluster-robust interval.



### World B: the candidate is 6 points better

Baseline 72.5%, candidate 83.5% on 200 questions in 50 passages.


**One model's accuracy** (the baseline, all 200 questions, and its first 40
questions as a pilot):

| interval | accuracy | 95% interval |
|---|---:|---|
| normal (CLT) | 72.5% | [66.3%, 78.7%] |
| Wilson | 72.5% | [65.9%, 78.2%] |
| Jeffreys | 72.5% | [66.0%, 78.3%] |
| Clopper-Pearson | 72.5% | [65.8%, 78.6%] |

Pilot (40 questions):

| interval | accuracy | 95% interval |
|---|---:|---|
| normal (CLT) | 70.0% | [55.6%, 84.4%] |
| Wilson | 70.0% | [54.6%, 81.9%] |
| Jeffreys | 70.0% | [54.8%, 82.4%] |
| Clopper-Pearson | 70.0% | [53.5%, 83.4%] |

**Candidate minus baseline**, by the naive and the matching methods:

| method | accuracy difference (points) | 95% interval | p-value |
|---|---:|---|---:|
| unpaired normal (Welch) | +11.0 | [+2.9, +19.1] | 0.008 |
| paired normal (paired t) | +11.0 | [+4.8, +17.2] | <0.001 |
| paired + clustered (CR1) | +11.0 | [+4.2, +17.8] | 0.002 |
| paired score interval + McNemar | +11.0 | [+4.9, +17.5] | 0.001 |
| paired Beta posterior (credible) | +10.9 | [+4.8, +17.1] | n/a |

The verdict `isbetter run` gives from `config.yaml` (`dependence: clustered, paired: true`),
margin 3 points:

#### candidate vs baseline — correct

**Verdict: candidate is better (practically)** (`A_better_practical`)

CI lies entirely above +0.03.


| Method | Estimate | 95.0% CI | p-value |
|---|---:|---|---:|
| paired_cluster_robust | 0.11 | [0.04243, 0.1776] | 0.001964 |
| paired_cluster_bootstrap | 0.11 | [0.045, 0.175] | n/a |

Oriented effect: 0.11 (CI 0.04243 to 0.1776), minimum effect of interest δ = 0.03.

**Why this method:** Paired difference over 200 items with a cluster-robust (CR1) standard error (50 clusters, t(49) inference): both models answer the same items, so the per-item differences are analysed, and items in a cluster are correlated, so the error bar counts clusters rather than items.

- *paired_cluster_bootstrap:* Percentile bootstrap of the paired difference resampling whole clusters (50 clusters); a cross-check of the cluster-robust interval.



## 5. How often is each method fooled?

Unconditional behaviour over 2000 fresh simulated evals of each kind (share of evals
in which the method says the models differ at the 5% level):

| method | World A (no difference): false positives | World B (6 points better): detections |
|---|---:|---:|
| unpaired normal (Welch) | 0.9% | 20.3% |
| paired normal (paired t) | 6.6% | 44.2% |
| paired + clustered (CR1) | 5.0% | 38.2% |
| paired score interval + McNemar | 4.8% | 39.0% |

Reading the false-positive column against the 5% it should be (Monte Carlo band
4.0% to 6.0%): the unpaired test is
conservative (0.9%), the paired t is too liberal (6.6%), the paired score
interval with McNemar is close to 5% (4.8%), and the cluster-robust test is
close to 5% (5.0%). Pairing
matters for power: the unpaired test finds the real 6-point gap in 20% of
evals, against 38% for the cluster-robust paired test, which removes the shared
difficulty and still counts passages rather than questions.

## 6. What would change the conclusion

* **Real models and data.** The synthetic eval has a particular size of passage effect (the
  baseline's accuracy varies from 20% to 84% across passages) and of model x passage interaction;
  real evals differ, and so does how much the clustered interval costs.
* **Few passages.** The cluster-robust interval relies on t with (passages - 1) degrees of
  freedom and is mildly liberal below roughly 30 clusters.
* **Graded scores or several runs per question** (sampling temperature) change the structure:
  repeated answers to one question are a further level of clustering.
* **Sequential reading.** Looking at the eval after every batch of questions needs a confidence
  sequence (`sequential:` in the config), which is not available for clustered data yet.
* Peak memory of this run: 0.11 GB.
