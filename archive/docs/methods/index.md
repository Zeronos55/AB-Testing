# Method cards

One card per method. Each says **when to use it**, **what it assumes**, **how it fails**, how to
**use it** from a config and from Python, what it was **checked against**, and the **references**.
The cards describe what the code does today, including its known limits; where an oracle could not
be the paper's own numbers, the card says so.

You do not pick a method by name. You declare the [dependence](../concepts/dependence.md) of your
data and the router picks, printing the plan before it runs. The cards are for knowing what that
choice means.

## Independent or paired means

| Card | Declared as | Use it for |
|---|---|---|
| [Welch t-test](welch-t.md) | `iid`, continuous | Two independent arms, a mean |
| [Paired t-test](paired-t.md) | `paired`, continuous | The same units under both variants |
| [Bootstrap and permutation](bootstrap-permutation.md) | cross-checks for `paired` | Not trusting normality |

## A/B experiments

| Card | Declared as | Use it for |
|---|---|---|
| [Two proportions](two-proportions.md) | `iid`, binary | Conversion and click rates |
| [CUPED](cuped.md) | `variance_reduction: cuped` | A tighter interval from a pre-experiment covariate |
| [Delta method](delta-method.md) | `kind: ratio` | Revenue per visit and other ratios |
| [Sample-ratio check](srm.md) | automatic | Is randomisation intact? |
| [Multiplicity](multiplicity.md) | `decision.correction` | Several candidates or metrics |
| [Guardrails and ship decision](guardrails.md) | `role: guardrail` | Do not ship if something breaks |
| [Sample size and power](power.md) | `isbetter plan-size` | How many units, how much more data |

## Clusters, cross-validation and small N

| Card | Declared as | Use it for |
|---|---|---|
| [Cluster-robust (CR1)](cluster-robust.md) | `clustered` | Sessions per user, questions per passage |
| [Nadeau-Bengio](nadeau-bengio.md) | `cv_overlap` | Comparing models from CV scores |
| [McNemar](mcnemar.md) | paired right/wrong | Two classifiers on the same items |
| [Small-N intervals](small-n.md) | `paired`, binary | 30 to 200 eval items |

## Time series and finance

| Card | Declared as | Use it for |
|---|---|---|
| [Diebold-Mariano](diebold-mariano.md) | `timeseries` | Two forecasters |
| [Block bootstrap](block-bootstrap.md) | cross-check for `timeseries` | Dependent data, block length |
| [SPA, StepM, MCS](spa-stepm-mcs.md) | many models | Best of many against a benchmark |
| [Sharpe ratio: PSR, DSR, MinTRL](sharpe-psr-dsr.md) | `kind: returns` | Edge or luck? |
| [PBO](pbo.md) | `kind: returns` | Was the winner a product of the search? |

## Sequential

| Card | Declared as | Use it for |
|---|---|---|
| [Confidence sequences](confidence-sequences.md) | `sequential:` | Reading a result while data arrive |
