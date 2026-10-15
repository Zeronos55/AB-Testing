"""Case study 5: is the candidate model really better than the baseline on a small LLM eval?

    uv run python case_studies/05_llm_eval/run.py                       # synthetic stand-in data
    uv run python case_studies/05_llm_eval/run.py --data items.jsonl --baseline A --candidate B

Real per-item results (for example from a public eval harness) could not be downloaded in the
environment this was built in, so by default a clearly labelled SYNTHETIC eval is used: 50
passages with 4 questions each (200 questions), two models that answered every question. A
passage is easy or hard for both models (shared difficulty), the models' mistakes are partly
shared, and the candidate is better on some passages and worse on others (a model x passage
interaction), as when two models differ in the topics they know. Results on synthetic data say
nothing about real models.

``--data`` takes per-item JSON Lines (``model``, ``item_id``, ``score``, ``cluster_id``) with a
0/1 score; that path is tested only on a tiny file.
"""

from __future__ import annotations

import argparse
import resource
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isbetter.adapters.llm_eval import (
    ItemScore,
    read_item_scores,
    write_item_scores,
)
from isbetter.core.config import load_config
from isbetter.core.plan import build_plan
from isbetter.core.result import Result
from isbetter.core.runner import run
from isbetter.methods.parametric import paired_t, welch_t
from isbetter.methods.proportions import jeffreys_interval, wilson_interval
from isbetter.methods.small_n import (
    clopper_pearson_interval,
    clt_interval,
    paired_beta_posterior,
    paired_binary_score,
)
from isbetter.report.markdown import render_markdown

HERE = Path(__file__).parent
BASE, CAND = "baseline", "candidate"
N_PASSAGES, PER_PASSAGE = 50, 4
PILOT_PASSAGES = 10  # the first 40 questions: a pilot eval
TRUE_GAP = 0.06  # candidate's accuracy advantage in the "real difference" world
N_REPEATS = 2000


# ------------------------------------------------------------------------------- data


def synthetic_eval(
    seed: int, gap: float, passages: int = N_PASSAGES, per: int = PER_PASSAGE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SYNTHETIC right/wrong answers of two models on the same questions, plus passage ids.

    The baseline's accuracy on a passage is logistic(0.9 + N(0, 0.7)) clipped to [0.2, 0.84].
    The candidate's is that plus ``gap`` plus a symmetric passage-specific shift of sd 0.15
    (clipped to +-0.15), so with ``gap = 0`` the expected accuracy difference is exactly zero
    while still differing from passage to passage. 60% of the time the candidate's answer uses
    the same random draw as the baseline's (shared mistakes).
    """
    rng = np.random.default_rng(seed)
    pa = np.clip(1 / (1 + np.exp(-(0.9 + rng.normal(0, 0.7, passages)))), 0.2, 0.84)
    shift = np.clip(rng.normal(0, 0.15, passages), -0.15, 0.15)
    pb = pa + gap + shift
    n = passages * per
    u = rng.random(n)
    v = np.where(rng.random(n) < 0.6, u, rng.random(n))
    a = (u < pa.repeat(per)).astype(float)
    b = (v < pb.repeat(per)).astype(float)
    return a, b, np.repeat([f"passage_{i:03d}" for i in range(passages)], per)


def records(a: np.ndarray, b: np.ndarray, passage: np.ndarray) -> list[ItemScore]:
    ids = [f"q{i:04d}" for i in range(len(a))]
    return [
        ItemScore(BASE, q, float(x), str(c)) for q, x, c in zip(ids, a, passage, strict=True)
    ] + [ItemScore(CAND, q, float(x), str(c)) for q, x, c in zip(ids, b, passage, strict=True)]


def arrays(table, baseline: str, candidate: str, metric: str):  # type: ignore[no-untyped-def]
    """Per-question scores of two models, aligned by item, with passage ids."""
    ra, rb = table.select(baseline, metric), table.select(candidate, metric)
    ka, kb = np.argsort(ra["pair_id"].astype(str)), np.argsort(rb["pair_id"].astype(str))
    if not np.array_equal(ra["pair_id"].astype(str)[ka], rb["pair_id"].astype(str)[kb]):
        raise ValueError("the two models were not scored on the same questions")
    return ra["value"][ka], rb["value"][kb], ra["cluster_id"][ka].astype(str)


# --------------------------------------------------------------------------- analysis


@dataclass
class Row:
    name: str
    estimate: float
    low: float | None
    high: float | None
    p: float | None


def comparison_rows(a: np.ndarray, b: np.ndarray, passage: np.ndarray, rng) -> list[Row]:  # type: ignore[no-untyped-def]
    """candidate - baseline accuracy by the naive and the matching methods."""
    from isbetter.methods.cluster import paired_cluster_robust

    w = welch_t(b, a)
    p = paired_t(b, a)
    c = paired_cluster_robust(b, a, passage)
    t = paired_binary_score(b, a)
    d = paired_beta_posterior(b, a, rng)
    rows = [
        Row("unpaired normal (Welch)", w.estimate, w.ci_low, w.ci_high, w.p_value),
        Row("paired normal (paired t)", p.estimate, p.ci_low, p.ci_high, p.p_value),
        Row("paired + clustered (CR1)", c.estimate, c.ci_low, c.ci_high, c.p_value),
        Row("paired score interval + McNemar", t.estimate, t.ci_low, t.ci_high, t.p_value),
        Row("paired Beta posterior (credible)", d.estimate, d.ci_low, d.ci_high, None),
    ]
    return rows


def accuracy_intervals(correct: np.ndarray) -> list[Row]:
    n, x = len(correct), int(correct.sum())
    out = []
    for name, (lo, hi) in (
        ("normal (CLT)", clt_interval(correct)),
        ("Wilson", wilson_interval(x, n)),
        ("Jeffreys", jeffreys_interval(x, n)),
        ("Clopper-Pearson", clopper_pearson_interval(x, n)),
    ):
        out.append(Row(name, x / n, lo, hi, None))
    return out


def fooled_rates(gap: float, seed: int, repeats: int) -> dict[str, float]:
    """How often each method rejects 'no difference' over fresh synthetic evals."""
    from isbetter.methods.cluster import paired_cluster_robust

    hits = {k: 0 for k in ("unpaired", "paired", "cr1", "tango")}
    for i in range(repeats):
        a, b, passage = synthetic_eval(seed + 10_000 + i, gap)
        hits["unpaired"] += welch_t(b, a).p_value < 0.05
        hits["paired"] += paired_t(b, a).p_value < 0.05
        hits["cr1"] += paired_cluster_robust(b, a, passage).p_value < 0.05
        hits["tango"] += paired_binary_score(b, a).p_value < 0.05
    return {k: v / repeats for k, v in hits.items()}


def word(rate: float, band: float) -> str:
    """Plain-English reading of a false-positive rate against the 5% it should be."""
    if rate > 0.05 + band:
        return f"too liberal ({rate:.1%})"
    if rate < 0.05 - band:
        return f"conservative ({rate:.1%})"
    return f"close to 5% ({rate:.1%})"


def first_fooled_seed(seed: int, limit: int = 200) -> tuple[int, int]:
    """First seed (from ``seed`` on) of a no-difference eval where the paired t flags a win."""
    for k in range(limit):
        a, b, _ = synthetic_eval(seed + k, 0.0)
        if paired_t(b, a).p_value < 0.05 and b.mean() > a.mean():
            return seed + k, k + 1
    raise RuntimeError("no fooled world found")


# ----------------------------------------------------------------------------- report


def pp(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:+.1f}"


def acc(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def fmt_p(p: float | None) -> str:
    return "n/a" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def comparison_table(rows: list[Row]) -> str:
    lines = [
        "| method | accuracy difference (points) | 95% interval | p-value |",
        "|---|---:|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {pp(r.estimate)} | [{pp(r.low)}, {pp(r.high)}] | {fmt_p(r.p)} |"
        )
    return "\n".join(lines)


def accuracy_table(rows: list[Row]) -> str:
    lines = ["| interval | accuracy | 95% interval |", "|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r.name} | {acc(r.estimate)} | [{acc(r.low)}, {acc(r.high)}] |")
    return "\n".join(lines)


def isbetter_section(result: Result) -> str:
    """The comparison part of the standard report, one heading level down."""
    text = render_markdown(result)
    body = text.split("### ", 1)[1] if "### " in text else text
    return "#### " + body.replace("\n### ", "\n#### ")


def world_section(title: str, a, b, passage, out: Path, tag: str, rng, truth: str = "") -> str:  # type: ignore[no-untyped-def]
    path = out / f"items_{tag}.jsonl"
    write_item_scores(path, records(a, b, passage))
    cfg = load_config(HERE / "config.yaml")
    plan = build_plan(cfg)
    table = read_item_scores(path, plan.dependence, metric=plan.metric)
    result = run(plan, table)
    n_pass = len(set(passage))
    pilot = slice(0, PILOT_PASSAGES * PER_PASSAGE)
    return f"""### {title}

Baseline {acc(a.mean())}, candidate {acc(b.mean())} on {len(a)} questions in {n_pass} passages.
{truth}

**One model's accuracy** (the baseline, all {len(a)} questions, and its first {len(a[pilot])}
questions as a pilot):

{accuracy_table(accuracy_intervals(a))}

Pilot ({len(a[pilot])} questions):

{accuracy_table(accuracy_intervals(a[pilot]))}

**Candidate minus baseline**, by the naive and the matching methods:

{comparison_table(comparison_rows(a, b, passage, rng))}

The verdict `isbetter run` gives from `config.yaml` (`dependence: clustered, paired: true`),
margin 3 points:

{isbetter_section(result)}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", help="per-item JSON Lines: model, item_id, score, cluster_id")
    ap.add_argument("--baseline", default=BASE)
    ap.add_argument("--candidate", default=CAND)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()

    cfg = load_config(HERE / "config.yaml")
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    if args.data:
        plan = build_plan(cfg)
        table = read_item_scores(args.data, plan.dependence, metric=plan.metric)
        a, b, passage = arrays(table, args.baseline, args.candidate, plan.metric)
        sections = world_section("Your data", a, b, passage, out, "data", rng)
        source = f"per-item results read from `{args.data}`"
        data_note = "Real data were used."
        repeated = "\nNot run for `--data`: it needs a model of how the data were generated.\n"
    else:
        seed_a, scanned = first_fooled_seed(cfg.seed)
        a0, b0, p0 = synthetic_eval(seed_a, 0.0)
        a1, b1, p1 = synthetic_eval(cfg.seed, TRUE_GAP)
        sections = "\n".join(
            [
                world_section(
                    "World A: the models are equally good",
                    a0,
                    b0,
                    p0,
                    out,
                    "null",
                    rng,
                    truth="Truth (known because the data are simulated): the models are equally "
                    "good, so any observed difference is chance.",
                ),  # fmt: skip
                world_section(
                    f"World B: the candidate is {100 * TRUE_GAP:.0f} points better",
                    a1,
                    b1,
                    p1,
                    out,
                    "gain",
                    rng,
                ),  # fmt: skip
            ]
        )
        r0 = fooled_rates(0.0, cfg.seed, args.repeats)
        r1 = fooled_rates(TRUE_GAP, cfg.seed, args.repeats)
        source = "SYNTHETIC (200 questions in 50 passages, two simulated models)"
        data_note = (
            "Real per-item results could not be downloaded here; rerun with `--data` for a "
            "real analysis. Nothing below is a claim about real models."
        )
        band = 2 * float(np.sqrt(0.05 * 0.95 / args.repeats))
        repeated = f"""
Unconditional behaviour over {args.repeats} fresh simulated evals of each kind (share of evals
in which the method says the models differ at the 5% level):

| method | World A (no difference): false positives | World B (6 points better): detections |
|---|---:|---:|
| unpaired normal (Welch) | {r0["unpaired"]:.1%} | {r1["unpaired"]:.1%} |
| paired normal (paired t) | {r0["paired"]:.1%} | {r1["paired"]:.1%} |
| paired + clustered (CR1) | {r0["cr1"]:.1%} | {r1["cr1"]:.1%} |
| paired score interval + McNemar | {r0["tango"]:.1%} | {r1["tango"]:.1%} |

Reading the false-positive column against the 5% it should be (Monte Carlo band
{100 * (0.05 - band):.1f}% to {100 * (0.05 + band):.1f}%): the unpaired test is
{word(r0["unpaired"], band)}, the paired t is {word(r0["paired"], band)}, the paired score
interval with McNemar is {word(r0["tango"], band)}, and the cluster-robust test is
{word(r0["cr1"], band)}. Pairing
matters for power: the unpaired test finds the real 6-point gap in {r1["unpaired"]:.0%} of
evals, against {r1["cr1"]:.0%} for the cluster-robust paired test, which removes the shared
difficulty and still counts passages rather than questions.
"""
        scan_note = (
            f"World A is seed {seed_a}, the first of {scanned} consecutive seeds in which the "
            "paired t-test (which ignores passages) declares the candidate better, so the "
            "problem is visible; how often that happens without conditioning is in section 5.\n"
        )
    if args.data:
        scan_note = ""

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    md = f"""# Case study 5: is the candidate model really better? (small LLM eval)

> **Data:** {source}.
> {data_note}

## 1. Question

Two models answered the same {N_PASSAGES * PER_PASSAGE} questions. Is the candidate better, and
by enough to matter (a margin of 3 accuracy points)? How many more questions would settle it?

## 2. Data and dependence structure

One 0/1 score per model and question, in per-item JSON Lines (`out/items_*.jsonl`). Two
dependences: both models answered **the same questions**, so scores are *paired* by question
(shared difficulty cancels in the difference), and questions come in **passages** of
{PER_PASSAGE} that share their source, so the per-question differences within a passage are
*correlated* and the effective sample is closer to the number of passages than of questions.
Hence `dependence: {{type: clustered, paired: true}}`.

## 3. Naive analysis, and why it misleads

Treat each model's questions as an independent sample and compare two accuracies with a normal
interval. Three things go wrong (section 4 shows them): the normal interval is too narrow for a
single model when the sample is small and accuracy is far from 50%; ignoring that both models
answered the same questions wastes the pairing and loses power; and ignoring passages makes the
paired test too confident.

{scan_note}
## 4. Correct analysis

{sections}
## 5. How often is each method fooled?
{repeated}
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
* Peak memory of this run: {peak_gb:.2f} GB.
"""
    (HERE / "REPORT.md").write_text(md)
    print(f"report written to {HERE / 'REPORT.md'}  (peak RSS {peak_gb:.2f} GB)")


if __name__ == "__main__":
    main()
