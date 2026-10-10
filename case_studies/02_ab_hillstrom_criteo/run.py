"""Case study 2: an email A/B test (Hillstrom-style) and a 14M-row ad test (Criteo-style).

    uv run python case_studies/02_ab_hillstrom_criteo/run.py          # synthetic stand-ins
    uv run python case_studies/02_ab_hillstrom_criteo/run.py \\
        --hillstrom hillstrom.csv --criteo criteo.csv                 # real files

Neither real dataset could be downloaded in the environment this was built in (outbound access
to their hosts was blocked), so by default clearly labelled SYNTHETIC stand-ins with the same
shape are used. Their effect sizes are assumptions, not measurements: nothing in the report is a
claim about the real campaigns. Pass the real files to get the real analysis.
"""

from __future__ import annotations

import argparse
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from isbetter.adapters.load import load_data
from isbetter.core.config import load_config
from isbetter.core.experiment import ExperimentResult, build_experiment_plan, run_experiment
from isbetter.methods.parametric import welch_t
from isbetter.methods.power import sample_size_proportions
from isbetter.methods.proportions import two_proportion_z
from isbetter.methods.srm import srm_check
from isbetter.report.markdown import render_experiment_markdown
from isbetter.simulate.ab_data import hillstrom_like, write_csv

HERE = Path(__file__).parent
OUT = HERE / "out"


def demote(md: str, levels: int = 1) -> str:
    """Push every Markdown heading down ``levels`` levels (to nest a report in a section)."""
    return re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + levels) + " ", md, flags=re.M)


def body_after(md: str, heading: str) -> str:
    return md[md.index(heading) :]


def generate_criteo(path: Path, rows: int, seed: int, treated_share: float = 0.85) -> None:
    """Write the synthetic Criteo-shaped CSV in a child process so its memory stays separate."""
    code = (
        "import numpy as np, sys;"
        "from isbetter.simulate.ab_data import write_criteo_like_csv;"
        "write_criteo_like_csv(sys.argv[1], int(sys.argv[2]),"
        " np.random.default_rng(int(sys.argv[3])), treated_share=float(sys.argv[4]))"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(path), str(rows), str(seed), str(treated_share)],
        check=True,
    )


def ci(m: object) -> str:
    return f"[{m.ci_low:+.4f}, {m.ci_high:+.4f}]"  # type: ignore[attr-defined]


def hillstrom_section(source: str, data_path: Path) -> tuple[str, ExperimentResult]:
    cfg = load_config(HERE / "config.yaml")
    eplan = build_experiment_plan(cfg)
    table = load_data(cfg, eplan.plans, HERE)
    result = run_experiment(eplan, table)
    rep = render_experiment_markdown(result)

    base, cands = cfg.variants.baseline, cfg.variants.candidates
    # --- naive analysis: separate uncorrected 95% tests, covariate ignored, no SRM ------------
    naive_rows, naive_sig = [], 0
    for p, res in zip(eplan.plans, result.results, strict=True):
        for c in res.comparisons:
            a = table.select(c.candidate, p.metric)["value"]  # type: ignore[union-attr]
            b = table.select(base, p.metric)["value"]  # type: ignore[union-attr]
            n = (
                two_proportion_z(int(a.sum()), len(a), int(b.sum()), len(b))
                if (p.metric_kind == "binary")
                else welch_t(a, b)
            )
            if p.role == "success":
                naive_sig += n.p_value < 0.05
            adj = f"{c.adjusted_p:.3g}" if c.adjusted_p is not None else "n/a (guardrail)"
            naive_rows.append(
                f"| {p.metric} | {c.candidate} | {n.estimate:+.4f} | {ci(n)} | {n.p_value:.3g} | "
                f"{ci(c.primary)} ({c.primary.ci_level:.1%}) | {adj} |"
            )
    wins = sum(
        ch.status in ("win", "trivial_win")
        for d in result.decisions
        for ch in d.checks
        if ch.role == "success"
    )
    n_success = sum(
        len(r.comparisons)
        for p, r in zip(eplan.plans, result.results, strict=True)
        if p.role == "success"
    )

    # --- CUPED: CI width with and without ------------------------------------------------------
    cuped_rows, shrink, rhos = [], [], []
    for p, res in zip(eplan.plans, result.results, strict=True):
        if p.variance_reduction == "none":
            continue
        for c in res.comparisons:
            adj_m, raw_m = c.primary, c.cross_checks[0]
            w_adj, w_raw = adj_m.ci_high - adj_m.ci_low, raw_m.ci_high - raw_m.ci_low
            d = adj_m.diagnostics
            shrink.append(1 - w_adj / w_raw)
            rhos.append(abs(d["rho"]))
            cuped_rows.append(
                f"| {p.metric} | {c.candidate} | {w_raw:.4f} | {w_adj:.4f} | "
                f"{1 - w_adj / w_raw:.1%} | {d['variance_reduction']:.1%} | "
                f"{d['predicted_variance_reduction']:.1%} |"
            )

    cuped_txt = (
        f"CUPED narrows the intervals by {min(shrink):.1%} to {max(shrink):.1%} here. The gain is "
        "capped by rho^2, the squared within-arm correlation between `history` and the outcome "
        f"(|rho| is {min(rhos):.2f} to {max(rhos):.2f}): a covariate that barely predicts a "
        "zero-inflated outcome cannot buy much precision, and a stronger pre-period metric would."
        if shrink
        else "No metric uses a covariate."
    )

    # --- SRM: what a silent logging bug would look like ---------------------------------------
    counts = dict(result.srm.observed) if result.srm else {}
    lost = {**counts, cands[-1]: counts[cands[-1]] - 3000}
    planted = srm_check(lost, None, eplan.srm_alpha)

    # --- planning: how big should the test have been? ------------------------------------------
    p0 = float(table.select(base, "conversion")["value"].mean())  # type: ignore[union-attr]
    size = sample_size_proportions(p0, 0.002)

    inconclusive = [
        f"* {c.candidate} / {c.metric}: about {c.verdict.n_needed:,.0f} observations in total "
        f"(currently {c.n_pairs:,}). {c.verdict.n_needed_note or ''}".rstrip()
        for res in result.results
        for c in res.comparisons
        if c.verdict.label == "inconclusive" and c.verdict.n_needed
    ]
    inc_txt = (
        "\n".join(inconclusive)
        if inconclusive
        else ("No comparison is inconclusive at this sample size, so no extra data is needed.")
    )

    md = f"""## 1. Question

Should either email variant ("Mens E-Mail", "Womens E-Mail") replace sending no email? Success is
a lift in site visits or purchase conversion (a lift smaller than 1 percentage point of visits or
0.2 points of conversion is treated as trivial). The guardrail is revenue: spend per customer must
not fall by more than $0.10.

## 2. Data and dependence structure

{source}. One row per customer, randomised once to one of three arms, so units are independent:
`dependence = iid`. Arm counts: {", ".join(f"{k}: {v:,}" for k, v in counts.items())}.
Last year's spend (`history`) is measured before the experiment and is the CUPED covariate.

## 3. Naive analysis, and why it misleads

The naive recipe runs each metric and arm as its own 95% test, ignores the pre-experiment
covariate and never checks that randomisation delivered the planned split:

| metric | candidate | naive diff | naive 95% CI | naive p | corrected CI | Holm-adjusted p |
|---|---|---:|---|---:|---|---:|
{chr(10).join(naive_rows)}

* **Multiplicity.** {int(naive_sig)} of {n_success} success hypotheses are naively significant;
  {wins} of {n_success} remain wins after Holm. When effects are large the correction costs
  wider intervals but changes no conclusion; with several weak metrics it would. The correction
  spans success hypotheses only: guardrails are tested one-sided at the unadjusted level so that
  a regression is not made easier to miss.
* **Ignored covariate.** Leaving `history` out of the analysis forgoes some precision; section 5
  measures how much.
* **No SRM check.** Effect tests cannot detect broken assignment. If 3,000 customers vanished from
  `{cands[-1]}` through a logging bug the SRM check would report p = {planted.p_value:.2g}
  ({"flagged" if planted.flagged else "not flagged"}) while every effect estimate would still
  look fine.

## 4. Correct analysis

Automatic sample-ratio check, Holm across the {n_success} success hypotheses, one-sided
non-inferiority for the guardrail, then one decision per candidate.

{demote(body_after(rep, "## Sample-ratio check"))}
## 5. Verdict, effect size, CI, CUPED and `n_needed`

The decisions and effect sizes with confidence intervals are in section 4. `conversion` and
`spend` are analysed with CUPED using `history`; widths below are at the same confidence level,
with the unadjusted interval as the comparison:

| metric | candidate | CI width, no CUPED | CI width, CUPED | narrower by | variance cut | rho^2 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(cuped_rows)}

{cuped_txt}

More data needed where a comparison is inconclusive:

{inc_txt}

Planning: with the control conversion rate of {p0:.4f} observed here, detecting a +0.2 point lift
at 80% power needs about **{size.n_control:,} customers per arm** (`isbetter plan-size --kind
proportions --baseline {p0:.4f} --mde 0.002`).

## 6. What would change the conclusion

* A failed SRM check (a real one, not the planted example above) invalidates every number here.
* A real guardrail such as unsubscribes or complaints. The data have none; spend per customer
  stands in to show the rule, and a guardrail with a narrower margin needs more data to pass.
* Customers who receive several emails or appear in several rows would make `dependence = iid`
  wrong; declare `clustered` with a customer id (see failure gallery #5 for the damage ignoring
  that does).
* Real data: the stand-in effect sizes are assumptions.
"""
    return md, result


def criteo_section(source: str, rows: int, path: Path, seed: int) -> str:
    cfg = load_config(HERE / "config_criteo.yaml")
    eplan = build_experiment_plan(cfg)
    OUT.mkdir(exist_ok=True)
    if cfg.data and path != OUT / "criteo_like.csv":
        cfg = cfg.model_copy(update={"data": str(path)})
    t0 = time.time()
    data = load_data(cfg, eplan.plans, HERE)
    t_agg = time.time() - t0
    result = run_experiment(eplan, data)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    rep = render_experiment_markdown(result)

    # a planted logging bug: about 2% of treated rows are lost, planned split unchanged
    bug_rows = min(rows, 2_000_000)
    bug_path = OUT / "criteo_like_bug.csv"
    generate_criteo(bug_path, bug_rows, seed + 1, treated_share=0.8474)
    bug_cfg = cfg.model_copy(update={"data": str(bug_path)})
    bug = run_experiment(eplan, load_data(bug_cfg, eplan.plans, HERE))
    bug_path.unlink()

    n_total = sum(result.srm.observed.values()) if result.srm else 0
    assert bug.srm is not None
    srm_state = "**FAILED**" if bug.srm.flagged else "passed"
    verdicts = "; ".join(
        f"{d.candidate}: {'SHIP' if d.ship else 'DO NOT SHIP'}" for d in bug.decisions
    )
    return f"""## Appendix: scale test, a Criteo-style table aggregated without loading it

{source}

The file is never loaded into memory: DuckDB streams it and keeps only n, sum and sum of squares
per arm (`table.mode: stats`), and the same two-proportion z-tests, Holm family and decision rule
run from those {len(eplan.plans) * (len(eplan.candidates) + 1) * 3} numbers.

* rows aggregated: **{n_total:,}**
* aggregation + analysis time: **{t_agg:.1f} s**
* peak resident memory of the analysis process: **{peak:,.0f} MB** (includes the Python and
  library imports; the target is under 1,024 MB)

{demote(body_after(rep, "## Sample-ratio check"), 1)}
### Planted logging bug

The same analysis on {bug_rows:,} rows where about 2% of treated rows were lost
(treated share 84.7% instead of the planned 85%) with the planned split left unchanged:
SRM {srm_state} (p = {bug.srm.p_value:.2g}), so every decision is blocked: {verdicts}.
The effect estimates alone would not have revealed the problem.

### Verify before trusting real Criteo numbers

The build environment could not reach the Criteo host, so the dataset URL and layout were **not
verified here**. Before using `--criteo`, check (a) the download URL on the Criteo AI Lab uplift
dataset page, (b) that the file has `treatment`, `visit` and `conversion` columns, and (c)
whether rows are users or impressions. If a user can appear in several rows, the units are not
independent: aggregate to user level or declare `dependence: clustered` with a user id. As far as
I recall the public file has no user id column, in which case user-level clustering cannot be
corrected from the file alone and the intervals should be read as optimistic. This is
unverified.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hillstrom", help="path to the real Hillstrom CSV")
    ap.add_argument("--criteo", help="path to the real Criteo uplift CSV")
    ap.add_argument("--criteo-rows", type=int, default=14_000_000, help="synthetic Criteo rows")
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.hillstrom:
        h_path = Path(args.hillstrom)
        data_note = f"Real Hillstrom data from {h_path}"
        (OUT / "hillstrom.csv").write_bytes(h_path.read_bytes())
    else:
        write_csv(OUT / "hillstrom.csv", hillstrom_like(np.random.default_rng(args.seed)))
        data_note = (
            "**SYNTHETIC stand-in** shaped like the Hillstrom email challenge (64,000 customers, "
            "three arms of 21,306 / 21,307 / 21,387, zero-inflated spend). Effect sizes are assumed"
        )
    h_md, _ = hillstrom_section(data_note, OUT / "hillstrom.csv")

    if args.criteo:
        c_path, c_rows = Path(args.criteo), 0
        c_note = f"Real Criteo data from {c_path}."
    else:
        c_path = OUT / "criteo_like.csv"
        c_rows = args.criteo_rows
        generate_criteo(c_path, c_rows, args.seed)
        c_note = (
            f"**SYNTHETIC stand-in**: {c_rows:,} rows with columns `treatment, visit, conversion`, "
            "about 85% treated and rare binary outcomes. Rates are assumptions, not measurements."
        )
    c_md = criteo_section(c_note, c_rows or 14_000_000, c_path, args.seed)
    if not args.criteo:
        c_path.unlink()

    head = """# Case study 2: an email A/B test and a Criteo-scale ad test

> **Data:** both datasets are SYNTHETIC stand-ins here (the real files could not be downloaded
> in the build environment). Rerun with `--hillstrom` and `--criteo` for the real analysis.
> Nothing below is a claim about real campaigns.

"""
    md = head + h_md + "\n" + c_md
    (HERE / "REPORT.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
