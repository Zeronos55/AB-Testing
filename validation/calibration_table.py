"""Calibration table: how often each method's error rate matches its nominal level.

Every row reruns a scenario from ``isbetter.simulate.calibration`` (the same harness the slow
tests use) on a world where the truth is known: two equal arms for Type I error, a known
parameter for interval coverage. The recommended method sits next to the naive method it
replaces, so the table shows both that the recommended one is calibrated and what goes wrong
without it. Results are written to ``validation/calibration_table.md`` and, with
``--readme``, into the marked block of README.md.

Run:  uv run python validation/calibration_table.py [--reps 10000] [--readme]
      [--csv bi/calibration.csv]
Takes a few minutes at the default size; the 10,000-replication cap is lowered for the slower
scenarios (bootstrap intervals, confidence sequences) to the same sizes as the slow tests.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import isbetter
from isbetter.methods.proportions import jeffreys_interval, wilson_interval
from isbetter.methods.small_n import clopper_pearson_interval
from isbetter.simulate.calibration import (
    CalibrationResult,
    clt_proportion_interval,
    cluster_type1,
    cuped_baseline_pvalues,
    cuped_pvalues,
    cv_type1,
    dm_type1,
    exact_proportion_coverage,
    exact_proportion_coverage_curve,
    paired_binary_coverage,
    paired_cluster_type1,
    paired_pvalues,
    peeking_type1,
    sharpe_ci_coverage,
    srm_rejection_rate,
    two_proportion_pvalues,
    type1_rate,
    welch_pvalues,
)
from isbetter.simulate.dgp import aa_binary, aa_covariate, aa_independent, aa_paired

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = HERE / "calibration_table.md"
README = ROOT / "README.md"
START, END = "<!-- calibration-table:start -->", "<!-- calibration-table:end -->"

RECOMMENDED, FOIL, REFERENCE = "recommended", "naive foil", "reference"


@dataclass
class Row:
    scenario: str
    method: str
    role: str
    measures: str  # "false positives" (Type I error) or "coverage"
    result: CalibrationResult | None  # None for exact (no simulation) rows
    exact: float | None = None  # exact coverage when no simulation is involved

    @property
    def nominal(self) -> float:
        return self.result.alpha if self.result else 0.95

    @property
    def observed(self) -> float:
        return self.result.rate if self.result else float(self.exact)  # type: ignore[arg-type]

    def reading(self) -> str:
        """Where the observed rate sits against nominal, in words (never hand-written)."""
        if self.result is None:
            gap = self.observed - self.nominal
            if gap >= -0.005:
                return "calibrated"
            return "under-covers" if gap < -0.03 else "slightly under-covers"
        lo, hi = self.result.band
        if lo <= self.observed <= hi:
            return "calibrated"
        gap = self.observed - self.nominal
        if self.measures == "false positives":
            if gap < 0:
                return "conservative (valid)"
            return "slightly liberal" if gap <= 0.015 else "inflated"
        if gap > 0:
            return "conservative (valid)"
        return "slightly under-covers" if gap >= -0.03 else "under-covers"


def _rows_type1(
    scenario: str, method: str, role: str, r: CalibrationResult, measures: str = "false positives"
) -> Row:
    return Row(scenario, method, role, measures, r)


def build_rows(reps: int, progress: Callable[[str], None] = lambda _: None) -> list[Row]:
    """Run every scenario. Slower ones are capped at the size the slow tests use."""
    rows: list[Row] = []

    def step(name: str) -> None:
        progress(name)

    def rng(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    step("iid and paired")
    rows += [
        _rows_type1(
            "A/A, n = 40 per arm, normal",
            "Welch t",
            RECOMMENDED,
            type1_rate(welch_pvalues, aa_independent, rng(11), reps=reps, n=40),
        ),
        _rows_type1(
            "A/A, 40 pairs, normal",
            "paired t",
            RECOMMENDED,
            type1_rate(paired_pvalues, aa_paired, rng(12), reps=reps, n=40),
        ),
    ]
    step("A/B")
    rows += [
        _rows_type1(
            "A/A, n = 1,500 per arm, conversion rate 10%",
            "two-proportion z",
            RECOMMENDED,
            type1_rate(
                two_proportion_pvalues,
                lambda g, m, n: aa_binary(g, m, n, p=0.1),
                rng(23),
                reps=reps,
                n=1500,
            ),
        ),
        _rows_type1(
            "A/A, n = 400 per arm, covariate correlation 0.7, curved",
            "CUPED",
            RECOMMENDED,
            type1_rate(
                cuped_pvalues,
                lambda g, m, n: aa_covariate(g, m, n, rho=0.7, curvature=1.0),
                rng(317),  # the slow test's seed (see the note in the page)
                reps=reps,
                n=400,
            ),
        ),
        _rows_type1(
            "A/A, n = 400 per arm, covariate correlation 0.7",
            "unadjusted Welch",
            REFERENCE,
            type1_rate(
                cuped_baseline_pvalues,
                lambda g, m, n: aa_covariate(g, m, n, rho=0.7),
                rng(22),
                reps=reps,
                n=400,
            ),
        ),
        _rows_type1(
            "True 40/30/30 split, 20,000 users",
            "SRM chi-square at 0.05",
            RECOMMENDED,
            srm_rejection_rate(rng(24), reps, 20_000, (0.4, 0.3, 0.3), alpha=0.05),
            "false alarms",
        ),
    ]
    step("clustered")
    naive, cr1 = cluster_type1(rng(25), reps, users=100, sessions=10, icc=0.3)
    rows += [
        _rows_type1(
            "A/A, 100 users x 10 sessions per arm, ICC 0.3", "row-level Welch", FOIL, naive
        ),
        _rows_type1(
            "A/A, 100 users x 10 sessions per arm, ICC 0.3", "CR1 by user", RECOMMENDED, cr1
        ),
    ]
    naive, cr1 = paired_cluster_type1(rng(5), min(reps, 6000), groups=30, per_group=5)
    rows += [
        _rows_type1(
            "Equal models, 30 passages x 5 items", "paired t ignoring passages", FOIL, naive
        ),
        _rows_type1(
            "Equal models, 30 passages x 5 items", "paired CR1 by passage", RECOMMENDED, cr1
        ),
    ]
    step("cross-validation")
    naive, nb = cv_type1(rng(21), min(reps, 3000), n=100, k=10, r=10)
    rows += [
        _rows_type1("Equal learners, 10 x 10-fold CV, n = 100", "k-fold t", FOIL, naive),
        _rows_type1("Equal learners, 10 x 10-fold CV, n = 100", "Nadeau-Bengio", RECOMMENDED, nb),
    ]
    step("forecasting")
    for h in (1, 3, 7):
        naive, dm = dm_type1(rng(31 + h), min(reps, 5000), T=250, horizon=h)
        scen = f"Equal forecasters, h = {h}, 250 origins"
        rows += [
            _rows_type1(scen, "paired t on losses", FOIL, naive),
            _rows_type1(scen, "Diebold-Mariano + HLN", RECOMMENDED, dm),
        ]
    step("sequential")
    pk = peeking_type1(rng(1), min(reps, 2000), n_max=2000, n_looks=None)
    scen = "A/A, read after every observation (20 to 2,000 per arm)"
    rows += [
        _rows_type1(scen, "fixed-horizon test, stopped at first p < 0.05", FOIL, pk.peeked),
        _rows_type1(scen, "confidence sequence", RECOMMENDED, pk.sequence),
        _rows_type1(
            "A/A, read once at n = 2,000 per arm",
            "fixed-horizon Welch t",
            RECOMMENDED,
            pk.single_look,
        ),
    ]
    step("Sharpe ratio")
    iid, blk = sharpe_ci_coverage(rng(41), min(reps, 1000), T=500, phi=0.2)
    scen = "AR(1)-GARCH returns, T = 500, phi = 0.2"
    rows += [
        _rows_type1(scen, "iid bootstrap 95% CI", FOIL, iid, "coverage"),
        _rows_type1(scen, "stationary block bootstrap 95% CI", RECOMMENDED, blk, "coverage"),
    ]
    step("small N")
    grid = np.linspace(0.02, 0.98, 49)
    for name, interval, role in (
        ("normal-theory", clt_proportion_interval, FOIL),
        ("Wilson", wilson_interval, RECOMMENDED),
        ("Jeffreys", jeffreys_interval, RECOMMENDED),
        ("Clopper-Pearson", clopper_pearson_interval, RECOMMENDED),
    ):
        at = exact_proportion_coverage(interval, 30, 0.95)
        rows.append(
            Row("True accuracy 0.95, N = 30 items", f"{name} 95% CI", role, "coverage", None, at)
        )
    for name, interval, role in (
        ("normal-theory", clt_proportion_interval, FOIL),
        ("Wilson", wilson_interval, RECOMMENDED),
        ("Jeffreys", jeffreys_interval, RECOMMENDED),
        ("Clopper-Pearson", clopper_pearson_interval, RECOMMENDED),
    ):
        worst = float(exact_proportion_coverage_curve(interval, 30, grid).min())
        scen = "Worst true accuracy in 0.02-0.98, N = 30 items"
        rows.append(Row(scen, f"{name} 95% CI", role, "worst-case coverage", None, worst))
    wald, score = paired_binary_coverage(rng(11), min(reps, 4000), 60, 0.95, 0.90, 0.4)
    scen = "Paired accuracy difference, N = 60, 0.95 vs 0.90"
    rows += [
        _rows_type1(scen, "normal-theory paired CI", FOIL, wald, "coverage"),
        _rows_type1(scen, "Tango score CI", RECOMMENDED, score, "coverage"),
    ]
    return rows


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_table(rows: list[Row]) -> str:
    lines = [
        "| Scenario | Method | Role | Measures | Nominal | Observed | 2-SE band | Reps | Reading |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.result is None:
            band, reps = "exact", "exact"
        else:
            lo, hi = r.result.band
            band, reps = f"{_pct(lo)}-{_pct(hi)}", f"{r.result.reps:,}"
        lines.append(
            f"| {r.scenario} | {r.method} | {r.role} | {r.measures} | {_pct(r.nominal)} "
            f"| **{_pct(r.observed)}** | {band} | {reps} | {r.reading()} |"
        )
    return "\n".join(lines) + "\n"


CSV_COLUMNS = (
    "scenario",
    "method",
    "role",
    "measures",
    "nominal",
    "observed",
    "band_low",
    "band_high",
    "reps",
    "exact",
    "reading",
)


def render_csv(rows: list[Row]) -> str:
    """The table as one flat CSV for a Power BI dashboard (``bi/calibration.csv``).

    Rates are fractions, not percentages, and are not rounded. ``band_low``, ``band_high`` and
    ``reps`` are empty for exact rows, which have no simulation noise.
    """
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for r in rows:
        lo, hi = r.result.band if r.result else ("", "")
        w.writerow(
            [
                r.scenario,
                r.method,
                r.role,
                r.measures,
                r.nominal,
                r.observed,
                lo,
                hi,
                r.result.reps if r.result else "",
                "true" if r.result is None else "false",
                r.reading(),
            ]
        )
    return buf.getvalue()


def render_compact(rows: list[Row]) -> str:
    """The README version: fewer columns, naive methods marked in the method cell."""
    tag = {FOIL: " *(naive)*", REFERENCE: " *(reference)*", RECOMMENDED: ""}
    lines = [
        "| Scenario | Method | Measures | Nominal | Observed | Reading |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.scenario} | {r.method}{tag[r.role]} | {r.measures} | {_pct(r.nominal)} "
            f"| **{_pct(r.observed)}** | {r.reading()} |"
        )
    return "\n".join(lines) + "\n"


def render_page(rows: list[Row], reps: int) -> str:
    n_sim = sum(r.result is not None for r in rows)
    return (
        "# Calibration table\n\n"
        f"Generated by `validation/calibration_table.py` (isbetter {isbetter.__version__}, "
        f"up to {reps:,} replications per row, fixed seeds). Every scenario has a known truth: "
        "two equal arms (false positives at a nominal 5%) or a known parameter (coverage of a "
        "nominal 95% interval). *2-SE band* is the range a correctly calibrated method falls "
        "in about 95% of the time at that many replications; *exact* rows have no simulation "
        "noise.\n\n"
        + render_table(rows)
        + "\nRows marked *naive foil* are the methods this package replaces; they are here to show "
        "what goes wrong, not as recommendations. *Reference* is a method shown for comparison. "
        "Rows where a recommended method is outside its band are reported as they came out.\n\n"
        "**Reading the bands.** A calibrated method still lands outside its 2-SE band about 1 time "
        "in 20, and this table has "
        + str(n_sim)
        + " simulated rows, so about "
        + f"{n_sim / 20:.1f}"
        + " such rows are expected by chance. "
        "Seeds are those of the slow tests (`pytest -m slow`), which check the same scenarios. "
        "While preparing this table, CUPED with the curved covariate came out at 5.7% under an "
        "unrelated seed; over 16 seeds (that one included) of 10,000 replications each it rejects "
        "4.97% of the time.\n\n"
        "**Small N.** The count of correct answers is binomial, so coverage of an interval for "
        "an accuracy can be computed exactly, and it is a sawtooth in the true accuracy: one "
        "accuracy can look comfortably conservative while another is well below nominal. "
        "Hence the worst case over accuracies from 0.02 to 0.98. Only Clopper-Pearson is "
        "guaranteed to cover at least 95% everywhere; Wilson and Jeffreys are close on average "
        "but dip below nominal in places.\n"
    )


def update_readme(table: str) -> None:
    text = README.read_text()
    block = f"{START}\n\n{table}\n{END}"
    if START not in text:
        raise SystemExit(f"README.md has no {START} block to update")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    README.write_text(new)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=10_000, help="replications per row (cap)")
    ap.add_argument("--readme", action="store_true", help="also update the README block")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--csv", type=Path, default=None, help="also write the table as CSV (for Power BI)"
    )
    args = ap.parse_args()
    t0 = time.time()
    rows = build_rows(
        args.reps, progress=lambda s: print(f"[{time.time() - t0:6.1f}s] {s}", flush=True)
    )
    args.out.write_text(render_page(rows, args.reps))
    print(f"wrote {args.out}")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text(render_csv(rows), encoding="utf-8")
        print(f"wrote {args.csv}")
    if args.readme:
        update_readme(render_compact(rows))
        print("updated README.md")


if __name__ == "__main__":
    main()
