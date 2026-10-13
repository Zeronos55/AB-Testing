"""Failure gallery #3: the normal-theory (CLT) error bar undercovers at N = 30-200 items.

An LLM eval often has only 30-200 items. The usual interval ``mean +/- 1.96 * sd / sqrt(n)``
relies on the central limit theorem, which is a poor approximation when n is small and accuracy
is far from 50%: the interval is too narrow and, near 0% or 100%, collapses to a point when every
item is right. Wilson, Jeffreys and Clopper-Pearson intervals stay close to (or above) nominal.

Single-arm coverage is computed *exactly* (the correct count is Binomial(n, p), so coverage is a
finite sum), with no simulation. The paired table simulates two correlated models and compares
the normal-theory paired interval with the Tango score interval used by ``isbetter``.

Run:  uv run python validation/failure_gallery/03_clt_small_n.py [--reps 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from isbetter.methods.proportions import jeffreys_interval, wilson_interval  # noqa: E402
from isbetter.methods.small_n import clopper_pearson_interval  # noqa: E402
from isbetter.simulate.calibration import (  # noqa: E402
    clt_proportion_interval,
    exact_proportion_coverage_curve,
    paired_binary_coverage,
)

HERE = Path(__file__).parent
METHODS = {
    "CLT (normal)": (clt_proportion_interval, "#d95f02"),
    "Wilson": (wilson_interval, "#1b9e77"),
    "Jeffreys": (jeffreys_interval, "#7570b3"),
    "Clopper-Pearson": (clopper_pearson_interval, "#666666"),
}
NS = [30, 50, 100, 200]
ACCURACIES = [0.5, 0.8, 0.9, 0.95]
GRID = np.linspace(0.02, 0.98, 481)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    curves = {
        (name, n): exact_proportion_coverage_curve(f, n, GRID)
        for name, (f, _) in METHODS.items()
        for n in NS
    }
    lines = [
        "Exact coverage of nominal 95% intervals for one model's accuracy (no simulation).\n",
        "| N items | true accuracy | " + " | ".join(METHODS) + " |",
        "|---:|---:|" + "---:|" * len(METHODS),
    ]
    for n in NS:
        for p in ACCURACIES:
            i = int(np.argmin(np.abs(GRID - p)))
            row = " | ".join(f"{curves[(m, n)][i]:.3f}" for m in METHODS)
            lines.append(f"| {n} | {p:.2f} | {row} |")
    lines += [
        "\nAverage coverage over true accuracies 0.02-0.98, and the worst case:\n",
        "| N items | " + " | ".join(f"{m} (mean / min)" for m in METHODS) + " |",
        "|---:|" + "---:|" * len(METHODS),
    ]
    for n in NS:
        cells = " | ".join(
            f"{curves[(m, n)].mean():.3f} / {curves[(m, n)].min():.3f}" for m in METHODS
        )
        lines.append(f"| {n} | {cells} |")

    lines += [
        f"\nPaired difference of two models ({args.reps} simulated evals per row, models' "
        "correctness correlated at 0.4): coverage of the nominal 95% interval for "
        "acc_A - acc_B.\n",
        "| N items | acc A / acc B | normal-theory paired | Tango score (isbetter) |",
        "|---:|---|---:|---:|",
    ]
    for i, (n, pa, pb) in enumerate([(30, 0.90, 0.85), (60, 0.95, 0.90), (100, 0.80, 0.75)]):
        w, s = paired_binary_coverage(np.random.default_rng(args.seed + i), args.reps, n, pa, pb)
        lines.append(f"| {n} | {pa:.2f} / {pb:.2f} | {w.rate:.3f} | {s.rate:.3f} |")
    lines.append(
        "\nCoverage of a discrete count is a sawtooth, so no interval is exact at every accuracy "
        "(Jeffreys dips to about 0.89 at N = 30-50). The normal-theory interval is the one that "
        "fails badly: at N = 30 it covers 78-81% when accuracy is 0.90-0.95 and 45% at its worst "
        "accuracy, where Wilson stays above 0.91 and Clopper-Pearson above 0.95. By N = 200 the "
        "gap is small except near the extremes. The paired normal-theory interval undercovers "
        "the same way when both models are strong."
    )
    table = "\n".join(lines)
    print(table)
    (HERE / "03_clt_small_n_results.md").write_text(table + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=150, sharey=True)
    for ax, n in zip(axes, (30, 100), strict=True):
        for name, (_, color) in METHODS.items():
            ax.plot(GRID, curves[(name, n)], color=color, lw=1.1, label=name)
        ax.axhline(0.95, color="black", lw=1, ls="--")
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("true accuracy of the model")
        ax.set_title(f"N = {n} items", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("coverage of the 95% interval")
    axes[0].legend(frameon=False, fontsize=8, loc="lower center")
    fig.tight_layout()
    out = HERE / "figures" / "03_clt_small_n.png"
    fig.savefig(out)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
