"""Failure gallery #1: the naive CV t-test vs Nadeau-Bengio, on a true null.

Two learners with identical expected error (y = x1 + x2 + noise; A uses x1, B uses x2) are
compared with repeated 10-fold CV on n = 100 rows. Any rejection is a false positive, so
the rejection rate should be about 5%.

Run:  uv run python validation/failure_gallery/01_naive_cv_ttest.py [--reps 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from isbetter.simulate.calibration import cv_type1  # noqa: E402

HERE = Path(__file__).parent
NAIVE_COLOR, NB_COLOR = "#d95f02", "#1b9e77"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    repeats = [1, 3, 5, 10]
    naive, nb, band = [], [], None
    for r in repeats:
        a, b = cv_type1(np.random.default_rng(args.seed + r), reps=args.reps, n=100, k=10, r=r)
        naive.append(a.rate)
        nb.append(b.rate)
        band = a.band

    lines = [
        f"False-positive rate at nominal 5% ({args.reps} simulated datasets per row, "
        f"Monte Carlo band {band[0]:.3f}-{band[1]:.3f}); n=100, 10-fold CV.\n",
        "| CV repeats | naive k-fold t | Nadeau-Bengio |",
        "|---:|---:|---:|",
    ]
    lines += [f"| {r} | {a:.3f} | {b:.3f} |" for r, a, b in zip(repeats, naive, nb, strict=True)]
    table = "\n".join(lines)
    print(table)
    (HERE / "01_naive_cv_ttest_results.md").write_text(table + "\n")

    x = np.arange(len(repeats))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    ax.bar(x - 0.19, naive, 0.38, color=NAIVE_COLOR, label="naive k-fold t-test")
    ax.bar(x + 0.19, nb, 0.38, color=NB_COLOR, label="Nadeau-Bengio")
    ax.axhline(0.05, color="black", lw=1, ls="--")
    ax.text(-0.45, 0.055, "nominal 5%", ha="left", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(r) for r in repeats])
    ax.set_xlabel("CV repeats (10 folds each)")
    ax.set_ylabel("false-positive rate")
    ax.set_title("Two equally good models: how often is one 'significantly better'?", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = HERE / "figures" / "01_naive_cv_ttest.png"
    fig.savefig(out)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
