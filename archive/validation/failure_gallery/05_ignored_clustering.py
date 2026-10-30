"""Failure gallery #5: ignoring user-level clustering inflates the false-positive rate.

Users are randomised to A or B and each user contributes several sessions. Sessions of one user
share a user effect (intra-class correlation, ICC), so 10 sessions from 100 users are NOT 1,000
independent observations. Both arms are identical, so any "significant" result is false.

The naive analysis treats every session row as an independent unit (Welch t on rows). The
cluster-robust analysis (CR1, users as clusters, t with G - 1 df) should reject 5% of the time.

Run:  uv run python validation/failure_gallery/05_ignored_clustering.py [--reps 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from isbetter.simulate.calibration import cluster_type1  # noqa: E402

HERE = Path(__file__).parent
NAIVE_COLOR, CR_COLOR = "#d95f02", "#1b9e77"
USERS = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    by_sessions = [1, 2, 5, 10, 20]
    by_icc = [0.0, 0.05, 0.2, 0.5]
    rows_a, rows_b = [], []
    band = None
    for s in by_sessions:
        naive, cr = cluster_type1(
            np.random.default_rng(args.seed + s), args.reps, USERS, s, icc=0.3
        )
        rows_a.append((s, naive.rate, cr.rate))
        band = naive.band
    for icc in by_icc:
        naive, cr = cluster_type1(
            np.random.default_rng(args.seed + 100 + int(icc * 100)), args.reps, USERS, 10, icc=icc
        )
        rows_b.append((icc, naive.rate, cr.rate))

    def design_effect(sessions: int, icc: float) -> float:
        return 1 + (sessions - 1) * icc

    lines = [
        f"False-positive rate at nominal 5% ({args.reps} simulated experiments per row, Monte "
        f"Carlo band {band[0]:.3f}-{band[1]:.3f}); {USERS} users per arm, identical arms.\n",
        "Varying sessions per user (ICC = 0.3):\n",
        "| sessions/user | rows analysed | design effect | naive row-level Welch "
        "| cluster-robust CR1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {s} | {2 * USERS * s:,} | {design_effect(s, 0.3):.1f} | {a:.3f} | {c:.3f} |"
        for s, a, c in rows_a
    ]
    lines += [
        "\nVarying the intra-class correlation (10 sessions per user):\n",
        "| ICC | design effect | naive row-level Welch | cluster-robust CR1 |",
        "|---:|---:|---:|---:|",
    ]
    lines += [f"| {i:g} | {design_effect(10, i):.1f} | {a:.3f} | {c:.3f} |" for i, a, c in rows_b]
    table = "\n".join(lines)
    print(table)
    (HERE / "05_ignored_clustering_results.md").write_text(table + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=150, sharey=True)
    for ax, rows, xs, xlabel, title in (
        (axes[0], rows_a, [str(r[0]) for r in rows_a], "sessions per user", "ICC = 0.3"),
        (axes[1], rows_b, [f"{r[0]:g}" for r in rows_b], "intra-class correlation", "10 sessions"),
    ):
        x = np.arange(len(rows))
        ax.bar(
            x - 0.19, [r[1] for r in rows], 0.38, color=NAIVE_COLOR, label="naive (rows as units)"
        )
        ax.bar(x + 0.19, [r[2] for r in rows], 0.38, color=CR_COLOR, label="cluster-robust (users)")
        ax.axhline(0.05, color="black", lw=1, ls="--")
        ax.set_xticks(x, xs)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("false-positive rate")
    axes[0].text(-0.45, 0.053, "nominal 5%", ha="left", va="bottom", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Identical arms, randomised by user: how often is B 'significantly better'?", fontsize=10
    )
    fig.tight_layout()
    out = HERE / "figures" / "05_ignored_clustering.png"
    fig.savefig(out)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
