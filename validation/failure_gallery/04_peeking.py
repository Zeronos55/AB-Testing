"""Failure gallery #4: peeking at a fixed-horizon test inflates false positives.

A fixed-horizon test is valid if you look once, at the planned sample size. Experiments are
usually watched: someone checks the dashboard every day and ships as soon as ``p < 0.05``.
Here two *identical* arms (N(0, 1), so any "win" is a false positive) are monitored up to 2,000
observations per arm. The more often the test is read, the more often it declares a difference
that does not exist. The normal-mixture confidence sequence (``isbetter.methods.sequential``)
is valid at every look, so its false-positive rate does not grow however often it is read.

Run:  uv run python validation/failure_gallery/04_peeking.py [--reps 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from isbetter.methods.sequential import mixture_boundary, optimal_rho  # noqa: E402
from isbetter.simulate.calibration import peeking_type1  # noqa: E402

HERE = Path(__file__).parent
FIXED_COLOR, CS_COLOR = "#d95f02", "#1b9e77"
N_MAX, MIN_N, ALPHA = 2_000, 20, 0.05
LOOKS = [1, 2, 5, 10, 20, 50, 200, None]  # None = after every observation (1,981 looks)


def example_path() -> tuple[np.ndarray, np.ndarray, int]:
    """First seed whose A/A path crosses 1.96 at some look while staying inside the sequence."""
    t = np.arange(MIN_N, N_MAX + 1)
    crit = stats.t.ppf(1 - ALPHA / 2, 2 * (t - 1))
    boundary = mixture_boundary(t, optimal_rho(N_MAX, ALPHA), ALPHA) / np.sqrt(t)
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        a, b = rng.normal(size=N_MAX), rng.normal(size=N_MAX)
        ma, mb = np.cumsum(a)[MIN_N - 1 :] / t, np.cumsum(b)[MIN_N - 1 :] / t
        va = (np.cumsum(a**2)[MIN_N - 1 :] - t * ma**2) / (t - 1)
        vb = (np.cumsum(b**2)[MIN_N - 1 :] - t * mb**2) / (t - 1)
        z = (ma - mb) / np.sqrt((va + vb) / t)
        if (np.abs(z) > crit).any() and (np.abs(z) < boundary).all():
            return t, z, seed
    raise RuntimeError("no example path found")  # pragma: no cover


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rows = []
    for i, k in enumerate(LOOKS):
        r = peeking_type1(
            np.random.default_rng(args.seed + i),
            args.reps,
            n_max=N_MAX,
            n_looks=k,
            n_target=N_MAX,
            min_n=MIN_N,
        )
        rows.append((r.n_looks, r))
    band = rows[0][1].single_look.band
    lines = [
        f"False-positive rate at nominal 5% when two identical arms are read at several looks "
        f"(up to {N_MAX:,} observations per arm; {args.reps} simulated experiments per row, "
        f"Monte Carlo band {band[0]:.3f}-{band[1]:.3f}).\n",
        "| number of looks | fixed-horizon test, stop at first p < 0.05 | confidence sequence "
        "(isbetter) | fixed-horizon test read once at the end |",
        "|---:|---:|---:|---:|",
    ]
    for n_looks, r in rows:
        lines.append(
            f"| {n_looks:,} | {r.peeked.rate:.3f} | {r.sequence.rate:.3f} | "
            f"{r.single_look.rate:.3f} |"
        )
    lines.append(
        "\nThe fixed-horizon test is fine when it is read once, and its false-positive rate climbs "
        "with every extra look (a daily check over a few weeks is the 20-look row). The "
        "confidence sequence stays below 5% at every number of looks; it is conservative (about "
        "2% when read continuously) because it pays for being valid at all sample sizes with a "
        "wider interval: about 3.0 standard errors instead of 1.96 at the sample size it is "
        "tuned for."
    )
    table = "\n".join(lines)
    print(table)
    (HERE / "04_peeking_results.md").write_text(table + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=150)
    xs = [n for n, _ in rows]
    ax1.plot(
        xs, [r.peeked.rate for _, r in rows], "o-", color=FIXED_COLOR, label="fixed-horizon test"
    )
    ax1.plot(
        xs, [r.sequence.rate for _, r in rows], "o-", color=CS_COLOR, label="confidence sequence"
    )
    ax1.axhline(ALPHA, color="black", lw=1, ls="--")
    ax1.text(1.1, ALPHA + 0.01, "nominal 5%", fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("number of times the experiment is read")
    ax1.set_ylabel("false-positive rate (identical arms)")
    ax1.set_ylim(0, 0.55)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_title("Reading it more often, winning more often", fontsize=10)

    t, z, seed = example_path()
    boundary = mixture_boundary(t, optimal_rho(N_MAX, ALPHA), ALPHA) / np.sqrt(t)
    ax2.plot(t, z, color="#444444", lw=0.9, label="difference / standard error")
    ax2.axhline(1.96, color=FIXED_COLOR, lw=1.2)
    ax2.axhline(-1.96, color=FIXED_COLOR, lw=1.2, label="fixed-horizon ±1.96")
    ax2.plot(t, boundary, color=CS_COLOR, lw=1.2, label="confidence sequence")
    ax2.plot(t, -boundary, color=CS_COLOR, lw=1.2)
    ax2.set_xlabel("observations per arm")
    ax2.set_title(f"One A/A experiment (seed {seed}): no true difference", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = HERE / "figures" / "04_peeking.png"
    fig.savefig(out)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
