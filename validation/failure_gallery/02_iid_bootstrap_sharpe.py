"""Failure gallery #2: an iid bootstrap interval for a Sharpe ratio is too narrow.

Strategy returns come from an AR(1) process with GARCH(1,1) Student-t innovations (T = 500
periods, true per-period Sharpe ratio 0.05). Positive autocorrelation makes the mean return
more variable than independent sampling implies, so resampling single periods (an iid
bootstrap) gives intervals that are too short and a "95%" interval misses the truth far more
than 5% of the time. Resampling blocks keeps the dependence inside each block and recovers
most, not all, of the coverage.

Run:  uv run python validation/failure_gallery/02_iid_bootstrap_sharpe.py [--reps 1000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from isbetter.simulate.calibration import sharpe_ci_coverage  # noqa: E402

HERE = Path(__file__).parent
IID_COLOR, BLOCK_COLOR = "#d95f02", "#1b9e77"
T = 500
PHIS = [0.0, 0.1, 0.2, 0.3, 0.4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    iid, blk, band = [], [], None
    for i, phi in enumerate(PHIS):
        a, b = sharpe_ci_coverage(np.random.default_rng(args.seed + i), args.reps, T=T, phi=phi)
        iid.append(a.rate)
        blk.append(b.rate)
        band = a.band

    assert band is not None
    lines = [
        f"Coverage of nominal 95% Sharpe-ratio intervals ({args.reps} simulated return series "
        f"per row, Monte Carlo band {band[0]:.3f}-{band[1]:.3f}); T={T} periods, AR(1) "
        "coefficient phi, GARCH(1,1) t(5) innovations, true per-period Sharpe 0.05.\n",
        "| autocorrelation phi | iid bootstrap | block bootstrap (Politis-White length) |",
        "|---:|---:|---:|",
    ]
    lines += [f"| {p:.1f} | {a:.3f} | {b:.3f} |" for p, a, b in zip(PHIS, iid, blk, strict=True)]
    lines.append(
        "\nThe iid bootstrap is fine only without autocorrelation (phi = 0) and degrades quickly "
        "as it grows. The block bootstrap holds up much better but is not exact: with T=500 it "
        "stays about 2-4 points short (see tests/calibration/test_sharpe_coverage.py), and at "
        "phi = 0 it gives up a little coverage for estimating a dependence that is not there."
    )
    table = "\n".join(lines)
    print(table)
    (HERE / "02_iid_bootstrap_sharpe_results.md").write_text(table + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    ax.plot(PHIS, iid, "o-", color=IID_COLOR, label="iid bootstrap")
    ax.plot(PHIS, blk, "o-", color=BLOCK_COLOR, label="block bootstrap")
    ax.axhline(0.95, color="black", lw=1, ls="--")
    ax.text(PHIS[-1], 0.953, "nominal 95%", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("autocorrelation of strategy returns (phi)")
    ax.set_ylabel("coverage of the 95% Sharpe interval")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("How often does the 95% interval contain the true Sharpe ratio?", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = HERE / "figures" / "02_iid_bootstrap_sharpe.png"
    fig.savefig(out)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
