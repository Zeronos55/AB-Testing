"""Failure gallery #2b: a Diebold-Mariano test at horizon h > 1 that ignores autocorrelation.

Two forecasters with identical expected loss are compared on T = 100 rolling forecast origins.
An h-step-ahead error overlaps the errors of the next h - 1 origins, so the loss differential
is autocorrelated. A plain paired t-test on it (variance gamma_0 / T) treats the origins as
independent and rejects a true null far too often as h grows; Diebold-Mariano with the
Harvey-Leybourne-Newbold correction uses h - 1 autocovariance lags. Any rejection is a false
positive, so the rate should be about 5%.

Run:  uv run python validation/failure_gallery/02b_dm_horizon.py [--reps 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from isbetter.simulate.calibration import dm_type1  # noqa: E402

HERE = Path(__file__).parent
NAIVE_COLOR, DM_COLOR = "#d95f02", "#1b9e77"
T = 100
HORIZONS = [1, 2, 3, 5, 7, 10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    naive, dm, band = [], [], None
    for h in HORIZONS:
        a, b = dm_type1(np.random.default_rng(args.seed + h), reps=args.reps, T=T, horizon=h)
        naive.append(a.rate)
        dm.append(b.rate)
        band = a.band

    assert band is not None
    lines = [
        f"False-positive rate at nominal 5% ({args.reps} simulated datasets per row, "
        f"Monte Carlo band {band[0]:.3f}-{band[1]:.3f}); T={T} forecast origins, "
        "two equally accurate forecasters, squared-error loss.\n",
        "| horizon h | naive paired t | Diebold-Mariano + HLN |",
        "|---:|---:|---:|",
    ]
    lines += [f"| {h} | {a:.3f} | {b:.3f} |" for h, a, b in zip(HORIZONS, naive, dm, strict=True)]
    lines.append(
        "\nDiebold-Mariano + HLN is far closer to 5% but not exact: with only T=100 origins it "
        "stays slightly liberal at long horizons (see tests/calibration/test_dm_type1.py)."
    )
    table = "\n".join(lines)
    print(table)
    (HERE / "02b_dm_horizon_results.md").write_text(table + "\n")

    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    ax.bar(x - 0.19, naive, 0.38, color=NAIVE_COLOR, label="naive paired t-test")
    ax.bar(x + 0.19, dm, 0.38, color=DM_COLOR, label="Diebold-Mariano + HLN")
    ax.axhline(0.05, color="black", lw=1, ls="--")
    ax.text(-0.45, 0.055, "nominal 5%", ha="left", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(h) for h in HORIZONS])
    ax.set_xlabel("forecast horizon h (steps ahead)")
    ax.set_ylabel("false-positive rate")
    ax.set_title(
        "Two equally good forecasters: how often is one 'significantly better'?", fontsize=10
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = HERE / "figures" / "02b_dm_horizon.png"
    fig.savefig(out)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
