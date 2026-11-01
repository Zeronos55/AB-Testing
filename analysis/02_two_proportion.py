"""Step 3: is the lift from each email real, or could it be luck? (two-proportion z-test)

For each email group versus the no-email group, on two yes/no outcomes (bought, visited):
the difference in rates, a 95% interval for it and a p-value for "the email does nothing".
The test and interval are written by hand, then checked against statsmodels and SciPy.
Saves results/02_two_proportion.csv and results/02_lift_ci.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hillstrom.csv"
OUT = ROOT / "results"

CONTROL = "No E-Mail"
TREATMENTS = ["Mens E-Mail", "Womens E-Mail"]
OUTCOMES = {"conversion": "Bought something", "visit": "Visited the website"}
ALPHA = 0.05


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> dict[str, float]:
    """Group 1 minus group 2. Pooled z-test for p-value; unpooled (Wald) 95% interval."""
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    # Under "the email does nothing" both groups share one rate, so pool them for the p-value.
    se_null = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_null
    p_value = 2 * stats.norm.sf(abs(z))
    # For the interval we no longer assume no effect, so each group keeps its own rate.
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    half = stats.norm.ppf(1 - ALPHA / 2) * se
    return {"diff": p1 - p2, "ci_low": p1 - p2 - half, "ci_high": p1 - p2 + half,
            "z": z, "p_value": p_value}


def check_against_libraries(mine: dict[str, float], x1: int, n1: int, x2: int, n2: int) -> None:
    """Print mine next to statsmodels and SciPy. Stop if they disagree."""
    z_sm, p_sm = proportions_ztest([x1, x2], [n1, n2])
    low_sm, high_sm = confint_proportions_2indep(x1, n1, x2, n2, method="wald", compare="diff")
    chi2, p_sp, _, _ = stats.chi2_contingency([[x1, n1 - x1], [x2, n2 - x2]], correction=False)
    print(f"    p-value  mine {mine['p_value']:.6g} | statsmodels {p_sm:.6g} | scipy chi2 {p_sp:.6g}")
    print(f"    z        mine {mine['z']:.6f} | statsmodels {z_sm:.6f} | scipy sqrt(chi2) {np.sqrt(chi2):.6f}")
    print(f"    95% CI   mine [{mine['ci_low']:.6f}, {mine['ci_high']:.6f}] | "
          f"statsmodels [{low_sm:.6f}, {high_sm:.6f}]")
    assert np.isclose(mine["p_value"], p_sm, rtol=1e-8) and np.isclose(mine["p_value"], p_sp, rtol=1e-6)
    assert np.isclose(mine["z"], z_sm, rtol=1e-8)
    assert np.isclose(mine["ci_low"], low_sm, rtol=1e-8)
    assert np.isclose(mine["ci_high"], high_sm, rtol=1e-8)


def main() -> None:
    df = pd.read_csv(DATA)
    control = df[df["segment"] == CONTROL]

    rows = []
    for outcome, label in OUTCOMES.items():
        for group in TREATMENTS:
            treated = df[df["segment"] == group]
            x1, n1 = int(treated[outcome].sum()), len(treated)
            x2, n2 = int(control[outcome].sum()), len(control)
            res = two_proportion_test(x1, n1, x2, n2)
            print(f"\n{label}: {group} vs {CONTROL}  ({x1}/{n1} vs {x2}/{n2})")
            check_against_libraries(res, x1, n1, x2, n2)
            rows.append({
                "outcome": outcome, "group": group, "control": CONTROL,
                "n_group": n1, "n_control": n2,
                "rate_group": x1 / n1, "rate_control": x2 / n2,
                "diff_points": res["diff"] * 100,
                "ci_low_points": res["ci_low"] * 100, "ci_high_points": res["ci_high"] * 100,
                "relative_lift": (x1 / n1) / (x2 / n2) - 1,
                "z": res["z"], "p_value": res["p_value"], "alpha": ALPHA,
                "method": "pooled z-test; Wald 95% CI",
            })

    table = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "02_two_proportion.csv", index=False)
    print("\nSummary (differences in percentage points):")
    show = table[["outcome", "group", "diff_points", "ci_low_points", "ci_high_points",
                  "relative_lift", "p_value"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, (outcome, label) in zip(axes, OUTCOMES.items()):
        sub = table[table["outcome"] == outcome].reset_index(drop=True)
        ypos = np.arange(len(sub))
        ax.errorbar(sub["diff_points"], ypos,
                    xerr=[sub["diff_points"] - sub["ci_low_points"],
                          sub["ci_high_points"] - sub["diff_points"]],
                    fmt="o", capsize=4)
        ax.axvline(0, color="grey", linestyle="--")
        ax.set_yticks(ypos, sub["group"])
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_title(label)
        ax.set_xlabel("lift vs no email (percentage points, 95% CI)")
    fig.tight_layout()
    fig.savefig(OUT / "02_lift_ci.png", dpi=130)


if __name__ == "__main__":
    main()
