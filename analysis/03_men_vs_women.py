"""Step 3c: is the men's email really better than the women's? (purchases)

All three pairwise comparisons (men's vs none, women's vs none, men's vs women's) with the
two-proportion z-test, Fisher's exact test and Wald and Newcombe intervals, plus Holm-adjusted
p-values because three comparisons mean three chances for a false alarm. Holm and Newcombe are
written by hand and checked against statsmodels. Saves results/03_pairwise_purchases.csv and
results/03_pairwise_lifts.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hillstrom.csv"
OUT = ROOT / "results"

ALPHA = 0.05
# (group, comparison group): the first minus the second.
PAIRS = [("Mens E-Mail", "No E-Mail"), ("Womens E-Mail", "No E-Mail"),
         ("Mens E-Mail", "Womens E-Mail")]


def z_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test: returns (z, two-sided p)."""
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (x1 / n1 - x2 / n2) / se
    return z, 2 * stats.norm.sf(abs(z))


def wald(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    p1, p2 = x1 / n1, x2 / n2
    half = stats.norm.ppf(1 - ALPHA / 2) * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p1 - p2 - half, p1 - p2 + half


def wilson(x: int, n: int) -> tuple[float, float]:
    z = stats.norm.ppf(1 - ALPHA / 2)
    p = x / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def newcombe(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment: smallest p times m, next times m-1, ...; never decreasing."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    df = pd.read_csv(DATA)
    counts = df.groupby("segment")["conversion"].agg(["sum", "size"])

    # Overall question first: do the three groups differ at all?
    table = [[counts.loc[g, "sum"], counts.loc[g, "size"] - counts.loc[g, "sum"]]
             for g in ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]]
    chi2, p_overall, dof, _ = stats.chi2_contingency(table)
    print(f"Overall (3 groups x bought/not): chi-square {chi2:.2f}, {dof} degrees of freedom, "
          f"p = {p_overall:.3g}")

    rows = []
    for g1, g2 in PAIRS:
        x1, n1 = int(counts.loc[g1, "sum"]), int(counts.loc[g1, "size"])
        x2, n2 = int(counts.loc[g2, "sum"]), int(counts.loc[g2, "size"])
        z, p = z_test(x1, n1, x2, n2)
        w_lo, w_hi = wald(x1, n1, x2, n2)
        n_lo, n_hi = newcombe(x1, n1, x2, n2)
        _, p_fisher = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])

        z_sm, p_sm = proportions_ztest([x1, x2], [n1, n2])
        sm_lo, sm_hi = confint_proportions_2indep(x1, n1, x2, n2, method="newcomb", compare="diff")
        assert np.isclose(z, z_sm, rtol=1e-8) and np.isclose(p, p_sm, rtol=1e-8)
        assert np.isclose(n_lo, sm_lo, rtol=1e-8) and np.isclose(n_hi, sm_hi, rtol=1e-8)

        print(f"\n{g1} vs {g2}: {x1}/{n1} ({x1 / n1:.3%}) vs {x2}/{n2} ({x2 / n2:.3%})")
        print(f"  z     mine {z:.6f} | statsmodels {z_sm:.6f}")
        print(f"  p     mine {p:.6g} | statsmodels {p_sm:.6g} | Fisher exact {p_fisher:.6g}")
        print(f"  Newcombe mine [{n_lo:.6f}, {n_hi:.6f}] | statsmodels [{sm_lo:.6f}, {sm_hi:.6f}]")
        rows.append({
            "group": g1, "versus": g2, "n_group": n1, "n_versus": n2,
            "rate_group": x1 / n1, "rate_versus": x2 / n2,
            "diff_points": (x1 / n1 - x2 / n2) * 100,
            "wald_low_points": w_lo * 100, "wald_high_points": w_hi * 100,
            "newcombe_low_points": n_lo * 100, "newcombe_high_points": n_hi * 100,
            "z": z, "p_value": p, "fisher_p_value": p_fisher,
        })

    raw = [r["p_value"] for r in rows]
    mine = holm(raw)
    sm_holm = multipletests(raw, alpha=ALPHA, method="holm")[1]
    print("\nHolm-adjusted p-values (3 comparisons):")
    for r, a, b in zip(rows, mine, sm_holm, strict=True):
        print(f"  {r['group']} vs {r['versus']}: raw {r['p_value']:.3g} | Holm mine {a:.3g} | "
              f"statsmodels {b:.3g}")
        r["holm_p_value"] = a
        r["significant_after_holm"] = bool(a < ALPHA)
    assert np.allclose(mine, sm_holm, rtol=1e-8)

    result = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / "03_pairwise_purchases.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3))
    for i, r in result.iterrows():
        ax.errorbar(r["diff_points"], i,
                    xerr=[[r["diff_points"] - r["newcombe_low_points"]],
                          [r["newcombe_high_points"] - r["diff_points"]]], fmt="o", capsize=4)
        ax.text(r["newcombe_high_points"] + 0.02, i, f"Holm p = {r['holm_p_value']:.2g}",
                va="center", fontsize=8)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(range(len(result)), [f"{g} vs {v}" for g, v in zip(result["group"], result["versus"],
                                                                      strict=True)])
    ax.set_xlim(-0.1, 1.2)
    ax.set_xlabel("purchase rate difference (percentage points, 95% Newcombe interval)")
    fig.tight_layout()
    fig.savefig(OUT / "03_pairwise_lifts.png", dpi=130)


if __name__ == "__main__":
    main()
