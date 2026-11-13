"""Step 3b: do other tests agree with the two-proportion z-test? (purchases only)

For each email versus no email: Fisher's exact test, a permutation test, the Newcombe and Wald
intervals side by side, and logistic regression. Newcombe is written by hand (from Wilson
intervals) and checked against statsmodels. Saves results/02b_robustness.csv and
results/02b_intervals.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hillstrom.csv"
OUT = ROOT / "results"

CONTROL = "No E-Mail"
TREATMENTS = ["Mens E-Mail", "Womens E-Mail"]
ALPHA = 0.05
SEED = 20261102
N_SHUFFLES = 10_000


def wilson(x: int, n: int) -> tuple[float, float]:
    """Wilson score interval for one proportion."""
    z = stats.norm.ppf(1 - ALPHA / 2)
    p = x / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def newcombe(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Newcombe's hybrid-score interval for p1 - p2: combine each group's Wilson interval."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def permutation_p(x1: int, n1: int, x2: int, n2: int, rng: np.random.Generator) -> float:
    """Shuffle the 'got the email' labels and count how often chance beats the real gap."""
    outcomes = np.concatenate([np.ones(x1 + x2, dtype=np.int8), np.zeros(n1 + n2 - x1 - x2, dtype=np.int8)])
    total = x1 + x2
    observed = abs(x1 / n1 - x2 / n2)
    hits = 0
    for _ in range(N_SHUFFLES):
        first = int(rng.permutation(outcomes)[:n1].sum())
        if abs(first / n1 - (total - first) / n2) >= observed - 1e-15:
            hits += 1
    return (hits + 1) / (N_SHUFFLES + 1)


def main() -> None:
    df = pd.read_csv(DATA)
    rng = np.random.default_rng(SEED)
    control = df[df["segment"] == CONTROL]
    rows = []

    for group in TREATMENTS:
        treated = df[df["segment"] == group]
        x1, n1 = int(treated["conversion"].sum()), len(treated)
        x2, n2 = int(control["conversion"].sum()), len(control)
        diff = (x1 / n1 - x2 / n2) * 100
        print(f"\n{group} vs {CONTROL} (purchases): {x1}/{n1} vs {x2}/{n2}, lift {diff:.3f} points")

        # Newcombe by hand, checked against the library, and Wilson per group.
        mine_lo, mine_hi = newcombe(x1, n1, x2, n2)
        sm_lo, sm_hi = confint_proportions_2indep(x1, n1, x2, n2, method="newcomb", compare="diff")
        w_sm = proportion_confint(x1, n1, alpha=ALPHA, method="wilson")
        print(f"  Newcombe  mine [{mine_lo:.6f}, {mine_hi:.6f}] | statsmodels [{sm_lo:.6f}, {sm_hi:.6f}]")
        print(f"  Wilson (one group) mine {tuple(round(float(v), 6) for v in wilson(x1, n1))} | "
              f"statsmodels {tuple(round(float(v), 6) for v in w_sm)}")
        assert np.isclose(mine_lo, sm_lo, rtol=1e-8) and np.isclose(mine_hi, sm_hi, rtol=1e-8)
        assert np.allclose(wilson(x1, n1), w_sm, rtol=1e-8)
        w_lo, w_hi = confint_proportions_2indep(x1, n1, x2, n2, method="wald", compare="diff")
        print(f"  Wald      [{w_lo:.6f}, {w_hi:.6f}]  (the step 2 interval)")

        def add(method: str, p: float | None, lo: float | None = None, hi: float | None = None,
                note: str = "", group: str = group, diff: float = diff) -> None:
            rows.append({
                "group": group, "method": method, "lift_points": diff,
                "ci_low_points": None if lo is None else lo * 100,
                "ci_high_points": None if hi is None else hi * 100,
                "p_value": p, "significant_at_5pct": None if p is None else p < ALPHA,
                "note": note, "seed": SEED if "permutation" in method else None,
            })

        z_p = 2 * stats.norm.sf(abs((x1 / n1 - x2 / n2) / np.sqrt(
            ((x1 + x2) / (n1 + n2)) * (1 - (x1 + x2) / (n1 + n2)) * (1 / n1 + 1 / n2))))
        add("two-proportion z (step 2)", z_p, w_lo, w_hi, "Wald interval")
        add("Newcombe interval", None, mine_lo, mine_hi, "interval only; same z-test p-value")

        odds, p_fisher = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])
        add("Fisher exact", p_fisher, note=f"odds ratio {odds:.3f}")

        p_perm = permutation_p(x1, n1, x2, n2, rng)
        add(f"permutation ({N_SHUFFLES:,} shuffles)", p_perm,
            note=f"smallest possible p is {1 / (N_SHUFFLES + 1):.1e}")

        sub = pd.concat([treated, control])
        sub["treated"] = (sub["segment"] == group).astype(int)
        fit = smf.logit("conversion ~ treated", data=sub).fit(disp=False)
        add("logistic regression (Wald)", float(fit.pvalues["treated"]),
            note=f"odds ratio {np.exp(fit.params['treated']):.3f}")
        add("logistic regression (likelihood ratio)", float(fit.llr_pvalue))

    table = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "02b_robustness.csv", index=False)

    pvals = table.dropna(subset=["p_value"])
    print("\np-values by method:")
    print(pvals.pivot(index="method", columns="group", values="p_value").to_string(
        float_format=lambda v: f"{v:.3g}"))
    significant = pvals["significant_at_5pct"].astype(bool)
    print(f"\nMethods run: {len(pvals)}. Significant at 5%: {int(significant.sum())}. "
          f"Flipped to not significant: {int((~significant).sum())}.")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ypos, labels = 0, []
    for group in TREATMENTS:
        sub = table[(table["group"] == group) & table["ci_low_points"].notna()]
        for _, r in sub.iterrows():
            ax.errorbar(r["lift_points"], ypos, xerr=[[r["lift_points"] - r["ci_low_points"]],
                                                       [r["ci_high_points"] - r["lift_points"]]],
                        fmt="o", capsize=4)
            labels.append(f"{group}: {'Wald' if 'Wald' in r['note'] else 'Newcombe'}")
            ypos += 1
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("purchase lift vs no email (percentage points, 95% CI)")
    fig.tight_layout()
    fig.savefig(OUT / "02b_intervals.png", dpi=130)


if __name__ == "__main__":
    main()
