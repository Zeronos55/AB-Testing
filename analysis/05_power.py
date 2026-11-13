"""Step 5: power and sample size. How many people do we need to catch a real lift?

Four dials: baseline rate, the smallest lift worth detecting, false-alarm rate (5%) and power
(80%); they fix the fifth, people per group. The formula is written by hand, checked against
statsmodels, and then checked by simulation (fake experiments with a TRUE lift: how often does the
z-test catch it?). Also shows the winner's curse: significant results from underpowered tests
exaggerate the lift. Saves results/05_sample_size.csv, results/05_power_checks.csv,
results/05_power.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.proportion import (
    power_proportions_2indep,
    samplesize_proportions_2indep_onetail,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hillstrom.csv"
OUT = ROOT / "results"

SEED = 20261113
REPS = 10_000
ALPHA = 0.05
POWERS = [0.80, 0.90]
LIFTS = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.0068]  # absolute lifts: 0.1 ... 0.68 points
WORTH = 0.003  # the smallest lift worth detecting, chosen by the user (0.3 points)
UNDERPOWERED_N = [2_000, 5_000, 10_000, 21_300]
OUR_N = 21_300
Z_A = stats.norm.ppf(1 - ALPHA / 2)


def n_per_group(base: float, lift: float, power: float) -> float:
    """People per group for a two-sided two-proportion z-test (pooled variance under no effect)."""
    p1, p2 = base + lift, base
    pbar = (p1 + p2) / 2
    z_b = stats.norm.ppf(power)
    num = Z_A * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return float(num**2 / lift**2)


def power_at(base: float, lift: float, n: float) -> float:
    """Chance the two-sided z-test declares a difference when the true lift is `lift`."""
    p1, p2 = base + lift, base
    pbar = (p1 + p2) / 2
    se_null = np.sqrt(2 * pbar * (1 - pbar) / n)
    se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    return float(stats.norm.sf((Z_A * se_null - lift) / se_alt)
                 + stats.norm.cdf((-Z_A * se_null - lift) / se_alt))


def min_detectable_lift(base: float, n: float, power: float) -> float:
    return float(optimize.brentq(lambda d: power_at(base, d, n) - power, 1e-6, 0.05))


def simulate(base: float, lift: float, n: int, rng: np.random.Generator) -> dict[str, float]:
    """Fake experiments with a true lift: share the z-test catches it, and the exaggeration."""
    x1 = rng.binomial(n, base + lift, size=REPS)
    x2 = rng.binomial(n, base, size=REPS)
    pooled = (x1 + x2) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (x1 - x2) / n / np.where(se > 0, se, 1), 0.0)
    significant = (2 * stats.norm.sf(np.abs(z)) < ALPHA)
    estimate = (x1 - x2) / n
    right_way = significant & (estimate > 0)
    power = float(significant.mean())
    half = 1.96 * np.sqrt(power * (1 - power) / REPS)
    return {
        "power_simulated": power, "sim_low": power - half, "sim_high": power + half,
        "mean_lift_if_significant": float(estimate[right_way].mean()) if right_way.any() else np.nan,
    }


def main() -> None:
    control = pd.read_csv(DATA).query("segment == 'No E-Mail'")["conversion"]
    base = float(control.mean())
    print(f"Baseline (no-email buying rate): {base:.4%}; alpha {ALPHA}; seeds from {SEED}\n")
    children = np.random.SeedSequence(SEED).spawn(len(LIFTS) + len(UNDERPOWERED_N))

    # 1. Sample sizes, checked against statsmodels.
    rows = []
    for power in POWERS:
        for lift in LIFTS:
            mine = n_per_group(base, lift, power)
            sm = float(samplesize_proportions_2indep_onetail(lift, base, power, alpha=ALPHA,
                                                              alternative="two-sided"))
            rows.append({"baseline_rate": base, "lift_points": lift * 100, "power_target": power,
                         "alpha": ALPHA, "n_per_group_formula": mine, "n_per_group_statsmodels": sm,
                         "n_per_group_rounded_up": int(np.ceil(mine)),
                         "total_people_both_groups": 2 * int(np.ceil(mine))})
    sizes = pd.DataFrame(rows)
    print("Sample size per group (formula | statsmodels):")
    print(sizes.assign(f=lambda d: d["n_per_group_formula"].round(0), s=lambda d: d["n_per_group_statsmodels"].round(0))
          [["power_target", "lift_points", "f", "s"]].to_string(index=False))
    rel = (sizes["n_per_group_formula"] / sizes["n_per_group_statsmodels"] - 1).abs().max()
    print(f"Largest relative difference formula vs statsmodels: {rel:.2e}")
    assert rel < 1e-6

    # 2. Power checks: at the formula's n for each lift, and at underpowered n for the 0.3-point lift.
    checks = []
    cases = [(lift, int(np.ceil(n_per_group(base, lift, 0.80))), "n for 80% power") for lift in LIFTS]
    cases += [(WORTH, n, "fixed n") for n in UNDERPOWERED_N]
    for (lift, n, why), child in zip(cases, children, strict=True):
        formula = power_at(base, lift, n)
        sm_power = float(power_proportions_2indep(lift, base, n, alpha=ALPHA, return_results=False))
        assert abs(formula - sm_power) < 1e-6, (lift, n, formula, sm_power)
        sim = simulate(base, lift, n, np.random.default_rng(child))
        checks.append({"case": why, "baseline_rate": base, "lift_points": lift * 100, "n_per_group": n,
                       "power_formula": formula, "power_statsmodels": sm_power, **sim,
                       "exaggeration_ratio": sim["mean_lift_if_significant"] / lift, "seed": SEED})
    table = pd.DataFrame(checks)
    print("\nPower: formula | statsmodels | simulated (10,000 fake experiments, true lift present):")
    print(table[["case", "lift_points", "n_per_group", "power_formula", "power_statsmodels",
                 "power_simulated", "exaggeration_ratio"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    # 3. What could our real experiment (about 21,300 per group) have detected?
    print(f"\nWith {OUR_N:,} per group:")
    for power in POWERS:
        print(f"  smallest lift detectable with {power:.0%} power: "
              f"{min_detectable_lift(base, OUR_N, power) * 100:.3f} points")
    print("  power if the TRUE lift were:",
          {f"{d * 100:.2f}pts": round(power_at(base, d, OUR_N), 3) for d in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.005]})

    OUT.mkdir(exist_ok=True)
    sizes.to_csv(OUT / "05_sample_size.csv", index=False)
    table.to_csv(OUT / "05_power_checks.csv", index=False)

    # Chart: power versus people, and people versus lift.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    grid = np.linspace(500, 60_000, 300)
    for lift in [0.002, 0.003, 0.005]:
        axes[0].plot(grid, [power_at(base, lift, n) * 100 for n in grid], label=f"true lift {lift * 100:.1f} points")
    sim03 = table[(table["case"] == "fixed n")]
    axes[0].plot(sim03["n_per_group"], sim03["power_simulated"] * 100, "ko", label="simulated (0.3 pts)")
    axes[0].axhline(80, color="grey", linestyle="--")
    axes[0].axvline(OUR_N, color="red", linestyle=":", label="our experiment")
    axes[0].set(xlabel="people per group", ylabel="power (%)", title="Chance of catching a real lift")
    axes[0].legend(fontsize=8)
    lifts = np.linspace(0.0007, 0.008, 100)
    axes[1].plot(lifts * 100, [n_per_group(base, d, 0.8) for d in lifts])
    axes[1].plot(WORTH * 100, n_per_group(base, WORTH, 0.8), "ro")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="lift to detect (percentage points)", ylabel="people per group (log scale)",
                title="Halve the lift, about quadruple the people (80% power)")
    fig.tight_layout()
    fig.savefig(OUT / "05_power.png", dpi=130)


if __name__ == "__main__":
    main()
