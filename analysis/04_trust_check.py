"""Step 4: the trust check. Does the test cry wolf as often as it promises?

A/A experiments: two groups with the SAME true buying rate, so any "difference" is a false alarm.
For each scenario, 10,000 fake experiments. We count how often the z-test and Fisher's exact test
declare a difference at 5%, and how often the Wald and Newcombe 95% intervals contain the truth
(a difference of 0). Also one real-data check: random halves of the real no-email group.
Saves results/04_trust_check.csv and results/04_trust_check.png.
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

SEED = 20261112
REPS = 10_000
ALPHA = 0.05
TRUE_RATE = 0.0057  # about the no-email buying rate
SIZES = [250, 500, 1_000, 2_000, 5_000, 10_000, 21_300]
# Same size as the small case, but a common event: shows that few buyers (not few people) is the issue.
CONTRAST = (500, 0.10)
Z = stats.norm.ppf(1 - ALPHA / 2)


def z_test_p(x1: np.ndarray, x2: np.ndarray, n: int) -> np.ndarray:
    """Pooled two-proportion z-test p-values for equal group sizes; p = 1 if nobody bought."""
    pooled = (x1 + x2) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (x1 / n - x2 / n) / se, 0.0)
    return 2 * stats.norm.sf(np.abs(z))


def wald_ci(x1: np.ndarray, x2: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    p1, p2 = x1 / n, x2 / n
    half = Z * np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    return p1 - p2 - half, p1 - p2 + half


def wilson(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    p = x / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return centre - half, centre + half


def newcombe_ci(x1: np.ndarray, x2: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    p1, p2 = x1 / n, x2 / n
    l1, u1 = wilson(x1, n)
    l2, u2 = wilson(x2, n)
    return (p1 - p2 - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            p1 - p2 + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def fisher_p(x1: np.ndarray, x2: np.ndarray, n: int) -> np.ndarray:
    """Fisher's exact p-values; computed once per distinct (x1, x2) pair, since many repeat."""
    pairs, inverse = np.unique(np.stack([x1, x2], axis=1), axis=0, return_inverse=True)
    p = np.array([stats.fisher_exact([[a, n - a], [b, n - b]])[1] for a, b in pairs])
    return p[inverse.reshape(-1)]


def check_hand_written_against_statsmodels(x1: np.ndarray, x2: np.ndarray, n: int) -> None:
    """Our vectorised test and intervals must match statsmodels on a sample of the fake experiments."""
    p_mine, (nl, nh) = z_test_p(x1, x2, n), newcombe_ci(x1, x2, n)
    wl, wh = wald_ci(x1, x2, n)
    checked = 0
    for i in range(len(x1)):
        if x1[i] + x2[i] == 0:
            continue
        _, p_sm = proportions_ztest([x1[i], x2[i]], [n, n])
        nl_sm, nh_sm = confint_proportions_2indep(x1[i], n, x2[i], n, method="newcomb", compare="diff")
        wl_sm, wh_sm = confint_proportions_2indep(x1[i], n, x2[i], n, method="wald", compare="diff")
        assert np.isclose(p_mine[i], p_sm, rtol=1e-8)
        assert np.isclose(nl[i], nl_sm, rtol=1e-8) and np.isclose(nh[i], nh_sm, rtol=1e-8)
        assert np.isclose(wl[i], wl_sm, rtol=1e-8) and np.isclose(wh[i], wh_sm, rtol=1e-8)
        checked += 1
        if checked == 50:
            break
    print(f"Hand-written z-test, Wald and Newcombe match statsmodels on {checked} fake experiments.")


def reading(observed: float, nominal: float, band: tuple[float, float], kind: str) -> str:
    if band[0] <= observed <= band[1]:
        return "calibrated"
    if kind == "test":
        return "too many false alarms" if observed > band[1] else "conservative (valid)"
    return "under-covers" if observed < band[0] else "over-covers (valid)"


def summarize(label: str, n: int, rate: float, x1: np.ndarray, x2: np.ndarray, source: str,
              seed: int | None, independent: bool = True) -> list[dict]:
    reps = len(x1)
    rows = []
    results = {
        ("z-test", "test"): z_test_p(x1, x2, n) < ALPHA,
        ("Fisher exact", "test"): fisher_p(x1, x2, n) < ALPHA,
    }
    for name, (lo, hi) in {"Wald interval": wald_ci(x1, x2, n),
                           "Newcombe interval": newcombe_ci(x1, x2, n)}.items():
        results[(name, "interval")] = (lo <= 0) & (0 <= hi)  # the truth in an A/A is a difference of 0
    for (method, kind), flags in results.items():
        nominal = ALPHA if kind == "test" else 1 - ALPHA
        half = 1.96 * np.sqrt(nominal * (1 - nominal) / reps)
        observed = float(flags.mean())
        band = (nominal - half, nominal + half)
        rows.append({
            "scenario": label, "source": source, "n_per_group": n, "true_rate": rate,
            "expected_buyers_per_group": n * rate, "method": method,
            "measures": "false-alarm rate" if kind == "test" else "interval coverage",
            "nominal": nominal, "observed": observed, "band_low": band[0], "band_high": band[1],
            "reading": (reading(observed, nominal, band, kind) if independent
                        else "indicative only (splits reuse the same people)"),
            "reps": reps, "seed": seed,
            "share_nobody_bought": float(((x1 + x2) == 0).mean()),
        })
    return rows


def main() -> None:
    root_seq = np.random.SeedSequence(SEED)
    scenarios = [(n, TRUE_RATE) for n in SIZES] + [CONTRAST]
    children = root_seq.spawn(len(scenarios))
    rows = []
    for (n, rate), child in zip(scenarios, children, strict=True):
        rng = np.random.default_rng(child)
        x1, x2 = rng.binomial(n, rate, size=REPS), rng.binomial(n, rate, size=REPS)
        if (n, rate) == (21_300, TRUE_RATE):
            check_hand_written_against_statsmodels(x1, x2, n)
        rows += summarize(f"n={n:,} per group, rate {rate:.2%}", n, rate, x1, x2, "simulated", SEED)

    # Real-data check: split the real no-email group into two random halves, many times.
    control = pd.read_csv(DATA).query("segment == 'No E-Mail'")["conversion"].to_numpy(dtype=np.int8)
    half, buyers = len(control) // 2, int(control.sum())
    rng = np.random.default_rng(root_seq.spawn(1)[0])
    x1 = np.array([int(rng.permutation(control)[:half].sum()) for _ in range(REPS)])
    # x1 comes from one half; the other half holds the remaining buyers (the full group has `buyers`).
    x2 = buyers - x1
    rows += summarize(f"real no-email group split in halves ({half:,} each)", half, buyers / len(control),
                      x1, x2, "real data (reused people)", SEED, independent=False)

    table = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "04_trust_check.csv", index=False)

    show = table.assign(observed=lambda d: d["observed"] * 100, nominal=lambda d: d["nominal"] * 100)
    print("\nObserved rates (%), simulated scenarios and the real-data split:")
    print(show[["scenario", "method", "measures", "nominal", "observed", "reading"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nShare of fake experiments where nobody bought in either group (small-n problem):")
    print(table.drop_duplicates("scenario")[["scenario", "share_nobody_bought"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    sim = table[(table["source"] == "simulated") & (table["true_rate"] == TRUE_RATE)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, measures, nominal in zip(axes, ["false-alarm rate", "interval coverage"], [ALPHA, 1 - ALPHA],
                                     strict=True):
        for method, sub in sim[sim["measures"] == measures].groupby("method"):
            ax.plot(sub["n_per_group"], sub["observed"] * 100, "o-", label=method)
        half_band = 1.96 * np.sqrt(nominal * (1 - nominal) / REPS) * 100
        ax.axhspan(nominal * 100 - half_band, nominal * 100 + half_band, color="green", alpha=0.15,
                   label="calibrated band")
        ax.axhline(nominal * 100, color="grey", linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("people per group (buying rate 0.57%)")
        ax.set_ylabel(f"{measures} (%)")
        ax.set_title("Cries wolf too often?" if measures == "false-alarm rate" else "Contains the truth?")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "04_trust_check.png", dpi=130)


if __name__ == "__main__":
    main()
