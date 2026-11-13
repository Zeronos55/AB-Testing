"""Step 6: who does the email work for? (pre-registered in notes/2026-11-13_step6_customer_type_plan.md)

Customer types come from past-purchase flags (not gender): men's-only, women's-only, both (context).
For each email: the lift (email rate minus no-email rate) within each type, then the difference of
lifts between the two pre-registered types, with Holm correction across the two tests. Checked
against a statsmodels interaction model. Also a trust check (equal true lifts: how often does the
test claim a gap?) and the smallest gap this data could reliably detect.
Saves results/06_lifts_by_type.csv, results/06_difference_of_lifts.csv, results/06_trust_and_power.csv,
results/06_customer_type.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hillstrom.csv"
OUT = ROOT / "results"

SEED = 20261113
REPS = 10_000
ALPHA = 0.05
Z = stats.norm.ppf(1 - ALPHA / 2)
CONTROL = "No E-Mail"
TYPES = {"Men's-only": (1, 0), "Women's-only": (0, 1), "Both": (1, 1)}
# The two pre-registered tests: (email, type A, type B), difference = lift in A minus lift in B.
TESTS = [("Mens E-Mail", "Men's-only", "Women's-only"),
         ("Womens E-Mail", "Women's-only", "Men's-only")]


def wilson(x: int, n: int) -> tuple[float, float]:
    p = x / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return centre - half, centre + half


def newcombe(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def z_test_p(x1: int, n1: int, x2: int, n2: int) -> float:
    pooled = (x1 + x2) / (n1 + n2)
    z = (x1 / n1 - x2 / n2) / np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return float(2 * stats.norm.sf(abs(z)))


def holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p_values)):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def diff_of_lifts(cells: dict[str, tuple[int, int]]) -> tuple[float, float]:
    """cells: email_A, control_A, email_B, control_B as (buyers, people). Returns (difference, SE)."""
    p = {k: x / n for k, (x, n) in cells.items()}
    d = (p["email_A"] - p["control_A"]) - (p["email_B"] - p["control_B"])
    var = sum(p[k] * (1 - p[k]) / cells[k][1] for k in cells)
    return float(d), float(np.sqrt(var))


def main() -> None:
    df = pd.read_csv(DATA)

    def cell(email: str, kind: str) -> tuple[int, int]:
        mens, womens = TYPES[kind]
        sub = df[(df["segment"] == email) & (df["mens"] == mens) & (df["womens"] == womens)]
        return int(sub["conversion"].sum()), len(sub)

    # 1. Lift within each customer type.
    lift_rows = []
    for email in ["Mens E-Mail", "Womens E-Mail"]:
        for kind in TYPES:
            xe, ne = cell(email, kind)
            xc, nc = cell(CONTROL, kind)
            lo, hi = newcombe(xe, ne, xc, nc)
            lift_rows.append({
                "email": email, "customer_type": kind, "n_email": ne, "n_control": nc,
                "buyers_email": xe, "buyers_control": xc, "rate_email": xe / ne, "rate_control": xc / nc,
                "lift_points": (xe / ne - xc / nc) * 100,
                "newcombe_low_points": lo * 100, "newcombe_high_points": hi * 100,
                "p_value": z_test_p(xe, ne, xc, nc),
                "role": "context only (not tested)" if kind == "Both" else "pre-registered type",
            })
    lifts = pd.DataFrame(lift_rows)
    print("Lift in purchases within each customer type (vs no email, same type):")
    print(lifts[["email", "customer_type", "n_email", "n_control", "rate_email", "rate_control",
                 "lift_points", "newcombe_low_points", "newcombe_high_points"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # 2. The two pre-registered differences of lifts, checked against statsmodels.
    diff_rows = []
    for email, a, b in TESTS:
        cells = {"email_A": cell(email, a), "control_A": cell(CONTROL, a),
                 "email_B": cell(email, b), "control_B": cell(CONTROL, b)}
        d, se = diff_of_lifts(cells)
        z = d / se
        p = float(2 * stats.norm.sf(abs(z)))

        parts = []
        for name, (mens, womens) in (("A", TYPES[a]), ("B", TYPES[b])):
            for who in (email, CONTROL):
                sub = df[(df["segment"] == who) & (df["mens"] == mens) & (df["womens"] == womens)]
                parts.append(sub.assign(email=int(who == email), typeA=int(name == "A")))
        reg = pd.concat(parts)
        lin = smf.ols("conversion ~ email * typeA", data=reg).fit(cov_type="HC0")
        logit = smf.logit("conversion ~ email * typeA", data=reg).fit(disp=False)
        sm_d, sm_se = float(lin.params["email:typeA"]), float(lin.bse["email:typeA"])
        print(f"\n{email}: lift in {a} minus lift in {b}")
        print(f"  difference mine {d * 100:.5f} | statsmodels {sm_d * 100:.5f} points")
        print(f"  std error  mine {se * 100:.5f} | statsmodels HC0 {sm_se * 100:.5f} points")
        assert np.isclose(d, sm_d, rtol=1e-8) and np.isclose(se, sm_se, rtol=1e-8)
        diff_rows.append({
            "email": email, "contrast": f"lift in {a} minus lift in {b}",
            "diff_points": d * 100, "ci_low_points": (d - Z * se) * 100, "ci_high_points": (d + Z * se) * 100,
            "z": z, "p_value": p, "logistic_interaction_p": float(logit.pvalues["email:typeA"]),
            "statsmodels_diff_points": sm_d * 100, "statsmodels_se_points": sm_se * 100,
        })
    raw = [r["p_value"] for r in diff_rows]
    mine = holm(raw)
    sm_holm = multipletests(raw, alpha=ALPHA, method="holm")[1]
    assert np.allclose(mine, sm_holm, rtol=1e-8)
    for r, a in zip(diff_rows, mine, strict=True):
        r["holm_p_value"], r["significant_after_holm"] = a, bool(a < ALPHA)
    diffs = pd.DataFrame(diff_rows)
    print("\nPre-registered tests (Holm across 2):")
    print(diffs[["email", "contrast", "diff_points", "ci_low_points", "ci_high_points", "p_value",
                 "holm_p_value", "logistic_interaction_p"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # 3. Trust check and power for this method, at the real cell sizes.
    rng_children = np.random.SeedSequence(SEED).spawn(2 * len(TESTS))
    tp_rows = []
    for k, (email, a, b) in enumerate(TESTS):
        cells = {"email_A": cell(email, a), "control_A": cell(CONTROL, a),
                 "email_B": cell(email, b), "control_B": cell(CONTROL, b)}
        n = {key: v[1] for key, v in cells.items()}
        rc_a, rc_b = cells["control_A"][0] / n["control_A"], cells["control_B"][0] / n["control_B"]
        overall_lift = float(lifts.query("email == @email and customer_type in [@a, @b]")["lift_points"].mean()) / 100

        def simulate(gap: float, child: np.random.SeedSequence, rc_a: float = rc_a, rc_b: float = rc_b,
                     n: dict = n, lift: float = overall_lift) -> float:
            rng = np.random.default_rng(child)
            rates = {"control_A": rc_a, "control_B": rc_b, "email_A": rc_a + lift + gap, "email_B": rc_b + lift}
            x = {key: rng.binomial(n[key], rates[key], size=REPS) for key in n}
            p = {key: x[key] / n[key] for key in n}
            d = (p["email_A"] - p["control_A"]) - (p["email_B"] - p["control_B"])
            var = sum(p[key] * (1 - p[key]) / n[key] for key in n)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(var > 0, d / np.sqrt(np.where(var > 0, var, 1)), 0.0)
            return float((2 * stats.norm.sf(np.abs(z)) < ALPHA).mean())

        def se_at(gap: float, rc_a: float = rc_a, rc_b: float = rc_b, n: dict = n, lift: float = overall_lift) -> float:
            rates = {"control_A": rc_a, "control_B": rc_b, "email_A": rc_a + lift + gap, "email_B": rc_b + lift}
            return float(np.sqrt(sum(rates[key] * (1 - rates[key]) / n[key] for key in n)))

        def power_formula(gap: float) -> float:
            s = se_at(gap)
            return float(stats.norm.sf(Z - abs(gap) / s) + stats.norm.cdf(-Z - abs(gap) / s))

        mde = float(optimize.brentq(lambda g: power_formula(g) - 0.80, 1e-5, 0.05))
        false_alarm = simulate(0.0, rng_children[2 * k])
        power_sim = simulate(mde, rng_children[2 * k + 1])
        print(f"\n{email}: {a} vs {b} (cells: {n})")
        print(f"  no real gap -> claims one {false_alarm:.1%} of the time (nominal 5%)")
        print(f"  smallest gap detectable with 80% power: {mde * 100:.3f} points "
              f"(simulated power at that gap {power_sim:.1%})")
        tp_rows.append({"email": email, "contrast": f"{a} vs {b}", **{f"n_{key}": v for key, v in n.items()},
                        "false_alarm_rate_no_gap": false_alarm, "nominal": ALPHA,
                        "mde_points_80pct_power": mde * 100, "power_simulated_at_mde": power_sim,
                        "reps": REPS, "seed": SEED})
    trust = pd.DataFrame(tp_rows)

    OUT.mkdir(exist_ok=True)
    lifts.to_csv(OUT / "06_lifts_by_type.csv", index=False)
    diffs.to_csv(OUT / "06_difference_of_lifts.csv", index=False)
    trust.to_csv(OUT / "06_trust_and_power.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, email in zip(axes, ["Mens E-Mail", "Womens E-Mail"], strict=True):
        sub = lifts[lifts["email"] == email].reset_index(drop=True)
        for i, r in sub.iterrows():
            ax.errorbar(r["lift_points"], i, xerr=[[r["lift_points"] - r["newcombe_low_points"]],
                                                   [r["newcombe_high_points"] - r["lift_points"]]],
                        fmt="o", capsize=4, color="grey" if r["customer_type"] == "Both" else None)
        ax.axvline(0, color="grey", linestyle="--")
        ax.set_yticks(range(len(sub)), sub["customer_type"])
        ax.set_title(f"{email}: lift in purchases by customer type")
        ax.set_xlabel("lift vs no email, same customer type (points, 95% CI)")
    fig.tight_layout()
    fig.savefig(OUT / "06_customer_type.png", dpi=130)


if __name__ == "__main__":
    main()
