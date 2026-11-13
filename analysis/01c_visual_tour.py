"""Step 2c: draw the data. Distributions of the customers and a balance check across the groups.

No outcomes by customer segment here (that comes later, with the hypothesis stated first).
Saves results/01c_distributions.png, results/01c_balance.png and results/01c_balance_smd.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hillstrom.csv"
OUT = ROOT / "results"

CONTROL = "No E-Mail"
TREATMENTS = ["Mens E-Mail", "Womens E-Mail"]
BEFORE = ["recency", "history", "mens", "womens", "newbie", "urban", "suburban", "rural",
          "web", "phone", "multichannel"]
THRESHOLD = 0.1  # common rule of thumb: a standardized difference under 0.1 is negligible


def smd(treated: pd.Series, control: pd.Series) -> float:
    """Standardized mean difference: gap in means divided by the typical spread of the two groups."""
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    return float((treated.mean() - control.mean()) / pooled_sd)


def main() -> None:
    df = pd.read_csv(DATA)
    df = df.assign(
        urban=df["zip_code"].eq("Urban").astype(int), suburban=df["zip_code"].eq("Surburban").astype(int),
        rural=df["zip_code"].eq("Rural").astype(int), web=df["channel"].eq("Web").astype(int),
        phone=df["channel"].eq("Phone").astype(int),
        multichannel=df["channel"].eq("Multichannel").astype(int),
    )
    OUT.mkdir(exist_ok=True)

    # 1. What the customers look like, and what happened after the email.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(df["recency"], bins=np.arange(0.5, 13.5, 1), edgecolor="white")
    axes[0, 0].set(title="Months since last purchase", xlabel="months", ylabel="customers")

    bins = np.logspace(np.log10(df["history"].min()), np.log10(df["history"].max()), 40)
    axes[0, 1].hist(df["history"], bins=bins, edgecolor="white")
    axes[0, 1].set_xscale("log")
    axes[0, 1].axvline(df["history"].median(), color="red", linestyle="--", label="median")
    axes[0, 1].axvline(df["history"].mean(), color="black", linestyle="--", label="mean")
    axes[0, 1].set(title="Past-year spend (log scale)", xlabel="dollars", ylabel="customers")
    axes[0, 1].legend()

    bought = int((df["spend"] > 0).sum())
    axes[1, 0].bar(["spent $0", "spent > $0"], [len(df) - bought, bought])
    for i, v in enumerate([len(df) - bought, bought]):
        axes[1, 0].text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    axes[1, 0].set(title="Spend after the email: mostly zero", ylabel="customers")
    axes[1, 0].set_ylim(0, len(df) * 1.15)

    axes[1, 1].hist(df.loc[df["spend"] > 0, "spend"], bins=30, edgecolor="white")
    axes[1, 1].set(title=f"Spend among the {bought} who spent", xlabel="dollars", ylabel="customers")
    fig.tight_layout()
    fig.savefig(OUT / "01c_distributions.png", dpi=130)

    # 2. Balance: how alike were the groups before the email?
    control = df[df["segment"] == CONTROL]
    rows = [{"before_email_fact": col, "group": g,
             "standardized_difference": smd(df.loc[df["segment"] == g, col], control[col])}
            for col in BEFORE for g in TREATMENTS]
    balance = pd.DataFrame(rows)
    balance.to_csv(OUT / "01c_balance_smd.csv", index=False)
    print("Largest absolute standardized difference:",
          round(balance["standardized_difference"].abs().max(), 4), f"(threshold {THRESHOLD})")
    print(balance.pivot(index="before_email_fact", columns="group", values="standardized_difference")
          .round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, g in zip((-0.15, 0.15), TREATMENTS, strict=True):
        sub = balance[balance["group"] == g]
        ax.plot(sub["standardized_difference"], np.arange(len(sub)) + offset, "o", label=f"{g} vs none")
    ax.axvspan(-THRESHOLD, THRESHOLD, color="green", alpha=0.12, label="negligible (< 0.1)")
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(range(len(BEFORE)), BEFORE)
    ax.set_xlim(-0.2, 0.2)
    ax.set_xlabel("standardized difference vs no-email group")
    ax.set_title("Were the groups alike before the email?")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "01c_balance.png", dpi=130)

    print(f"\nPast-year spend: median ${df['history'].median():.0f}, mean ${df['history'].mean():.0f}")
    top = df["history"].sort_values(ascending=False)
    print(f"Top 20% of customers account for {top.head(len(df) // 5).sum() / top.sum():.0%} of "
          "past-year spend")
    buyers = df.loc[df["spend"] > 0, "spend"].sort_values(ascending=False)
    print(f"Top 20% of buyers account for {buyers.head(len(buyers) // 5).sum() / buyers.sum():.0%} of "
          "post-email spend")


if __name__ == "__main__":
    main()
