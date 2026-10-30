"""Step 2: meet the Hillstrom email data. Look only; no tests yet.

Reads data/hillstrom.csv, prints its shape, missing values, group sizes and the raw outcome
rate per group, and saves one summary table (results/01_group_summary.csv) and one chart.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hillstrom.csv"
OUT = ROOT / "results"

# Group sizes published for this dataset, so we notice a damaged or different copy.
EXPECTED_ROWS = 64_000
EXPECTED_GROUPS = {"No E-Mail": 21_306, "Mens E-Mail": 21_307, "Womens E-Mail": 21_387}


def main() -> None:
    df = pd.read_csv(DATA)

    print(f"Rows: {len(df):,} (expected {EXPECTED_ROWS:,})")
    print("\nColumns:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    sizes = df["segment"].value_counts()
    print("\nPeople per group (observed vs expected):")
    for name, expected in EXPECTED_GROUPS.items():
        print(f"  {name:<14} {sizes.get(name, 0):>7,}   expected {expected:>7,}")

    # visit/conversion are 0/1 flags, so their mean is the share of people who did it.
    summary = (
        df.groupby("segment")
        .agg(people=("visit", "size"), visit_rate=("visit", "mean"),
             purchase_rate=("conversion", "mean"), spend_per_person=("spend", "mean"))
        .reset_index()
    )
    print("\nRaw outcome rates per group (no test yet):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "01_group_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, col, title in zip(
        axes, ["visit_rate", "purchase_rate"], ["Visited the website", "Bought something"]
    ):
        ax.bar(summary["segment"], summary[col] * 100)
        ax.set_title(title)
        ax.set_ylabel("% of people")
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "01_outcome_rates.png", dpi=130)


if __name__ == "__main__":
    main()
