"""Step 2b: a guided tour of every Hillstrom column, plus a check that the groups are alike.

Reads data/hillstrom.csv, prints what each column holds and how it is spread, and compares the
three email groups on everything known *before* the email (random assignment should make them
alike). Saves results/01b_balance_check.csv.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hillstrom.csv"
OUT = ROOT / "results"


def main() -> None:
    df = pd.read_csv(DATA)

    print("Numbers (summary):")
    print(df[["recency", "history", "spend"]].describe().round(2).to_string())

    for col in ["history_segment", "zip_code", "channel", "segment"]:
        print(f"\n{col}:")
        print(df[col].value_counts().sort_index().to_string())

    print("\nPast-purchase flags (what they bought in the year BEFORE the email):")
    print(pd.crosstab(df["mens"], df["womens"], rownames=["mens"], colnames=["womens"]).to_string())
    print(f"\nnewbie (new customer in last 12 months): {df['newbie'].mean():.1%}")

    spenders = df["spend"] > 0
    print(f"\nspend > 0 exactly when conversion == 1: {(spenders == (df['conversion'] == 1)).all()}")
    print(f"visit == 0 but conversion == 1 (bought without a recorded visit): "
          f"{((df['visit'] == 0) & (df['conversion'] == 1)).sum()}")

    # Balance check: before-the-email facts, per group. Randomisation should make these similar.
    before = df.assign(
        urban=df["zip_code"].eq("Urban"), suburban=df["zip_code"].eq("Surburban"),
        rural=df["zip_code"].eq("Rural"), web=df["channel"].eq("Web"),
        phone=df["channel"].eq("Phone"), multichannel=df["channel"].eq("Multichannel"),
    )
    cols = ["recency", "history", "mens", "womens", "newbie", "urban", "suburban", "rural",
            "web", "phone", "multichannel"]
    balance = before.groupby("segment")[cols].mean().T
    balance.index.name = "before_email_fact"
    print("\nBalance check: average of each before-the-email fact, per group:")
    print(balance.round(3).to_string())

    OUT.mkdir(exist_ok=True)
    balance.reset_index().to_csv(OUT / "01b_balance_check.csv", index=False)


if __name__ == "__main__":
    main()
