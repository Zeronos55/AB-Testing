"""Case study 1: which fraud classifier is really better, on PR-AUC?

    uv run python case_studies/01_fraud_classifiers/run.py            # synthetic stand-in data
    uv run python case_studies/01_fraud_classifiers/run.py --data creditcard.csv   # real ULB data

The real ULB credit-card file (Kaggle "creditcardfraud", 284,807 rows, 0.172% fraud, label
column ``Class``) needs a Kaggle login and could not be downloaded in the environment this was
built in, so by default a clearly labelled SYNTHETIC imbalanced dataset is used. Results on
synthetic data say nothing about real fraud.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from isbetter.adapters.sklearn_cv import run_cv
from isbetter.core.config import load_config
from isbetter.core.dependence import CVOverlap
from isbetter.core.plan import build_plan
from isbetter.core.runner import run
from isbetter.methods.classification import mcnemar
from isbetter.methods.cv import naive_kfold_t
from isbetter.report.markdown import render_markdown

HERE = Path(__file__).parent


def load_data(path: str | None, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    if path:
        raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        y = np.asarray(raw["Class"], dtype=int)
        cols = [c for c in raw.dtype.names if c not in ("Class", "Time")]
        X = np.column_stack([raw[c] for c in cols]).astype(np.float64)
        return X, y, f"real data from {path} ({len(y):,} rows, {y.mean():.3%} positive)"
    X, y = make_classification(
        n_samples=30_000,
        n_features=30,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=[0.99],
        flip_y=0.002,
        class_sep=1.0,
        random_state=seed,
    )
    return (
        X,
        y,
        f"SYNTHETIC stand-in ({len(y):,} rows, {y.mean():.3%} positive) -- NOT real fraud data",
    )


def main() -> None:
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to creditcard.csv (Kaggle ULB file)")
    args = ap.parse_args()

    cfg = load_config(HERE / "config.yaml")
    dep = cfg.to_dependence()
    assert isinstance(dep, CVOverlap)
    X, y, source = load_data(args.data, cfg.seed)
    print("data:", source)

    estimators = {
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=60, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=cfg.seed
        ),
        "lightgbm": lambda: lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            verbose=-1,
            random_state=cfg.seed,
        ),
    }
    out = run_cv(X, y, estimators, n_splits=dep.folds, n_repeats=dep.repeats, seed=cfg.seed)
    if args.data:  # keep the declared design honest for the actual data size
        dep = CVOverlap(out.n_train, out.n_test, dep.repeats, dep.folds)
    elif (out.n_train, out.n_test) != (dep.n_train, dep.n_test):
        sys.exit(
            f"config.yaml n_train/n_test do not match the CV design: {out.n_train}/{out.n_test}"
        )

    (HERE / "out").mkdir(exist_ok=True)
    cols = out.table.columns
    with open(HERE / "out" / "cv_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit_id", "pair_id", "variant", "metric", "value"])
        for i in range(len(out.table)):
            w.writerow(
                [cols[k][i] for k in ("unit_id", "pair_id", "variant", "metric")]
                + [f"{cols['value'][i]:.8f}"]
            )

    plan = build_plan(cfg)
    plan.dependence = dep
    result = run(plan, out.table)

    # --- naive analysis for contrast --------------------------------------------------
    def fold_scores(v: str) -> np.ndarray:
        r = out.table.select(v, "pr_auc")
        return r["value"][np.argsort(r["pair_id"].astype(str), kind="stable")]

    naive_rows = []
    for c in result.comparisons:
        d = fold_scores(c.candidate) - fold_scores("logreg")
        nv = naive_kfold_t(d)
        naive_rows.append(
            f"| {c.candidate} | {nv.estimate:+.4f} | [{nv.ci_low:+.4f}, {nv.ci_high:+.4f}] | "
            f"{nv.p_value:.3g} | [{c.primary.ci_low:+.4f}, {c.primary.ci_high:+.4f}] "
            f"(p={c.primary.p_value:.3g}) |"
        )

    # --- McNemar: frauds caught under an equal alert budget (repeat 0) ----------------
    budget = max(1, int(round(0.01 * len(y))))
    pos = y == 1
    caught = {}
    for v, proba in out.oof_proba.items():
        thr = np.sort(proba[0])[-budget]
        caught[v] = (proba[0] >= thr)[pos]
    mc_rows = []
    for c in result.comparisons:
        m = mcnemar(caught[c.candidate], caught["logreg"])
        mc_rows.append(
            f"| {c.candidate} vs logreg | {int(caught[c.candidate].sum())} / "
            f"{int(caught['logreg'].sum())} of {int(pos.sum())} | "
            f"b={m.diagnostics['b']}, c={m.diagnostics['c']} | {m.method} | {m.p_value:.3g} |"
        )

    data_note = (
        "Real data were used."
        if args.data
        else "The real ULB file could not be downloaded here; rerun with "
        "`--data creditcard.csv` for the real analysis. Nothing below is a claim about "
        "real fraud data."
    )
    base = render_markdown(result)
    md = f"""# Case study 1: fraud classifiers (PR-AUC)

> **Data:** {source}.
> {data_note}

## 1. Question

Are a random forest or LightGBM *really* better than a logistic-regression baseline at ranking
fraud, measured by PR-AUC (average precision)? Accuracy is not used: with about 1% positives a
model that never flags fraud already scores about 99%. A gain smaller than **0.01 PR-AUC** is
treated as practically irrelevant.

## 2. Data and dependence structure

Repeated stratified 5-fold CV, 6 repeats = 30 fold scores per model, all models on identical
folds, so scores are paired by (repeat, fold). Training sets overlap across folds and repeats, so
the scores are **not independent**: `dependence = cv_overlap` with n_train = {dep.n_train:,} and
n_test = {dep.n_test:,}.

## 3. Naive analysis, and why it misleads

A plain paired t-test on the 30 fold scores treats them as independent. In the failure gallery
(`validation/failure_gallery/`) that test falsely flags two equally good models in 28-57% of
datasets when 3-10 repeats are used.

| candidate | naive mean diff | naive 95% CI | naive p | corrected 95% CI (Holm level) |
|---|---:|---|---:|---|
{chr(10).join(naive_rows)}

## 4. Correct analysis (Nadeau-Bengio + Holm across the 2 candidates)

{base.split("## Results", 1)[1]}
## 5. Second opinion: McNemar on frauds caught

Per-row out-of-fold scores from repeat 0; each model flags its top 1% highest-scoring rows
({budget} alerts). Items are the {int(pos.sum())} true frauds; "correct" means the fraud was in
the alert budget.

| comparison | frauds caught (candidate / logreg) | discordant pairs | test | p |
|---|---|---|---|---:|
{chr(10).join(mc_rows)}

## 6. What would change the conclusion

* A smaller margin than 0.01 would turn "equivalent"/"may be trivial" verdicts into decisive
  ones only if the CI narrows; check the `n_needed` note above for whether repeats or more data
  are required.
* Different hyperparameters, features or a different alert budget could reorder the models.
* Real fraud data are non-stationary (the ULB set covers two days); random-split CV can
  overstate how well a model transfers to next week.
"""
    (HERE / "REPORT.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
