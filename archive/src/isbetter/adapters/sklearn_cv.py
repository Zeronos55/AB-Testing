"""Run repeated stratified k-fold CV and emit canonical per-fold scores.

Requires the ``[ml]`` extra (scikit-learn). Imported lazily so the base install stays light.
Every variant sees identical folds, so per-fold scores are paired by ``(repeat, fold)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from isbetter.core.schema import CanonicalTable, build_table

EstimatorFactory = Callable[[], Any]


@dataclass
class CVOutput:
    table: CanonicalTable
    y: np.ndarray
    # variant -> (repeats, n) out-of-fold P(y=1); each row is predicted once per repeat
    oof_proba: dict[str, np.ndarray]
    n_train: int
    n_test: int
    repeats: int
    folds: int
    seed: int


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    estimators: Mapping[str, EstimatorFactory],
    n_splits: int = 10,
    n_repeats: int = 10,
    seed: int = 0,
    metric: str = "pr_auc",
) -> CVOutput:
    """Score every estimator on identical repeated stratified folds (PR-AUC)."""
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    if metric != "pr_auc":
        raise ValueError(
            "only metric='pr_auc' is supported (accuracy is meaningless when imbalanced)"
        )
    if len(estimators) < 2:
        raise ValueError("need at least two estimators to compare")
    y = np.asarray(y)
    n = len(y)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            "unit_id",
            "pair_id",
            "variant",
            "metric",
            "value",
            "repeat",
            "fold",
            "n_train",
            "n_test",
        )
    }
    oof = {v: np.full((n_repeats, n), np.nan, dtype=np.float32) for v in estimators}
    sizes: set[tuple[int, int]] = set()
    for i, (tr, te) in enumerate(cv.split(X, y)):
        rep, fold = divmod(i, n_splits)
        sizes.add((len(tr), len(te)))
        for name, make in estimators.items():
            model = make()
            model.fit(X[tr], y[tr])
            proba = model.predict_proba(X[te])[:, 1]
            oof[name][rep, te] = proba
            uid = f"r{rep}f{fold}"
            for key, val in (
                ("unit_id", uid),
                ("pair_id", uid),
                ("variant", name),
                ("metric", metric),
                ("value", float(average_precision_score(y[te], proba))),
                ("repeat", rep),
                ("fold", fold),
                ("n_train", len(tr)),
                ("n_test", len(te)),
            ):
                cols[key].append(val)
    # Folds differ in size by at most one row; report the mean sizes.
    n_train = round(float(np.mean([s[0] for s in sizes])))
    n_test = round(float(np.mean([s[1] for s in sizes])))
    return CVOutput(
        table=build_table(cols),
        y=y,
        oof_proba=oof,
        n_train=n_train,
        n_test=n_test,
        repeats=n_repeats,
        folds=n_splits,
        seed=seed,
    )
