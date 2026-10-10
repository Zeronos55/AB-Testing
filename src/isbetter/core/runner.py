"""Execute an AnalysisPlan against a CanonicalTable (pure: no I/O)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isbetter.core.dependence import Clustered, CVOverlap, Iid, Paired
from isbetter.core.plan import AnalysisPlan
from isbetter.core.result import ComparisonResult, MethodResult, Result
from isbetter.core.schema import CanonicalTable, SchemaError
from isbetter.decision.verdict import CVSpec, decide
from isbetter.methods.cluster import cluster_bootstrap, cluster_robust_diff
from isbetter.methods.cv import nadeau_bengio
from isbetter.methods.multiplicity import adjust
from isbetter.methods.parametric import paired_t, welch_t
from isbetter.methods.resampling import paired_bootstrap, sign_flip_permutation


@dataclass
class _Outcome:
    primary: MethodResult
    cross: list[MethodResult]
    n: int
    cv: CVSpec | None = None


def _rows(table: CanonicalTable, variant: str, metric: str) -> dict[str, np.ndarray]:
    r = table.select(variant, metric)
    if len(r["value"]) == 0:
        raise SchemaError(f"no rows for variant '{variant}' and metric '{metric}'")
    return r


def _aligned(table: CanonicalTable, a: str, b: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    ra, rb = _rows(table, a, metric), _rows(table, b, metric)
    ka = np.argsort(ra["pair_id"].astype(str), kind="stable")
    kb = np.argsort(rb["pair_id"].astype(str), kind="stable")
    return ra["value"][ka], rb["value"][kb]


def _analyse(
    plan: AnalysisPlan,
    table: CanonicalTable,
    cand: str,
    alpha: float,
    rng: np.random.Generator,
    with_cross: bool,
) -> _Outcome:
    dep = plan.dependence
    base, metric = plan.baseline, plan.metric
    if isinstance(dep, Paired):
        x, y = _aligned(table, cand, base, metric)
        cross = (
            [paired_bootstrap(x, y, rng, alpha=alpha), sign_flip_permutation(x, y, rng)]
            if with_cross
            else []
        )
        return _Outcome(paired_t(x, y, alpha), cross, len(x))
    if isinstance(dep, CVOverlap):
        x, y = _aligned(table, cand, base, metric)
        if len(x) != dep.repeats * dep.folds:
            raise SchemaError(
                f"expected repeats*folds = {dep.repeats * dep.folds} scores per variant, "
                f"found {len(x)}"
            )
        res = nadeau_bengio(x - y, dep.n_train, dep.n_test, alpha)
        return _Outcome(res, [], len(x), CVSpec(len(x), dep.n_train, dep.n_test))
    if isinstance(dep, Clustered):
        ra, rb = _rows(table, cand, metric), _rows(table, base, metric)
        if dep.cluster_col not in ra or dep.cluster_col not in rb:
            raise SchemaError(f"missing cluster column '{dep.cluster_col}'")
        args = (ra["value"], ra[dep.cluster_col], rb["value"], rb[dep.cluster_col])
        cross = [cluster_bootstrap(*args, rng, alpha=alpha)] if with_cross else []
        return _Outcome(cluster_robust_diff(*args, alpha=alpha), cross, len(args[0]) + len(args[2]))
    if isinstance(dep, Iid):
        x = _rows(table, cand, metric)["value"]
        y = _rows(table, base, metric)["value"]
        return _Outcome(welch_t(x, y, alpha), [], len(x) + len(y))
    raise ValueError(f"unsupported dependence {dep.kind!r}")  # pragma: no cover


def _ci_alpha(correction: str, alpha: float, m: int, adj_p: np.ndarray) -> float:
    """Per-comparison CI level matching the correction (conservative where noted)."""
    if correction == "none" or m == 1:
        return alpha
    if correction in ("bonferroni", "holm"):
        return alpha / m  # Holm's CIs are not narrower than Bonferroni's in general
    # BH: false-coverage-rate adjustment of Benjamini & Yekutieli (2005)
    return alpha * max(int((adj_p < alpha).sum()), 1) / m


def run(plan: AnalysisPlan, table: CanonicalTable) -> Result:
    rng = np.random.default_rng(plan.seed)
    m = len(plan.candidates)
    # Pass 1: p-values at the nominal level (p-values do not depend on alpha).
    first = [_analyse(plan, table, c, plan.alpha, rng, with_cross=False) for c in plan.candidates]
    pvals = np.array([o.primary.p_value if o.primary.p_value is not None else 1.0 for o in first])
    adj = adjust(pvals, plan.correction)
    alpha_ci = _ci_alpha(plan.correction, plan.alpha, m, adj)

    sign = 1.0 if plan.higher_is_better else -1.0
    comps: list[ComparisonResult] = []
    for i, cand in enumerate(plan.candidates):
        out = _analyse(plan, table, cand, alpha_ci, rng, with_cross=True)
        primary = out.primary
        assert primary.ci_low is not None and primary.ci_high is not None
        lo, hi = sorted((sign * primary.ci_low, sign * primary.ci_high))
        verdict = decide(lo, hi, plan.min_effect, n=out.n, cv=out.cv)
        comps.append(
            ComparisonResult(
                baseline=plan.baseline,
                candidate=cand,
                metric=plan.metric,
                primary=primary,
                cross_checks=out.cross,
                verdict=verdict,
                oriented_estimate=sign * primary.estimate,
                oriented_ci_low=lo,
                oriented_ci_high=hi,
                n_pairs=out.n,
                adjusted_p=float(adj[i]) if plan.correction != "none" else None,
                alpha_used=alpha_ci,
            )
        )
    return Result(experiment=plan.experiment, plan=plan, comparisons=comps, seed=plan.seed)
