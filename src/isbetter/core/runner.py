"""Execute an AnalysisPlan against a CanonicalTable (pure: no I/O)."""

from __future__ import annotations

import numpy as np

from isbetter.core.dependence import Iid, Paired
from isbetter.core.plan import AnalysisPlan
from isbetter.core.result import ComparisonResult, MethodResult, Result
from isbetter.core.schema import CanonicalTable, SchemaError
from isbetter.decision.verdict import decide
from isbetter.methods.parametric import paired_t, welch_t
from isbetter.methods.resampling import paired_bootstrap, sign_flip_permutation


def _paired_arrays(
    table: CanonicalTable, a: str, b: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    ra, rb = table.select(a, metric), table.select(b, metric)
    if len(ra["value"]) == 0 or len(rb["value"]) == 0:
        raise SchemaError(f"no rows for variants '{a}'/'{b}' and metric '{metric}'")
    ka = np.argsort(ra["pair_id"].astype(str), kind="stable")
    kb = np.argsort(rb["pair_id"].astype(str), kind="stable")
    return ra["value"][ka], rb["value"][kb]


def run(plan: AnalysisPlan, table: CanonicalTable) -> Result:
    rng = np.random.default_rng(plan.seed)
    comps: list[ComparisonResult] = []
    sign = 1.0 if plan.higher_is_better else -1.0
    for cand in plan.candidates:
        cross: list[MethodResult] = []
        if isinstance(plan.dependence, Paired):
            x, y = _paired_arrays(table, cand, plan.baseline, plan.metric)
            primary = paired_t(x, y, plan.alpha)
            cross.append(paired_bootstrap(x, y, rng, alpha=plan.alpha))
            cross.append(sign_flip_permutation(x, y, rng))
            n = len(x)
        elif isinstance(plan.dependence, Iid):
            x = table.select(cand, plan.metric)["value"]
            y = table.select(plan.baseline, plan.metric)["value"]
            if len(x) == 0 or len(y) == 0:
                raise SchemaError(f"no rows for '{cand}'/'{plan.baseline}' on '{plan.metric}'")
            primary = welch_t(x, y, plan.alpha)
            n = len(x) + len(y)
        else:  # pragma: no cover - router already rejects
            raise ValueError("unsupported dependence")
        assert primary.ci_low is not None and primary.ci_high is not None
        lo, hi = sorted((sign * primary.ci_low, sign * primary.ci_high))
        verdict = decide(lo, hi, plan.min_effect, n=n)
        comps.append(
            ComparisonResult(
                baseline=plan.baseline,
                candidate=cand,
                metric=plan.metric,
                primary=primary,
                cross_checks=cross,
                verdict=verdict,
                oriented_estimate=sign * primary.estimate,
                oriented_ci_low=lo,
                oriented_ci_high=hi,
                n_pairs=n,
            )
        )
    return Result(experiment=plan.experiment, plan=plan, comparisons=comps, seed=plan.seed)
