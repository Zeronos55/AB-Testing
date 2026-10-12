"""Execute an AnalysisPlan against a CanonicalTable or AggregatedTable (pure: no I/O)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from isbetter.core.dependence import Clustered, CVOverlap, Iid, Paired, TimeSeries
from isbetter.core.plan import AnalysisPlan
from isbetter.core.result import ComparisonResult, MethodResult, Result
from isbetter.core.schema import CanonicalTable, SchemaError
from isbetter.core.sufficient import AggregatedTable, SufficientStats
from isbetter.decision.verdict import CVSpec, decide
from isbetter.methods.cluster import cluster_bootstrap, cluster_robust_diff
from isbetter.methods.cv import nadeau_bengio
from isbetter.methods.forecasting import diebold_mariano, suggest_block_length
from isbetter.methods.multiplicity import adjust
from isbetter.methods.parametric import paired_t, welch_t_stats
from isbetter.methods.proportions import fisher_exact_stats, two_proportion_z_stats
from isbetter.methods.resampling import block_bootstrap, paired_bootstrap, sign_flip_permutation
from isbetter.methods.variance_reduction import (
    cuped,
    delta_method_ratio,
    ratio_bootstrap,
    regression_adjustment,
)

Data = CanonicalTable | AggregatedTable

# Methods computed from sufficient statistics (work on both data types) ...
_STATS_METHODS = {
    "welch_t": welch_t_stats,
    "two_proportion_z": two_proportion_z_stats,
    "fisher_exact": fisher_exact_stats,
    "cuped": cuped,
    "regression_adjustment": regression_adjustment,
    "delta_method": delta_method_ratio,
}
# ... and cross-checks that need unit-level rows (skipped on aggregated data).
_NEEDS_ROWS = {"ratio_bootstrap"}


@dataclass(frozen=True)
class FamilyAdjustment:
    """Multiplicity handled outside the plan (e.g. across all success metrics of an experiment).

    ``adjusted_p`` is aligned with ``plan.candidates`` (None = report no adjusted p);
    ``alpha_ci`` is the per-comparison level used for confidence intervals.
    """

    alpha_ci: float
    family_size: int
    adjusted_p: np.ndarray | None = None


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


def _arm_stats(
    plan: AnalysisPlan, data: Data, variant: str
) -> tuple[SufficientStats, tuple[np.ndarray, np.ndarray] | None]:
    """Sufficient statistics of one arm (plus raw (y, x) rows when available)."""
    need_x = plan.metric_kind == "ratio" or plan.variance_reduction != "none"
    if isinstance(data, AggregatedTable):
        try:
            s = data.get(variant, plan.metric)
        except KeyError as exc:
            raise SchemaError(str(exc.args[0])) from exc
        if need_x and not s.has_x:
            what = "denominator" if plan.metric_kind == "ratio" else "covariate"
            raise SchemaError(
                f"aggregated data for metric '{plan.metric}' lacks {what} moments; "
                f"aggregate with the {what} column"
            )
        return s, None
    r = _rows(data, variant, plan.metric)
    y = r["value"]
    if plan.metric_kind == "ratio":
        assert plan.denominator is not None
        den = _rows(data, variant, plan.denominator)
        uy, ud = r["unit_id"].astype(str), den["unit_id"].astype(str)
        if len(set(uy)) != len(uy) or len(set(ud)) != len(ud):
            raise SchemaError("ratio metrics need one row per unit (duplicate unit_id found)")
        oy, od = np.argsort(uy, kind="stable"), np.argsort(ud, kind="stable")
        if not np.array_equal(uy[oy], ud[od]):
            raise SchemaError(
                f"numerator '{plan.metric}' and denominator '{plan.denominator}' "
                f"cover different units in variant '{variant}'"
            )
        y, x = y[oy], den["value"][od]
    elif plan.variance_reduction != "none":
        col = f"cov_{plan.covariate}"
        if col not in r:
            raise SchemaError(f"missing covariate column '{col}'")
        x = r[col].astype(np.float64)
    else:
        return SufficientStats.from_arrays(y), None
    return SufficientStats.from_arrays(y, x), (y, x)


def _analyse_iid(
    plan: AnalysisPlan,
    data: Data,
    cand: str,
    alpha: float,
    rng: np.random.Generator,
    with_cross: bool,
) -> _Outcome:
    sa, rows_a = _arm_stats(plan, data, cand)
    sb, rows_b = _arm_stats(plan, data, plan.baseline)
    primary = _STATS_METHODS[plan.primary](sa, sb, alpha)
    cross: list[MethodResult] = []
    if with_cross:
        for name in plan.cross_checks:
            if name == "ratio_bootstrap":
                if rows_a is None or rows_b is None:
                    continue  # needs unit rows; aggregated data cannot supply them
                cross.append(ratio_bootstrap(*rows_a, *rows_b, rng, alpha=alpha))
                continue
            res = _STATS_METHODS[name](sa, sb, alpha)
            if plan.variance_reduction != "none":
                res = dataclasses.replace(
                    res, rationale=f"Unadjusted comparison, for reference. {res.rationale}"
                )
            cross.append(res)
    return _Outcome(primary, cross, sa.n + sb.n)


def _loss_series(
    table: CanonicalTable, variant: str, metric: str, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-origin losses of one variant, in time order, plus their time stamps."""
    r = _rows(table, variant, metric)
    if "time" not in r:
        raise SchemaError("dependence=timeseries requires a 'time' column")
    if "horizon" in r and not (r["horizon"].astype(float) == horizon).all():
        found = sorted({int(h) for h in r["horizon"].astype(float)})
        raise SchemaError(
            f"metric '{metric}' holds losses at horizon(s) {found} but the plan declares "
            f"horizon={horizon}; the Diebold-Mariano variance depends on the horizon"
        )
    order = np.argsort(r["time"], kind="stable")
    return r["value"][order], r["time"][order]


def _analyse_timeseries(
    plan: AnalysisPlan,
    table: CanonicalTable,
    cand: str,
    dep: TimeSeries,
    alpha: float,
    rng: np.random.Generator,
    with_cross: bool,
) -> _Outcome:
    la, ta = _loss_series(table, cand, plan.metric, dep.horizon)
    lb, tb = _loss_series(table, plan.baseline, plan.metric, dep.horizon)
    if not np.array_equal(ta, tb):
        raise SchemaError(
            f"variants '{cand}' and '{plan.baseline}' were not scored on the same forecast "
            f"origins for metric '{plan.metric}'"
        )
    res = diebold_mariano(la, lb, dep.horizon, alpha, hln=True, hac_lags=dep.hac_lags)
    cross: list[MethodResult] = []
    if with_cross:
        d = la - lb
        suggestion = suggest_block_length(d)
        user = dep.block_len is not None
        length = (
            dep.block_len if dep.block_len is not None else suggestion.integer("stationary", len(d))
        )
        boot = block_bootstrap(la, lb, rng, length, "stationary", alpha=alpha)
        boot.diagnostics.update(
            block_len_source="user" if user else suggestion.source,
            suggested_block_len={
                "stationary": suggestion.stationary,
                "circular": suggestion.circular,
                "source": suggestion.source,
            },
        )
        cross.append(boot)
    return _Outcome(res, cross, len(la))


def _analyse(
    plan: AnalysisPlan,
    table: Data,
    cand: str,
    alpha: float,
    rng: np.random.Generator,
    with_cross: bool,
) -> _Outcome:
    dep = plan.dependence
    if isinstance(dep, Iid):
        return _analyse_iid(plan, table, cand, alpha, rng, with_cross)
    if isinstance(table, AggregatedTable):
        raise SchemaError(
            f"aggregated data supports dependence=iid only (plan declares '{dep.kind}')"
        )
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
    if isinstance(dep, TimeSeries):
        return _analyse_timeseries(plan, table, cand, dep, alpha, rng, with_cross)
    raise ValueError(f"unsupported dependence {dep.kind!r}")  # pragma: no cover


def family_ci_alpha(correction: str, alpha: float, m: int, adj_p: np.ndarray) -> float:
    """Per-comparison CI level matching the correction (conservative where noted)."""
    if correction == "none" or m == 1:
        return alpha
    if correction in ("bonferroni", "holm"):
        return alpha / m  # Holm's CIs are not narrower than Bonferroni's in general
    # BH: false-coverage-rate adjustment of Benjamini & Yekutieli (2005)
    return alpha * max(int((adj_p < alpha).sum()), 1) / m


def raw_pvalues(plan: AnalysisPlan, data: Data) -> np.ndarray:
    """Unadjusted p-value of every candidate vs the baseline (p-values do not depend on alpha)."""
    rng = np.random.default_rng(plan.seed)
    first = [_analyse(plan, data, c, plan.alpha, rng, with_cross=False) for c in plan.candidates]
    return np.array([o.primary.p_value if o.primary.p_value is not None else 1.0 for o in first])


def run(plan: AnalysisPlan, data: Data, family: FamilyAdjustment | None = None) -> Result:
    """Run every candidate of ``plan``; ``family`` overrides the within-plan multiplicity."""
    if isinstance(data, AggregatedTable):
        keep = [c for c in plan.cross_checks if c not in _NEEDS_ROWS]
        plan = dataclasses.replace(plan, cross_checks=keep)
    rng = np.random.default_rng(plan.seed)
    m = len(plan.candidates)
    # Pass 1: p-values at the nominal level.
    pvals = raw_pvalues(plan, data)
    if family is None:
        adj = adjust(pvals, plan.correction)
        alpha_ci = family_ci_alpha(plan.correction, plan.alpha, m, adj)
        adj_out: np.ndarray | None = adj if plan.correction != "none" else None
        fam_size = m
    else:
        alpha_ci, fam_size, adj_out = family.alpha_ci, family.family_size, family.adjusted_p

    sign = 1.0 if plan.higher_is_better else -1.0
    comps: list[ComparisonResult] = []
    for i, cand in enumerate(plan.candidates):
        out = _analyse(plan, data, cand, alpha_ci, rng, with_cross=True)
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
                adjusted_p=float(adj_out[i]) if adj_out is not None else None,
                alpha_used=alpha_ci,
                family_size=fam_size,
            )
        )
    return Result(experiment=plan.experiment, plan=plan, comparisons=comps, seed=plan.seed)
