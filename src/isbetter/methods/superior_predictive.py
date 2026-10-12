"""Many models against a benchmark: Hansen's SPA, Romano-Wolf StepM and the Model Confidence Set.

These are thin wrappers around :mod:`arch.bootstrap` (``SPA``, ``StepM``, ``MCS``), which
needs the ``[forecast]`` extra. They take **per-origin losses** (lower is better), run the
block bootstrap with a passed-in ``numpy.random.Generator`` and report through
:class:`~isbetter.core.result.MethodResult`.

Unlike a pairwise test, the answer is a *set* of models, so it lives in ``diagnostics``:

* :func:`spa_test`  : H0 "no model is better than the benchmark" (one p-value).
* :func:`stepm`     : which models are significantly better than the benchmark, with
  family-wise error control.
* :func:`model_confidence_set` : the models that cannot be told apart from the best one.

When ``block_len`` is not given it is the largest Politis-White stationary block length
among the loss differentials, the conservative choice for several dependent series.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from isbetter.core.result import MethodResult
from isbetter.methods.forecasting import suggest_block_length

_BOOTSTRAPS = ("stationary", "circular")
_INSTALL_HINT = "install the forecast extra: pip install 'isbetter[forecast]' (provides arch)"


def _arch() -> Any:
    try:
        import arch.bootstrap as ab
    except ImportError as exc:  # pragma: no cover - exercised only without arch installed
        raise ImportError(f"SPA / StepM / MCS wrap the arch package: {_INSTALL_HINT}") from exc
    return ab


def _frames(benchmark: np.ndarray, models: Mapping[str, np.ndarray]) -> tuple[Any, Any, np.ndarray]:
    import pandas as pd

    bench = np.asarray(benchmark, dtype=np.float64)
    if bench.ndim != 1 or len(bench) < 10:
        raise ValueError("benchmark must be a 1-D loss series with at least 10 origins")
    if len(models) < 1:
        raise ValueError("need at least one model to compare with the benchmark")
    cols = {}
    for name, loss in models.items():
        arr = np.asarray(loss, dtype=np.float64)
        if arr.shape != bench.shape:
            raise ValueError(f"loss series of model {name!r} must have the benchmark's length")
        cols[str(name)] = arr
    frame = pd.DataFrame(cols)
    if not (np.isfinite(frame.to_numpy()).all() and np.isfinite(bench).all()):
        raise ValueError("loss series contain NaN or infinite values")
    return pd.Series(bench, name="benchmark"), frame, bench


def _check(bootstrap: str, size: float, reps: int) -> None:
    if bootstrap not in _BOOTSTRAPS:
        raise ValueError(f"bootstrap must be one of {_BOOTSTRAPS}")
    if not 0 < size < 1:
        raise ValueError("size must be in (0, 1)")
    if reps < 100:
        raise ValueError("reps must be at least 100")


def _block_len(diffs: np.ndarray, given: int | None, bootstrap: str) -> tuple[int, str]:
    n = diffs.shape[0]
    if given is not None:
        if not 1 <= given <= n:
            raise ValueError(f"block_len must be between 1 and {n}")
        return given, "user"
    sug = [suggest_block_length(diffs[:, j]) for j in range(diffs.shape[1])]
    longest = max(sug, key=lambda s: s.stationary if bootstrap == "stationary" else s.circular)
    return longest.integer(bootstrap, n), longest.source


def _mean_diff(bench: np.ndarray, frame: Any) -> dict[str, float]:
    """Mean of (model loss - benchmark loss) per model; negative means the model is better."""
    return {str(c): float((frame[c].to_numpy() - bench).mean()) for c in frame.columns}


def spa_test(
    benchmark: np.ndarray,
    models: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    reps: int = 1000,
    block_len: int | None = None,
    bootstrap: str = "stationary",
    studentize: bool = True,
    alpha: float = 0.05,
) -> MethodResult:
    """Hansen (2005) Superior Predictive Ability test, via ``arch.bootstrap.SPA``.

    H0: no model has lower expected loss than the benchmark. ``p_value`` is Hansen's
    *consistent* p-value; ``diagnostics`` also holds the lower and upper bounds. ``estimate``
    is the smallest mean of ``loss_model - loss_benchmark`` (the best model; negative means
    it beats the benchmark). ``diagnostics['better_models']`` lists the models flagged at
    level ``alpha`` by the consistent p-value's critical value.
    """
    _check(bootstrap, alpha, reps)
    bench_s, frame, bench = _frames(benchmark, models)
    diffs = bench[:, None] - frame.to_numpy()
    length, source = _block_len(diffs, block_len, bootstrap)
    spa = _arch().SPA(
        bench_s, frame, block_size=length, reps=reps, bootstrap=bootstrap,
        studentize=studentize, seed=rng,
    )  # fmt: skip
    spa.compute()
    pv = {k: float(v) for k, v in spa.pvalues.items()}
    better = [str(frame.columns[i]) for i in spa.better_models(alpha, "consistent")]
    means = _mean_diff(bench, frame)
    best = min(means, key=lambda k: means[k])
    return MethodResult(
        method="hansen_spa",
        estimate=means[best],
        ci_low=None,
        ci_high=None,
        ci_level=None,
        p_value=pv["consistent"],
        statistic=None,
        df=None,
        n_effective=float(len(bench)),
        rationale=(
            "Hansen's Superior Predictive Ability test (arch): tests whether ANY of "
            f"{frame.shape[1]} models beats the benchmark while accounting for having "
            "searched over all of them, with a "
            f"{bootstrap} block bootstrap ({reps} resamples, mean block length {length})"
            + (", studentized." if studentize else ".")
        ),
        diagnostics={
            "pvalues": pv,
            "better_models": better,
            "best_model": best,
            "mean_loss_difference": means,
            "block_len": length,
            "block_len_source": source,
            "bootstrap": bootstrap,
            "reps": reps,
            "studentize": studentize,
            "alpha": alpha,
        },
    )


def stepm(
    benchmark: np.ndarray,
    models: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    size: float = 0.05,
    reps: int = 1000,
    block_len: int | None = None,
    bootstrap: str = "stationary",
    studentize: bool = True,
) -> MethodResult:
    """Romano & Wolf (2005) StepM, via ``arch.bootstrap.StepM``.

    Finds the models that significantly beat the benchmark, controlling the family-wise
    error rate at ``size``. ``diagnostics['superior_models']`` is the set (it can be empty).
    """
    _check(bootstrap, size, reps)
    bench_s, frame, bench = _frames(benchmark, models)
    diffs = bench[:, None] - frame.to_numpy()
    length, source = _block_len(diffs, block_len, bootstrap)
    sm = _arch().StepM(
        bench_s, frame, size=size, block_size=length, reps=reps, bootstrap=bootstrap,
        studentize=studentize, seed=rng,
    )  # fmt: skip
    sm.compute()
    superior = [str(m) for m in sm.superior_models]
    means = _mean_diff(bench, frame)
    best = min(means, key=lambda k: means[k])
    return MethodResult(
        method="romano_wolf_stepm",
        estimate=means[best],
        ci_low=None,
        ci_high=None,
        ci_level=None,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(len(bench)),
        rationale=(
            "Romano-Wolf StepM (arch): a step-down procedure that returns every model "
            f"significantly better than the benchmark with family-wise error rate {size:g} "
            f"over {frame.shape[1]} models ({bootstrap} block bootstrap, {reps} resamples, "
            f"mean block length {length}). Found {len(superior)} superior model(s)."
        ),
        diagnostics={
            "superior_models": superior,
            "best_model": best,
            "mean_loss_difference": means,
            "block_len": length,
            "block_len_source": source,
            "bootstrap": bootstrap,
            "reps": reps,
            "studentize": studentize,
            "size": size,
        },
    )


def model_confidence_set(
    losses: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    size: float = 0.1,
    reps: int = 1000,
    block_len: int | None = None,
    bootstrap: str = "stationary",
    method: str = "R",
) -> MethodResult:
    """Hansen, Lunde & Nason (2011) Model Confidence Set, via ``arch.bootstrap.MCS``.

    The set of models that cannot be distinguished from the best at confidence ``1 - size``.
    ``method`` is the elimination statistic (``'R'``: range, ``'max'``: maximum). The
    ``estimate`` is the number of models in the set; the members, the eliminated models and
    the MCS p-values are in ``diagnostics`` (``included``, ``excluded``, ``pvalues``).
    """
    _check(bootstrap, size, reps)
    if method not in ("R", "max"):
        raise ValueError("method must be 'R' or 'max'")
    if len(losses) < 2:
        raise ValueError("the model confidence set needs at least two models")
    import pandas as pd

    frame = pd.DataFrame({str(k): np.asarray(v, dtype=np.float64) for k, v in losses.items()})
    arr = frame.to_numpy()
    if arr.shape[0] < 10:
        raise ValueError("need at least 10 forecast origins")
    if not np.isfinite(arr).all():
        raise ValueError("loss series contain NaN or infinite values")
    diffs = arr[:, :1] - arr[:, 1:]
    length, source = _block_len(diffs, block_len, bootstrap)
    mcs = _arch().MCS(
        frame, size, reps=reps, block_size=length, method=method, bootstrap=bootstrap, seed=rng
    )
    mcs.compute()
    included = [str(m) for m in mcs.included]
    excluded = [str(m) for m in mcs.excluded]
    pv = {str(k): float(v) for k, v in mcs.pvalues["Pvalue"].items()}
    best = min(frame.columns, key=lambda c: float(frame[c].mean()))
    return MethodResult(
        method="model_confidence_set",
        estimate=float(len(included)),
        ci_low=None,
        ci_high=None,
        ci_level=None,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(arr.shape[0]),
        rationale=(
            "Model Confidence Set (arch): sequentially eliminates the worst model while the "
            f"equal-accuracy hypothesis is rejected at size {size:g}; the survivors "
            f"({len(included)} of {frame.shape[1]}) cannot be separated from the best model "
            f"({bootstrap} block bootstrap, {reps} resamples, mean block length {length})."
        ),
        diagnostics={
            "included": included,
            "excluded": excluded,
            "pvalues": pv,
            "lowest_mean_loss": str(best),
            "block_len": length,
            "block_len_source": source,
            "bootstrap": bootstrap,
            "reps": reps,
            "method": method,
            "size": size,
        },
    )
