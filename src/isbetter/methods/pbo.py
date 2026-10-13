"""Probability of Backtest Overfitting by combinatorially symmetric cross-validation (CSCV).

Bailey, Borwein, Lopez de Prado & Zhu (2017), *The Probability of Backtest Overfitting*,
J. Computational Finance 20(4). Given the performance of ``N`` strategy configurations over
``T`` periods:

1. split the ``T`` periods into ``S`` contiguous blocks (``S`` even);
2. for each of the ``C(S, S/2)`` ways to pick half of the blocks as in-sample (IS) data, the
   other half being out-of-sample (OOS):
   find the configuration ``n*`` with the best IS performance and look up its relative OOS
   rank ``omega = rank / (N + 1)`` (1 = worst), then ``lambda = ln(omega / (1 - omega))``;
3. ``PBO = P(lambda <= 0)``: how often the configuration that looked best in-sample ends up at
   or below the median out-of-sample.

With ``S = 16`` there are 12,870 splits. The computation is vectorised: per-block counts, sums
and sums of squares are combined with a 0/1 split-by-block matrix, so a split's IS and OOS
Sharpe ratios for all ``N`` configurations come from one matrix product, never from slicing
the returns again. Splits are processed in chunks, so memory is O(chunk * N).

PBO needs *every* configuration that was tried, not just the winner (see
:class:`isbetter.adapters.strategy_returns.TrialRegistry`); a PBO computed on a pre-filtered
set of survivors says nothing about the search that produced them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

from isbetter.core.result import MethodResult

_MAX_BLOCKS = 20  # C(20, 10) = 184,756 splits
_CHUNK_ELEMENTS = 4_000_000  # elements of one (chunk, N) work array
METRICS = ("sharpe", "mean")


@dataclass(frozen=True)
class CSCVResult:
    """Per-split outcome of CSCV (all arrays have one entry per split)."""

    logits: np.ndarray  # lambda = ln(omega / (1 - omega))
    omega: np.ndarray  # relative OOS rank of the IS-best configuration, in (0, 1)
    best_index: np.ndarray  # column of the IS-best configuration
    is_perf: np.ndarray  # IS performance of the IS-best configuration
    oos_perf: np.ndarray  # OOS performance of the IS-best configuration
    n_blocks: int
    n_configs: int

    @property
    def n_splits(self) -> int:
        return len(self.logits)

    @property
    def pbo(self) -> float:
        return float(np.mean(self.logits <= 0.0))


def _split_matrix(n_blocks: int) -> np.ndarray:
    """0/1 matrix ``(C(S, S/2), S)``: row = a split, 1 marks the blocks used in-sample."""
    half = n_blocks // 2
    idx = np.array(list(combinations(range(n_blocks), half)), dtype=np.intp)
    mat = np.zeros((len(idx), n_blocks), dtype=np.float64)
    np.put_along_axis(mat, idx, 1.0, axis=1)
    return mat


def _performance(n: np.ndarray, s: np.ndarray, q: np.ndarray, metric: str) -> np.ndarray:
    """Performance of every configuration from counts, sums and sums of squares.

    ``n`` is ``(rows,)`` and ``s``, ``q`` are ``(rows, N)``. The Sharpe ratio uses ddof=1; a
    configuration with no variance in a half (for example one that never trades) gets 0.
    """
    mean = s / n[:, None]
    if metric == "mean":
        return np.asarray(mean)
    var = (q - n[:, None] * mean**2) / (n[:, None] - 1.0)
    scale = np.maximum(q / n[:, None], np.finfo(float).tiny)
    ok = var > 1e-12 * scale
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = mean / np.sqrt(np.where(ok, var, 1.0))
    return np.asarray(np.where(ok, sr, 0.0))


def cscv(returns: Any, n_blocks: int = 16, metric: str = "sharpe") -> CSCVResult:
    """Run CSCV on a ``(T, N)`` matrix of per-period returns of ``N`` configurations."""
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("returns must be a (T, N) matrix: periods by configurations")
    T, N = x.shape
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if n_blocks < 2 or n_blocks % 2 or n_blocks > _MAX_BLOCKS:
        raise ValueError(f"n_blocks must be an even integer in 2..{_MAX_BLOCKS}")
    if N < 2:
        raise ValueError("PBO compares configurations: need at least 2 columns")
    if T < 2 * n_blocks:
        raise ValueError(f"need at least 2 observations per block: T={T} < {2 * n_blocks}")
    if not np.isfinite(x).all():
        raise ValueError("returns contain NaN or infinite values")

    bounds = np.cumsum([0] + [len(b) for b in np.array_split(np.arange(T), n_blocks)])
    cnt = np.diff(bounds).astype(np.float64)
    sums = np.add.reduceat(x, bounds[:-1], axis=0)
    sqs = np.add.reduceat(x * x, bounds[:-1], axis=0)

    splits = _split_matrix(n_blocks)
    oos_splits = 1.0 - splits
    rows = max(1, _CHUNK_ELEMENTS // N)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("omega", "best", "is", "oos")}
    for lo in range(0, len(splits), rows):
        a, b = splits[lo : lo + rows], oos_splits[lo : lo + rows]
        is_perf = _performance(a @ cnt, a @ sums, a @ sqs, metric)
        oos_perf = _performance(b @ cnt, b @ sums, b @ sqs, metric)
        best = is_perf.argmax(axis=1)  # first maximum on ties
        pick = np.arange(len(best))
        star = oos_perf[pick, best]
        below = (oos_perf < star[:, None]).sum(axis=1)
        tied = (oos_perf == star[:, None]).sum(axis=1)  # includes the configuration itself
        rank = below + (tied + 1.0) / 2.0  # 1 = worst; ties share the mean rank
        parts["omega"].append(rank / (N + 1.0))
        parts["best"].append(best)
        parts["is"].append(is_perf[pick, best])
        parts["oos"].append(star)
    omega = np.concatenate(parts["omega"])
    return CSCVResult(
        logits=np.log(omega / (1.0 - omega)),
        omega=omega,
        best_index=np.concatenate(parts["best"]),
        is_perf=np.concatenate(parts["is"]),
        oos_perf=np.concatenate(parts["oos"]),
        n_blocks=n_blocks,
        n_configs=N,
    )


def pbo_cscv(
    returns: Any,
    n_blocks: int = 16,
    metric: str = "sharpe",
    names: Sequence[str] | None = None,
) -> MethodResult:
    """Probability of Backtest Overfitting of a grid of configurations (``estimate`` = PBO).

    ``returns`` is ``(T, N)``. Diagnostics hold the number of splits, the share of splits where
    the IS-best configuration lost money OOS, the slope of OOS on IS performance (negative
    means in-sample ranking does not carry over) and how often each configuration was chosen.
    """
    x = np.asarray(returns, dtype=np.float64)
    res = cscv(x, n_blocks, metric)
    N = res.n_configs
    labels = list(names) if names is not None else [f"config_{j}" for j in range(N)]
    if len(labels) != N:
        raise ValueError("names must have one entry per column of returns")
    counts = np.bincount(res.best_index, minlength=N)
    top = np.argsort(-counts, kind="stable")[:5]
    if res.is_perf.std() > 0:
        slope = float(np.polyfit(res.is_perf, res.oos_perf, 1)[0])
    else:
        slope = float("nan")
    pbo = res.pbo
    return MethodResult(
        method="pbo_cscv",
        estimate=pbo,
        ci_low=None,
        ci_high=None,
        ci_level=None,
        p_value=None,
        statistic=float(np.median(res.logits)),
        df=None,
        n_effective=float(x.shape[0]),
        rationale=(
            f"Probability of Backtest Overfitting (combinatorially symmetric CV): {N} "
            f"configurations over {x.shape[0]} periods in {res.n_blocks} blocks, "
            f"{res.n_splits} IS/OOS splits. In {pbo:.0%} of them the configuration that looked "
            f"best in-sample ranked at or below the median out-of-sample (about 50% means the "
            f"in-sample ranking is no better than chance; near 0% means it carries over)."
        ),
        diagnostics={
            "pbo": pbo,
            "n_splits": res.n_splits,
            "n_blocks": res.n_blocks,
            "n_configs": N,
            "metric": metric,
            "median_logit": float(np.median(res.logits)),
            "prob_oos_loss": float(np.mean(res.oos_perf < 0)),
            "oos_on_is_slope": slope,
            "mean_is_perf": float(res.is_perf.mean()),
            "mean_oos_perf": float(res.oos_perf.mean()),
            "most_selected": [(labels[j], int(counts[j])) for j in top if counts[j] > 0],
        },
    )


def n_splits(n_blocks: int) -> int:
    """Number of IS/OOS splits for ``n_blocks`` blocks: ``C(S, S/2)``."""
    return comb(n_blocks, n_blocks // 2)
