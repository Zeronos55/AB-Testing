"""Rolling-origin backtests: per-origin, per-horizon forecast losses as a canonical table.

A backtest forecasts ``H`` steps ahead from a sequence of forecast origins. For every
(variant, origin, horizon) the adapter emits one loss. Losses for different horizons are
different metrics (``mase_h1``, ``mase_h7``, ...) because each horizon has its own
autocorrelation structure: an ``h``-step-ahead error overlaps the errors of the next
``h - 1`` origins, so the Diebold-Mariano variance needs ``h - 1`` autocovariances.

Loss kinds (the ``value`` column holds the *per-observation* loss, so that a difference of
losses is the Diebold-Mariano loss differential):

* ``mae``   : absolute error ``|e|``
* ``mase``  : ``|e| / scale``, with ``scale`` the in-sample mean absolute seasonal-naive
  error (Hyndman & Koehler, 2006)
* ``rmsse`` : ``e^2 / scale^2``, with ``scale^2`` the in-sample mean squared seasonal-naive
  error. The M5 headline RMSSE is the square root of the *mean* of these values.

The scale is computed once on the first training window (``y[:initial]``). It is a constant,
so it changes the units but not the Diebold-Mariano statistic, and it uses no data from the
future of any origin.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from isbetter.core.dependence import TimeSeries
from isbetter.core.schema import CanonicalTable, build_table

#: A point forecaster: ``(history, horizon) -> forecasts`` of shape ``(horizon,)``.
Forecaster = Callable[[np.ndarray, int], np.ndarray]

LOSS_KINDS = ("mae", "mase", "rmsse")


def seasonal_naive(season: int = 1) -> Forecaster:
    """Forecast each step with the value one season earlier (``season = 1``: last value)."""
    if season < 1:
        raise ValueError("season must be >= 1")

    def forecast(history: np.ndarray, horizon: int) -> np.ndarray:
        if len(history) < season:
            raise ValueError(f"need at least {season} observations of history")
        last = np.asarray(history[-season:], dtype=np.float64)
        return np.tile(last, horizon // season + 1)[:horizon]

    return forecast


def seasonal_scale(insample: np.ndarray, season: int = 1, squared: bool = False) -> float:
    """Mean (squared) absolute seasonal-naive error ``y_t - y_{t-season}`` on ``insample``."""
    y = np.asarray(insample, dtype=np.float64)
    if season < 1:
        raise ValueError("season must be >= 1")
    if len(y) <= season:
        raise ValueError(f"need more than season={season} in-sample observations to scale")
    diff = y[season:] - y[:-season]
    scale = float(np.mean(diff**2) if squared else np.mean(np.abs(diff)))
    if scale == 0:
        raise ValueError("in-sample seasonal-naive error is zero: scaled losses are undefined")
    return scale


@dataclass
class BacktestOutput:
    """Backtest losses plus the raw forecasts they came from."""

    table: CanonicalTable
    origins: np.ndarray  # position in ``y`` of the first forecast target of each origin
    forecasts: dict[str, np.ndarray]  # variant -> (n_origins, horizon)
    actuals: np.ndarray  # (n_origins, horizon)
    metrics: list[str]
    loss: str
    scale: float


def metric_name(loss: str, horizon: int) -> str:
    return f"{loss}_h{horizon}"


def _check_y(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("y must be a 1-D series")
    if not np.isfinite(arr).all():
        raise ValueError("y contains NaN or infinite values")
    return arr


def backtest_table(
    y: np.ndarray,
    forecasts: Mapping[str, np.ndarray],
    origins: Sequence[int] | np.ndarray,
    loss: str = "mase",
    horizons: Sequence[int] | None = None,
    season: int = 1,
    initial: int | None = None,
    times: np.ndarray | None = None,
) -> BacktestOutput:
    """Per-origin, per-horizon losses of precomputed forecasts.

    ``forecasts[variant]`` has shape ``(n_origins, H)``; row ``i`` holds the forecasts made
    from information up to (excluding) position ``origins[i]`` of ``y`` for positions
    ``origins[i], ..., origins[i] + H - 1``. ``horizons`` (1-based, default ``1..H``) selects
    which horizons become metrics. ``initial`` is the length of the first training window
    (default ``origins[0]``), used for the scale of ``mase`` and ``rmsse``. ``times`` gives
    the time stamp of each position of ``y`` (default: the position); the ``time`` column holds
    the stamp of the origin and must be strictly increasing.
    """
    if loss not in LOSS_KINDS:
        raise ValueError(f"loss must be one of {LOSS_KINDS}, got {loss!r}")
    series = _check_y(y)
    org = np.asarray(origins, dtype=np.int64)
    if org.ndim != 1 or len(org) < 2:
        raise ValueError("need at least 2 forecast origins")
    if (np.diff(org) <= 0).any():
        raise ValueError("origins must be strictly increasing")
    if len(forecasts) < 2:
        raise ValueError("need at least two variants to compare")
    first = next(iter(forecasts.values()))
    H = np.asarray(first).shape[1] if np.asarray(first).ndim == 2 else -1
    if H < 1:
        raise ValueError("forecasts must have shape (n_origins, horizon)")
    for name, f in forecasts.items():
        f = np.asarray(f, dtype=np.float64)
        if f.shape != (len(org), H):
            raise ValueError(f"forecasts[{name!r}] has shape {f.shape}, expected {(len(org), H)}")
        if not np.isfinite(f).all():
            raise ValueError(f"forecasts[{name!r}] contains NaN or infinite values")
    if org[0] < 1 or org[-1] + H > len(series):
        raise ValueError("origins + horizon must lie inside the series")
    hs = list(range(1, H + 1)) if horizons is None else [int(h) for h in horizons]
    if not hs or min(hs) < 1 or max(hs) > H or len(set(hs)) != len(hs):
        raise ValueError(f"horizons must be distinct integers in 1..{H}")
    stamps = np.arange(len(series)) if times is None else np.asarray(times)
    if len(stamps) != len(series):
        raise ValueError("times must have the same length as y")

    train = int(org[0]) if initial is None else int(initial)
    scale = 1.0
    if loss != "mae":
        scale = seasonal_scale(series[:train], season, squared=loss == "rmsse")

    idx = org[:, None] + np.arange(H)[None, :]
    actuals = series[idx]
    cols: dict[str, list[object]] = {
        k: []
        for k in ("unit_id", "pair_id", "variant", "metric", "value", "time", "origin", "horizon")
    }
    metrics = [metric_name(loss, h) for h in hs]
    origin_stamp = stamps[org]
    for name, f in forecasts.items():
        err = actuals - np.asarray(f, dtype=np.float64)
        per_obs = err**2 / scale if loss == "rmsse" else np.abs(err) / scale
        for h, metric in zip(hs, metrics, strict=True):
            col = per_obs[:, h - 1]
            n = len(org)
            cols["unit_id"] += [f"o{int(o)}" for o in org]
            cols["pair_id"] += [f"o{int(o)}" for o in org]
            cols["variant"] += [name] * n
            cols["metric"] += [metric] * n
            cols["value"] += col.tolist()
            cols["time"] += origin_stamp.tolist()
            cols["origin"] += org.tolist()
            cols["horizon"] += [h] * n
    table = build_table(cols, TimeSeries())  # fails loudly if `time` is not increasing
    return BacktestOutput(
        table=table,
        origins=org,
        forecasts={k: np.asarray(v, dtype=np.float64) for k, v in forecasts.items()},
        actuals=actuals,
        metrics=metrics,
        loss=loss,
        scale=scale,
    )


def rolling_origin_backtest(
    y: np.ndarray,
    forecasters: Mapping[str, Forecaster],
    initial: int,
    horizon: int,
    step: int = 1,
    loss: str = "mase",
    horizons: Sequence[int] | None = None,
    season: int = 1,
    times: np.ndarray | None = None,
) -> BacktestOutput:
    """Expanding-window rolling-origin backtest.

    Origins are ``initial, initial + step, ...`` while ``origin + horizon <= len(y)``. Every
    forecaster sees exactly ``y[:origin]`` and must return ``horizon`` finite values, so all
    variants are scored on identical origins and targets (the pairing the tests rely on).
    ``step = 1`` gives the dense origins for which the ``h``-step Diebold-Mariano variance
    (lags ``0..h-1``) is the right correction.
    """
    series = _check_y(y)
    if horizon < 1 or step < 1:
        raise ValueError("horizon and step must be >= 1")
    if initial < 1 or initial + horizon > len(series):
        raise ValueError("initial + horizon must not exceed the series length")
    origins = np.arange(initial, len(series) - horizon + 1, step)
    if len(origins) < 2:
        raise ValueError("series too short for at least 2 forecast origins")
    out: dict[str, np.ndarray] = {}
    for name, fc in forecasters.items():
        mat = np.empty((len(origins), horizon))
        for i, o in enumerate(origins):
            f = np.asarray(fc(series[:o], horizon), dtype=np.float64)
            if f.shape != (horizon,):
                raise ValueError(f"forecaster {name!r} returned shape {f.shape}, not ({horizon},)")
            mat[i] = f
        out[name] = mat
    return backtest_table(
        series, out, origins, loss, horizons, season, initial=initial, times=times
    )
