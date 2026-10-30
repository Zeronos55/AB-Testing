"""Strategy returns as a canonical table, plus the append-only :class:`TrialRegistry`.

A trading "edge" is usually the best of many variants that were tried. The Deflated Sharpe
Ratio and the Probability of Backtest Overfitting both need to know *how many* variants were
evaluated and how spread out their Sharpe ratios were, and that number is the one people
under-report. The registry therefore logs every evaluated variant as it is evaluated, instead of
asking the analyst to remember afterwards.

* :class:`TrialRegistry`: append-only JSON Lines file; ``log`` / ``track`` (decorator) /
  ``trial`` (context manager) record a variant, ``n_trials`` and ``sharpe_variance`` summarise.
* :func:`resolve_deflation`: the inputs the DSR needs; raises :class:`MissingTrialsError`
  unless a registry or an explicit ``n_trials`` is given. It never guesses a trial count.
* :func:`returns_table`, :func:`read_returns_csv`, :func:`prices_to_returns`: turn return series
  into the long :class:`~isbetter.core.schema.CanonicalTable` (``dependence=timeseries``).

Returns are per-period (daily, monthly, ...). Sharpe ratios in the registry are per period and
not annualised, which is the scale the DSR formulas need.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from isbetter.core.dependence import TimeSeries
from isbetter.core.schema import CanonicalTable, SchemaError, build_table
from isbetter.methods.finance import return_moments

RETURN_METRIC = "return"

F = TypeVar("F", bound=Callable[..., Any])


class MissingTrialsError(ValueError):
    """The number of trials behind a selected strategy is unknown, so it cannot be deflated."""


def _clean_returns(returns: Any, what: str = "returns") -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"{what} must be a 1-D series of per-period returns")
    if len(r) < 3:
        raise ValueError(f"{what} needs at least 3 observations")
    if not np.isfinite(r).all():
        raise ValueError(f"{what} contain NaN or infinite values")
    return r


@dataclass(frozen=True)
class TrialRecord:
    """One evaluated variant. ``sharpe`` is per period (mean / std, ddof=1)."""

    name: str
    params: dict[str, Any]
    n_obs: int
    mean: float
    std: float
    sharpe: float
    skew: float
    kurtosis: float  # raw (normal = 3)
    returns_sha256: str
    logged_at: str

    @property
    def key(self) -> str:
        """Identity of a variant: its name plus its parameters."""
        return _trial_key(self.name, self.params)


def _canonical(value: Any) -> Any:
    """JSON-safe form of a trial parameter that keeps different values different.

    ``repr`` of a large numpy array is abbreviated with ``...``, so two arrays that differ only
    in the elided middle would share a key and count as one trial. Arrays are therefore stored
    as their shape, dtype and a SHA-256 of their bytes; numpy scalars become Python scalars.
    """
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        meta = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest}
        return {"ndarray": meta}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    return value


def _trial_key(name: str, params: Mapping[str, Any]) -> str:
    return name + json.dumps(params, sort_keys=True, default=repr)


def _describe(name: str, params: Mapping[str, Any], returns: np.ndarray) -> TrialRecord:
    try:
        m = return_moments(returns)
    except ValueError as exc:
        raise ValueError(f"trial {name!r}: {exc}") from exc
    return TrialRecord(
        name=name,
        params={str(k): _canonical(v) for k, v in params.items()},
        n_obs=m.n,
        mean=m.mean,
        std=m.std,
        sharpe=m.sharpe,
        skew=m.skew,
        kurtosis=m.kurtosis,
        returns_sha256=hashlib.sha256(np.ascontiguousarray(returns).tobytes()).hexdigest(),
        logged_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


@dataclass
class _OpenTrial:
    """Handle yielded by :meth:`TrialRegistry.trial`; set ``returns`` before the block ends."""

    name: str
    params: dict[str, Any]
    returns: Any = None


@dataclass(frozen=True)
class DeflationInputs:
    """What the Deflated Sharpe Ratio needs to know about the search that found the strategy."""

    n_trials: int
    sr_variance: float
    source: str


@dataclass
class TrialRegistry:
    """Append-only JSON Lines log of every strategy variant that was evaluated.

    Each ``log`` appends one line and never rewrites the file. The same variant (same name and
    parameters) logged twice is one trial, not two, so re-running a script does not inflate the
    trial count; the latest record of a variant is the one that counts.
    """

    path: Path
    _cache: list[TrialRecord] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    # -- writing ---------------------------------------------------------------------------
    def log(self, name: str, returns: Any, params: Mapping[str, Any] | None = None) -> TrialRecord:
        """Record the per-period returns of one variant and return its summary record."""
        rec = _describe(name, params or {}, _clean_returns(returns, f"returns of {name!r}"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), default=repr) + "\n")
        self._cache = None
        return rec

    def track(self, fn: F | None = None, *, name: str | None = None) -> Any:
        """Decorator: log the returns of every call of ``fn(**params) -> returns``.

        The call's arguments become the trial parameters. The wrapped function still returns the
        returns, so the registry cannot be forgotten by the code that evaluates variants.
        """

        def decorate(f: F) -> F:
            sig = inspect.signature(f)
            label = name or f.__name__

            @functools.wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                out = f(*args, **kwargs)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                self.log(label, out, dict(bound.arguments))
                return out

            return wrapper  # type: ignore[return-value]

        return decorate(fn) if fn is not None else decorate

    @contextmanager
    def trial(self, name: str, **params: Any) -> Iterator[_OpenTrial]:
        """Context manager: assign ``t.returns`` inside the block; it is logged on a clean exit.

        A block that raises logs nothing; a block that ends without setting ``returns`` is an
        error, because a variant that was evaluated but not logged is exactly the bug this guards.
        """
        handle = _OpenTrial(name, dict(params))
        yield handle
        if handle.returns is None:
            raise ValueError(f"trial {name!r} finished without setting `returns`")
        self.log(name, handle.returns, handle.params)

    # -- reading ---------------------------------------------------------------------------
    def records(self) -> list[TrialRecord]:
        """Every line of the log, oldest first (including repeated evaluations)."""
        if self._cache is not None:
            return list(self._cache)
        out: list[TrialRecord] = []
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        out.append(TrialRecord(**json.loads(line)))
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"{self.path}: line {i} is not a trial record") from exc
        self._cache = out
        return list(out)

    def trials(self) -> dict[str, TrialRecord]:
        """Distinct variants (latest record wins), keyed by name + parameters."""
        return {r.key: r for r in self.records()}

    def n_trials(self) -> int:
        return len(self.trials())

    def sharpe_variance(self) -> float:
        """Sample variance (ddof=1) of the per-period Sharpe ratios of the distinct trials."""
        sr = np.array([t.sharpe for t in self.trials().values()])
        if len(sr) < 2:
            raise MissingTrialsError(
                f"{self.path} holds {len(sr)} trial(s); the variance of the trials' Sharpe "
                "ratios needs at least 2"
            )
        return float(sr.var(ddof=1))

    def deflation_inputs(self) -> DeflationInputs:
        return DeflationInputs(self.n_trials(), self.sharpe_variance(), f"registry {self.path}")


def resolve_deflation(
    registry: TrialRegistry | None = None,
    n_trials: int | None = None,
    sr_variance: float | None = None,
) -> DeflationInputs:
    """Inputs for the Deflated Sharpe Ratio; refuses to guess.

    With a ``registry`` the trial count and the Sharpe variance come from the log. An explicit
    ``n_trials`` (e.g. variants tried outside the registry) overrides the registry's count, and
    an explicit ``sr_variance`` overrides its variance. Without a registry, both are required.
    """
    if registry is None and n_trials is None:
        raise MissingTrialsError(
            "the Deflated Sharpe Ratio needs to know how many strategy variants were tried: "
            "pass a TrialRegistry (every evaluated variant logged) or an explicit n_trials. "
            "Refusing to assume a single trial, which would hide the selection bias."
        )
    if n_trials is not None and n_trials < 1:
        raise MissingTrialsError(f"n_trials must be >= 1, got {n_trials}")
    if registry is None:
        if sr_variance is None:
            raise MissingTrialsError(
                "an explicit n_trials also needs sr_variance, the variance of the trials' "
                "per-period Sharpe ratios (a registry would supply it)"
            )
        assert n_trials is not None
        return DeflationInputs(int(n_trials), float(sr_variance), "explicit")
    count = registry.n_trials() if n_trials is None else int(n_trials)
    var = registry.sharpe_variance() if sr_variance is None else float(sr_variance)
    parts = [
        "explicit n_trials" if n_trials is not None else "registry count",
        "explicit sr_variance" if sr_variance is not None else "registry variance",
    ]
    return DeflationInputs(count, var, f"{registry.path} ({', '.join(parts)})")


# ---------------------------------------------------------------------------------- tables


@dataclass(frozen=True)
class ReturnsPanel:
    """Aligned per-period returns of several strategies: ``matrix`` is ``(T, N)``."""

    names: list[str]
    matrix: np.ndarray
    times: np.ndarray

    def series(self, name: str) -> np.ndarray:
        return np.asarray(self.matrix[:, self.names.index(name)])


def make_panel(returns: Mapping[str, Any], times: Any | None = None) -> ReturnsPanel:
    """Stack named, equally long return series into a :class:`ReturnsPanel`."""
    if len(returns) < 1:
        raise ValueError("need at least one return series")
    cols = {str(k): _clean_returns(v, f"returns of {k!r}") for k, v in returns.items()}
    lengths = {len(v) for v in cols.values()}
    if len(lengths) != 1:
        raise ValueError(f"return series have different lengths: {sorted(lengths)}")
    T = lengths.pop()
    stamps = np.arange(T) if times is None else np.asarray(times)
    if len(stamps) != T:
        raise ValueError("times must have one entry per period")
    return ReturnsPanel(list(cols), np.column_stack(list(cols.values())), stamps)


def _numeric_times(times: np.ndarray) -> np.ndarray:
    """Numeric time stamps: numbers pass through, ISO dates / datetimes become seconds."""
    if np.issubdtype(times.dtype, np.number):
        return times
    try:
        return np.asarray(times.astype("datetime64[s]").astype(np.int64))
    except (ValueError, TypeError) as exc:
        raise SchemaError(
            "time stamps must be numbers or ISO dates such as 2024-01-31; "
            f"could not read {times[0]!r}"
        ) from exc


def returns_table(
    returns: Mapping[str, Any], times: Any | None = None, metric: str = RETURN_METRIC
) -> CanonicalTable:
    """Long table of per-period returns, validated for ``dependence=timeseries``.

    One row per (strategy, period); ``time`` must be strictly increasing, and the rows of
    different strategies share ``pair_id`` (the period), so any two can be compared pairwise.
    """
    panel = make_panel(returns, times)
    T = len(panel.times)
    stamps = _numeric_times(panel.times)
    cols: dict[str, list[Any]] = {
        k: [] for k in ("unit_id", "pair_id", "variant", "metric", "value", "time")
    }
    ids = [f"t{i}" for i in range(T)]
    for j, name in enumerate(panel.names):
        cols["unit_id"] += ids
        cols["pair_id"] += ids
        cols["variant"] += [name] * T
        cols["metric"] += [metric] * T
        cols["value"] += panel.matrix[:, j].tolist()
        cols["time"] += stamps.tolist()
    return build_table(cols, TimeSeries())


def prices_to_returns(prices: Any, kind: str = "simple") -> np.ndarray:
    """Per-period returns from a price series (``simple``: ``p_t / p_{t-1} - 1``; ``log``)."""
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("prices must be a 1-D series with at least 2 observations")
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("prices must be finite and strictly positive")
    if kind == "simple":
        out: np.ndarray = p[1:] / p[:-1] - 1.0
    elif kind == "log":
        out = np.diff(np.log(p))
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    return out


def read_returns_csv(path: str | Path, time_col: str = "date") -> ReturnsPanel:
    """Wide CSV (one ``time_col`` column, one column of per-period returns per strategy)."""
    import csv

    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            names = reader.fieldnames or []
            rows = list(reader)
    except OSError as exc:
        raise SchemaError(f"cannot read returns file {path}: {exc}") from exc
    if time_col not in names:
        raise SchemaError(f"returns file {path} has no '{time_col}' column")
    strategies = [n for n in names if n != time_col]
    if not strategies or not rows:
        raise SchemaError(f"returns file {path} holds no strategy columns or no rows")
    try:
        data = {n: [float(r[n]) for r in rows] for n in strategies}
    except ValueError as exc:
        raise SchemaError(f"returns file {path} has a non-numeric return: {exc}") from exc
    return make_panel(data, [r[time_col] for r in rows])
