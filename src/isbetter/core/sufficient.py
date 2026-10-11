"""Sufficient statistics: a few sums per arm that are enough for every mean-based test.

Large A/B tables (Criteo has about 14 million rows) are aggregated in the database to these
sums, so memory use is independent of the row count. All methods that accept a
:class:`SufficientStats` also work from raw arrays through :meth:`SufficientStats.from_arrays`,
which guarantees the unit-level and aggregated paths agree.

Raw sums lose about ``log10((mean / sd)^2)`` digits of precision in the variance; that is
negligible for experiment metrics but would matter for values like epoch timestamps.

``y`` is the outcome. The optional second variable ``x`` is whatever the analysis pairs with
``y`` per unit: a pre-period covariate (CUPED, regression adjustment) or the denominator of a
ratio metric (delta method).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_OPTIONAL = ("sum_x", "sum_x_sq", "sum_xy")


@dataclass(frozen=True)
class SufficientStats:
    """Moments of one arm: n, sum(y), sum(y^2) and optionally those of a paired x and sum(xy)."""

    n: int
    sum: float
    sum_sq: float
    sum_x: float | None = None
    sum_x_sq: float | None = None
    sum_xy: float | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("SufficientStats needs n >= 1")
        given = [getattr(self, k) is not None for k in _OPTIONAL]
        if any(given) and not all(given):
            raise ValueError("sum_x, sum_x_sq and sum_xy must be given together")
        for name in ("sum", "sum_sq", *_OPTIONAL):
            v = getattr(self, name)
            if v is not None and not np.isfinite(v):
                raise ValueError(f"{name} is not finite")
        if self.sum_sq < 0 or (self.sum_x_sq is not None and self.sum_x_sq < 0):
            raise ValueError("sums of squares must be non-negative")
        _check_moment(self.n, self.sum, self.sum_sq, "y")
        if self.sum_x is not None:
            assert self.sum_x_sq is not None and self.sum_xy is not None
            _check_moment(self.n, self.sum_x, self.sum_x_sq, "x")

    @property
    def has_x(self) -> bool:
        return self.sum_x is not None

    @property
    def mean(self) -> float:
        return self.sum / self.n

    @property
    def var(self) -> float:
        """Sample variance of y (ddof=1); 0 when n == 1 is rejected by callers."""
        return _var(self.n, self.sum, self.sum_sq)

    @property
    def mean_x(self) -> float:
        return self._x(self.sum_x) / self.n

    @property
    def var_x(self) -> float:
        return _var(self.n, self._x(self.sum_x), self._x(self.sum_x_sq))

    @property
    def cov_xy(self) -> float:
        """Sample covariance of x and y (ddof=1)."""
        if self.n < 2:
            return float("nan")
        return (self._x(self.sum_xy) - self._x(self.sum_x) * self.sum / self.n) / (self.n - 1)

    @staticmethod
    def _x(v: float | None) -> float:
        if v is None:
            raise ValueError("this SufficientStats has no x moments (covariate or denominator)")
        return v

    def __add__(self, other: SufficientStats) -> SufficientStats:
        """Merge two shards of the same arm (e.g. chunks of a large file)."""
        if self.has_x != other.has_x:
            raise ValueError("cannot merge stats with and without x moments")
        if not self.has_x:
            return SufficientStats(
                self.n + other.n, self.sum + other.sum, self.sum_sq + other.sum_sq
            )
        return SufficientStats(
            self.n + other.n,
            self.sum + other.sum,
            self.sum_sq + other.sum_sq,
            self._x(self.sum_x) + self._x(other.sum_x),
            self._x(self.sum_x_sq) + self._x(other.sum_x_sq),
            self._x(self.sum_xy) + self._x(other.sum_xy),
        )

    @classmethod
    def from_arrays(cls, y: np.ndarray, x: np.ndarray | None = None) -> SufficientStats:
        yy = np.asarray(y, dtype=np.float64)
        if yy.ndim != 1 or len(yy) == 0:
            raise ValueError("y must be a non-empty 1-D array")
        if not np.isfinite(yy).all():
            raise ValueError("y contains NaN or infinite values")
        if x is None:
            return cls(len(yy), float(yy.sum()), float(np.dot(yy, yy)))
        xx = np.asarray(x, dtype=np.float64)
        if xx.shape != yy.shape:
            raise ValueError("x and y must have the same length")
        if not np.isfinite(xx).all():
            raise ValueError("x contains NaN or infinite values")
        return cls(
            len(yy),
            float(yy.sum()),
            float(np.dot(yy, yy)),
            float(xx.sum()),
            float(np.dot(xx, xx)),
            float(np.dot(xx, yy)),
        )


def _var(n: int, s: float, ss: float) -> float:
    if n < 2:
        return float("nan")
    return max((ss - s * s / n) / (n - 1), 0.0)


def _check_moment(n: int, s: float, ss: float, name: str) -> None:
    """Reject sums that cannot come from real data (variance clearly negative)."""
    raw = ss - s * s / n
    if raw < -1e-9 * max(ss, 1.0):
        raise ValueError(f"inconsistent moments for {name}: sum_sq < sum^2 / n")


@dataclass(frozen=True)
class AggregatedTable:
    """Aggregated A/B data: sufficient statistics per (variant, metric).

    ``units`` holds the number of randomisation units per variant, used by the sample-ratio
    check. Only dependence=iid can be analysed from aggregates.
    """

    stats: dict[tuple[str, str], SufficientStats]
    units: dict[str, int] = field(default_factory=dict)

    def get(self, variant: str, metric: str) -> SufficientStats:
        try:
            return self.stats[(variant, metric)]
        except KeyError:
            raise KeyError(
                f"no aggregated data for variant '{variant}', metric '{metric}'"
            ) from None

    def variants(self) -> list[str]:
        return sorted({v for v, _ in self.stats})

    def metrics(self) -> list[str]:
        return sorted({m for _, m in self.stats})
