"""Dependence declarations: the first thing a user states about their data.

The declaration decides which statistical method is valid (see ``core/plan.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Iid:
    """Units are independent; the two variants are measured on different units."""

    kind = "iid"


@dataclass(frozen=True)
class Paired:
    """Both variants are measured on the same units, linked by ``pair_id``."""

    kind = "paired"


@dataclass(frozen=True)
class Clustered:
    """Units sit in correlated groups (users, passages); cluster_col names the group."""

    cluster_col: str = "cluster_id"
    kind = "clustered"


@dataclass(frozen=True)
class CVOverlap:
    """Scores come from repeated k-fold CV, so training sets overlap."""

    n_train: int
    n_test: int
    repeats: int
    folds: int
    kind = "cv_overlap"

    def __post_init__(self) -> None:
        for name in ("n_train", "n_test", "repeats", "folds"):
            if getattr(self, name) < 1:
                raise ValueError(f"CVOverlap.{name} must be >= 1")


@dataclass(frozen=True)
class TimeSeries:
    """Observations are ordered in time; losses are autocorrelated."""

    block_len: int | None = None
    hac_lags: int | None = None
    horizon: int = 1
    kind = "timeseries"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("TimeSeries.horizon must be >= 1")


Dependence = Iid | Paired | Clustered | CVOverlap | TimeSeries
