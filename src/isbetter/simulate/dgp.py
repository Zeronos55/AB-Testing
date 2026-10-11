"""Data-generating processes for calibration. All take a Generator and return arrays."""

from __future__ import annotations

import numpy as np


def aa_independent(
    rng: np.random.Generator, reps: int, n: int, skew: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Two independent samples from the SAME distribution (a null world), shape (reps, n)."""
    if skew:
        return rng.exponential(1.0, (reps, n)), rng.exponential(1.0, (reps, n))
    return rng.normal(0.0, 1.0, (reps, n)), rng.normal(0.0, 1.0, (reps, n))


def aa_paired(
    rng: np.random.Generator, reps: int, n: int, unit_sd: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Paired null world: shared unit effect plus independent noise; no true difference."""
    unit = rng.normal(0.0, unit_sd, (reps, n))
    return unit + rng.normal(0.0, 1.0, (reps, n)), unit + rng.normal(0.0, 1.0, (reps, n))


def cv_null_diffs(
    rng: np.random.Generator, reps: int, n: int = 100, k: int = 10, r: int = 1
) -> np.ndarray:
    """Per-fold MSE differences between two learners that are equally good by symmetry.

    ``y = x1 + x2 + noise``; learner A regresses y on x1 alone, learner B on x2 alone, so
    their expected test error is identical (the null). Scores come from ``r`` repeats of
    ``k``-fold CV on the same data. Returns an array of shape (reps, r * k) of
    ``mse_A - mse_B`` per fold.
    """
    x1, x2 = rng.normal(size=(reps, n)), rng.normal(size=(reps, n))
    y = x1 + x2 + rng.normal(size=(reps, n))
    out = np.empty((reps, r * k))
    col = 0
    for _ in range(r):
        perm = rng.permutation(n)
        for te in np.array_split(perm, k):
            tr = np.setdiff1d(perm, te, assume_unique=True)
            mse = []
            for x in (x1, x2):
                xt, yt = x[:, tr], y[:, tr]
                xm, ym = xt.mean(1, keepdims=True), yt.mean(1, keepdims=True)
                slope = ((xt - xm) * (yt - ym)).sum(1, keepdims=True) / ((xt - xm) ** 2).sum(
                    1, keepdims=True
                )
                pred = ym + slope * (x[:, te] - xm)
                mse.append(((y[:, te] - pred) ** 2).mean(1))
            out[:, col] = mse[0] - mse[1]
            col += 1
    return out


def aa_covariate(
    rng: np.random.Generator,
    reps: int,
    n: int,
    rho: float = 0.7,
    skew: bool = True,
    curvature: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Null A/B world with a pre-experiment covariate, shape (reps, n, 2) per arm.

    Channel 0 is the outcome ``y`` and channel 1 the covariate ``x``. Both arms are drawn from
    the same distribution (no treatment effect). ``corr(x, y) = rho`` for the linear part; with
    ``skew`` the outcome noise is a centred exponential, so the normal approximation is only
    approximate. ``curvature > 0`` adds a quadratic term in the covariate that a linear
    adjustment cannot remove (a misspecified-adjustment stress test).
    """

    def arm() -> np.ndarray:
        zx = rng.normal(size=(reps, n))
        e = rng.exponential(1.0, (reps, n)) - 1.0 if skew else rng.normal(size=(reps, n))
        quad = curvature * 4 * rho * (zx**2 - 1) / np.sqrt(2)
        y = 20 + 4 * rho * zx + quad + 4 * np.sqrt(1 - rho * rho) * e
        return np.stack([y, 50 + 10 * zx], axis=-1)

    return arm(), arm()


def aa_binary(
    rng: np.random.Generator, reps: int, n: int, p: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Two independent Bernoulli(p) arms (a null world), shape (reps, n), dtype int8."""
    return (
        (rng.random((reps, n)) < p).astype(np.int8),
        (rng.random((reps, n)) < p).astype(np.int8),
    )


def aa_clustered(
    rng: np.random.Generator, reps: int, n_users: int, sessions: int = 5, icc: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Null A/B world randomised by USER with several sessions per user.

    Returns two arrays of shape (reps, n_users, sessions): sessions of one user share a user
    effect, so rows are correlated within user. The intra-class correlation is ``icc``. Users
    are disjoint across arms (each user sees one variant) and there is no treatment effect.
    """

    def arm() -> np.ndarray:
        user = rng.normal(0.0, np.sqrt(icc), (reps, n_users, 1))
        return user + rng.normal(0.0, np.sqrt(1 - icc), (reps, n_users, sessions))

    return arm(), arm()


def dm_null_losses(
    rng: np.random.Generator, reps: int, T: int, h: int, rho: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Squared-error losses of two equally accurate ``h``-step forecasters, shape (reps, T).

    The ``h``-step error at origin ``t`` is the sum of the ``h`` shocks ``eps_t .. eps_{t+h-1}``
    (an MA(h - 1) process), so errors at neighbouring origins overlap and the loss
    differential is autocorrelated up to lag ``h - 1``. The two forecasters have shock
    correlation ``rho`` and identical expected loss ``h``, so ``E[L_A - L_B] = 0``.
    """
    ea = rng.standard_normal((reps, T + h - 1))
    eb = rho * ea + np.sqrt(1 - rho * rho) * rng.standard_normal((reps, T + h - 1))
    out = []
    for e in (ea, eb):
        c = np.concatenate([np.zeros((reps, 1)), np.cumsum(e, axis=1)], axis=1)
        out.append((c[:, h : h + T] - c[:, :T]) ** 2)
    return out[0], out[1]
