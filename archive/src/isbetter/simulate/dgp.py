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


def ar1_garch_returns(
    rng: np.random.Generator,
    reps: int,
    T: int,
    sharpe: float = 0.05,
    vol: float = 0.01,
    phi: float = 0.2,
    alpha: float = 0.1,
    beta: float = 0.85,
    df: float | None = 5.0,
    burn: int = 200,
) -> np.ndarray:
    """Per-period strategy returns with serial correlation and volatility clustering, (reps, T).

    ``r_t = m + x_t`` with ``x_t = phi x_{t-1} + e_t``, ``e_t = s_t z_t`` and
    ``s_t^2 = w + alpha e_{t-1}^2 + beta s_{t-1}^2`` (GARCH(1,1)); ``z_t`` is standardised
    Student-t with ``df`` degrees of freedom (normal if ``df`` is None). ``m`` and ``w`` are set
    so the unconditional standard deviation of ``r`` is ``vol`` and its true per-period Sharpe
    ratio is exactly ``sharpe``. Positive ``phi`` makes the mean of ``r`` more variable than
    iid sampling suggests, so an iid bootstrap interval for the Sharpe ratio is too narrow.
    """
    if not (0 <= alpha and 0 <= beta and alpha + beta < 1):
        raise ValueError("need alpha, beta >= 0 and alpha + beta < 1 (stationary GARCH)")
    if not -1 < phi < 1:
        raise ValueError("phi must be in (-1, 1)")
    if df is not None and df <= 4:
        raise ValueError("df must exceed 4 (finite kurtosis) or be None")
    var_e = vol**2 * (1.0 - phi**2)  # unconditional variance of the GARCH innovations
    omega = var_e * (1.0 - alpha - beta)
    scale = np.sqrt((df - 2.0) / df) if df is not None else 1.0
    out = np.empty((reps, T))
    x = np.zeros(reps)
    s2 = np.full(reps, var_e)
    e_prev = np.zeros(reps)
    for t in range(-burn, T):
        s2 = omega + alpha * e_prev**2 + beta * s2
        z = rng.standard_t(df, reps) * scale if df is not None else rng.standard_normal(reps)
        e_prev = np.sqrt(s2) * z
        x = phi * x + e_prev
        if t >= 0:
            out[:, t] = x
    return out + sharpe * vol


def paired_binary_cells(
    rng: np.random.Generator, reps: int, n: int, p_a: float, p_b: float, rho: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """Counts ``(b, c)`` of discordant items for two models scored right/wrong on ``n`` items.

    ``b`` = only A right, ``c`` = only B right. Each item is a draw from the 2x2 cell
    probabilities with marginal accuracies ``p_a`` and ``p_b`` and a phi correlation ``rho``
    between the two models' correctness (models fail on the same hard items). The true accuracy
    difference is ``p_a - p_b``. Raises if ``rho`` is infeasible for the marginals.
    """
    p11 = p_a * p_b + rho * np.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    cells = np.array([p11, p_a - p11, p_b - p11, 1 - p_a - p_b + p11])
    if (cells < -1e-12).any():
        raise ValueError(f"rho={rho} is infeasible for accuracies p_a={p_a}, p_b={p_b}")
    counts = rng.multinomial(n, np.clip(cells, 0, None) / np.clip(cells, 0, None).sum(), size=reps)
    return counts[:, 1], counts[:, 2]


def aa_paired_clustered(
    rng: np.random.Generator, reps: int, groups: int, per_group: int, icc: float = 0.5
) -> np.ndarray:
    """Per-item score differences of two equally good models, items grouped in clusters.

    Shape (reps, groups, per_group). The difference has mean zero, and items in a group share a
    component (intra-class correlation ``icc``), as when questions on one passage are all easier
    for one model. Any rejection of "no difference" is a false positive.
    """
    shared = rng.normal(0.0, np.sqrt(icc), (reps, groups, 1))
    return shared + rng.normal(0.0, np.sqrt(1 - icc), (reps, groups, per_group))
