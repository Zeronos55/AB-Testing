"""Is a Sharpe ratio edge or luck? PSR, DSR and the minimum track record length.

All Sharpe ratios here are **per period** (mean / standard deviation of the periodic returns,
not annualised); ``T`` is the number of return observations. Annualising multiplies the Sharpe
ratio by ``sqrt(periods_per_year)`` and must be undone before using these formulas, otherwise
the probabilities saturate at 1.

* :func:`probabilistic_sharpe`: PSR, the probability that the true Sharpe ratio exceeds a
  benchmark, accounting for the sample length, skewness and kurtosis of the returns
  (Bailey & Lopez de Prado, 2012, *The Sharpe Ratio Efficient Frontier*, J. Risk 15).
* :func:`deflated_sharpe`: DSR, the PSR against the Sharpe ratio the best of ``N`` unskilled
  trials would reach by luck (Bailey & Lopez de Prado, 2014, *The Deflated Sharpe Ratio*,
  J. Portfolio Management 40).
* :func:`min_track_record`: how many observations are needed before the PSR clears a
  confidence level.
* :func:`sharpe_block_bootstrap`: a Sharpe ratio (or Sharpe ratio difference) interval that
  survives serial dependence in returns, by resampling blocks.
* :func:`spa_vs_buy_and_hold` / :func:`stepm_vs_buy_and_hold`: many strategies against a
  benchmark, wrapping the ``arch`` tests in :mod:`isbetter.methods.superior_predictive`.

Formulas (``gamma3`` skewness, ``gamma4`` raw kurtosis, normal = 3)::

    PSR(SR*)  = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - gamma3 SR + (gamma4 - 1)/4 SR^2) )
    SR0       = sqrt(V[SR_trials]) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e))),  g = Euler
    DSR       = PSR(SR0)
    MinTRL    = 1 + (1 - gamma3 SR + (gamma4 - 1)/4 SR^2) (z_alpha / (SR - SR*))^2

``V[SR_trials]`` is the variance of the per-period Sharpe ratios of the ``N`` trials that were
evaluated, which is why the number of trials must come from a registry
(:class:`isbetter.adapters.strategy_returns.TrialRegistry`) or be stated explicitly.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from isbetter.core.result import MethodResult
from isbetter.methods.forecasting import suggest_block_length
from isbetter.methods.resampling import _block, _block_indices
from isbetter.methods.superior_predictive import spa_test, stepm

EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class ReturnMoments:
    """Per-period summary of a return series (kurtosis is raw: normal = 3)."""

    n: int
    mean: float
    std: float
    sharpe: float
    skew: float
    kurtosis: float


def return_moments(returns: Any) -> ReturnMoments:
    """Mean, standard deviation (ddof=1), Sharpe ratio, skewness and raw kurtosis of returns.

    Skewness and kurtosis are the plug-in (biased) sample moments used by Bailey & Lopez de
    Prado.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1 or len(r) < 3:
        raise ValueError("returns must be a 1-D series with at least 3 observations")
    if not np.isfinite(r).all():
        raise ValueError("returns contain NaN or infinite values")
    std = float(r.std(ddof=1))
    mean = float(r.mean())
    if std <= 1e-12 * max(abs(mean), float(np.abs(r).max())):
        raise ValueError("returns are constant: the Sharpe ratio is undefined")
    return ReturnMoments(
        n=len(r),
        mean=mean,
        std=std,
        sharpe=mean / std,
        skew=float(stats.skew(r, bias=True)),
        kurtosis=float(stats.kurtosis(r, fisher=False, bias=True)),
    )


def annualize_sharpe(sharpe: float, periods_per_year: float) -> float:
    """Per-period Sharpe ratio to annual terms (valid for serially uncorrelated returns)."""
    return sharpe * math.sqrt(periods_per_year)


def sharpe_variance_factor(sharpe: float, skew: float, kurtosis: float) -> float:
    """``1 - skew SR + (kurtosis - 1)/4 SR^2``: T times the asymptotic variance of the SR.

    Equals ``1 + SR^2/2`` for normal returns. It is positive for any real distribution; a
    non-positive value means the supplied moments are inconsistent.
    """
    factor = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if not factor > 0:
        raise ValueError(
            f"inconsistent moments: 1 - skew*SR + (kurtosis-1)/4*SR^2 = {factor:.4g} <= 0 "
            f"(SR={sharpe:.4g}, skew={skew:.4g}, kurtosis={kurtosis:.4g}; is kurtosis excess "
            "rather than raw?)"
        )
    return factor


def _check_n(n_obs: int) -> None:
    if n_obs < 2:
        raise ValueError("need at least 2 observations")


def psr(
    sharpe: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """Probabilistic Sharpe Ratio ``P(true SR > sr_benchmark)`` from summary statistics."""
    _check_n(n_obs)
    factor = sharpe_variance_factor(sharpe, skew, kurtosis)
    z = (sharpe - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(factor)
    return float(stats.norm.cdf(z))


def probabilistic_sharpe(
    returns: Any, sr_benchmark: float = 0.0, alpha: float = 0.05
) -> MethodResult:
    """PSR of a return series against ``sr_benchmark`` (per period; default: no skill).

    ``estimate`` is the per-period Sharpe ratio with a normal-approximation interval of level
    ``1 - alpha`` from the same asymptotic variance; ``p_value`` is the one-sided
    ``1 - PSR`` for H0: true SR <= ``sr_benchmark``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    m = return_moments(returns)
    factor = sharpe_variance_factor(m.sharpe, m.skew, m.kurtosis)
    se = math.sqrt(factor / (m.n - 1))
    z = (m.sharpe - sr_benchmark) / se
    value = float(stats.norm.cdf(z))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return MethodResult(
        method="probabilistic_sharpe",
        estimate=m.sharpe,
        ci_low=m.sharpe - crit * se,
        ci_high=m.sharpe + crit * se,
        ci_level=1 - alpha,
        p_value=1.0 - value,
        statistic=float(z),
        df=None,
        n_effective=float(m.n),
        rationale=(
            f"Probabilistic Sharpe Ratio: probability that the true Sharpe ratio exceeds "
            f"{sr_benchmark:g}, given {m.n} observations with skewness {m.skew:.2f} and "
            f"kurtosis {m.kurtosis:.2f}. Skew and fat tails widen the Sharpe ratio's "
            "uncertainty, so a plain t-statistic on the mean would be too confident."
        ),
        diagnostics={
            "psr": value,
            "sr_benchmark": sr_benchmark,
            "skew": m.skew,
            "kurtosis": m.kurtosis,
            "se": se,
            "n_obs": m.n,
        },
    )


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Sharpe ratio the best of ``n_trials`` unskilled trials is expected to reach by luck.

    Bailey & Lopez de Prado (2014), eq. for ``E[max SR]`` (the "False Strategy Theorem"
    approximation): ``sqrt(V) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))``. A single
    trial has no selection effect, so the threshold is 0.
    """
    if n_trials < 1 or int(n_trials) != n_trials:
        raise ValueError("n_trials must be a positive integer")
    if not sr_variance >= 0:
        raise ValueError("sr_variance must be >= 0")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    z = (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n) + EULER_GAMMA * stats.norm.ppf(
        1.0 - 1.0 / (n * math.e)
    )
    return float(math.sqrt(sr_variance) * z)


def deflated_sharpe(
    returns: Any, *, n_trials: int, sr_variance: float, alpha: float = 0.05
) -> MethodResult:
    """Deflated Sharpe Ratio: the PSR against the best-of-``n_trials`` luck threshold.

    ``n_trials`` and ``sr_variance`` (variance of the trials' per-period Sharpe ratios) are
    required keywords with no default: the number of variants tried cannot be assumed. Take
    both from a :class:`~isbetter.adapters.strategy_returns.TrialRegistry` through
    :func:`~isbetter.adapters.strategy_returns.resolve_deflation`. ``p_value`` is ``1 - DSR``.
    """
    if n_trials is None or sr_variance is None:
        raise ValueError("deflated_sharpe needs n_trials and sr_variance; see resolve_deflation")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    base = probabilistic_sharpe(returns, sr_benchmark=sr0, alpha=alpha)
    zero = psr(
        base.estimate, 0.0, int(base.n_effective), base.diagnostics["skew"],
        base.diagnostics["kurtosis"],
    )  # fmt: skip
    value = base.diagnostics["psr"]
    return MethodResult(
        method="deflated_sharpe",
        estimate=base.estimate,
        ci_low=base.ci_low,
        ci_high=base.ci_high,
        ci_level=base.ci_level,
        p_value=1.0 - value,
        statistic=base.statistic,
        df=None,
        n_effective=base.n_effective,
        rationale=(
            f"Deflated Sharpe Ratio: this strategy is the best of {n_trials} variants that were "
            f"tried, so its Sharpe ratio is compared with {sr0:.4f} per period (what the best of "
            f"{n_trials} unskilled variants reaches by luck, given the Sharpe spread "
            f"{math.sqrt(sr_variance):.4f} across variants), not with 0. DSR is the probability "
            "that its true Sharpe ratio exceeds that threshold."
        ),
        diagnostics={
            "dsr": value,
            "sr_star": sr0,
            "n_trials": int(n_trials),
            "sr_variance": float(sr_variance),
            "psr_vs_zero": zero,
            "skew": base.diagnostics["skew"],
            "kurtosis": base.diagnostics["kurtosis"],
            "se": base.diagnostics["se"],
            "n_obs": base.diagnostics["n_obs"],
        },
    )


def min_track_record_length(
    sharpe: float,
    sr_benchmark: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    alpha: float = 0.05,
) -> float:
    """Observations needed for ``PSR(sr_benchmark) >= 1 - alpha`` (``inf`` if SR <= benchmark).

    ``1 + (1 - skew SR + (kurtosis - 1)/4 SR^2) (z_{1-alpha} / (SR - SR*))^2`` with the
    one-sided normal quantile. At exactly this length the PSR equals ``1 - alpha``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    factor = sharpe_variance_factor(sharpe, skew, kurtosis)
    if sharpe <= sr_benchmark:
        return math.inf
    z = float(stats.norm.ppf(1.0 - alpha))
    return 1.0 + factor * (z / (sharpe - sr_benchmark)) ** 2


def min_track_record(returns: Any, sr_benchmark: float = 0.0, alpha: float = 0.05) -> MethodResult:
    """MinTRL of a return series: ``estimate`` is the number of observations required.

    ``diagnostics['enough']`` says whether the observed track record is already that long.
    """
    m = return_moments(returns)
    need = min_track_record_length(m.sharpe, sr_benchmark, m.skew, m.kurtosis, alpha)
    enough = m.n >= need
    if math.isinf(need):
        verdict = (
            f"the observed Sharpe ratio {m.sharpe:.4f} does not exceed the benchmark "
            f"{sr_benchmark:g}, so no track record length would establish it"
        )
    else:
        verdict = (
            f"{need:.0f} observations are needed and {m.n} are available "
            f"({'enough' if enough else 'not enough'})"
        )
    return MethodResult(
        method="min_track_record_length",
        estimate=float(need),
        ci_low=None,
        ci_high=None,
        ci_level=None,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(m.n),
        rationale=(
            f"Minimum track record length: observations before the Sharpe ratio is above "
            f"{sr_benchmark:g} at {1 - alpha:.0%} confidence, given skewness {m.skew:.2f} and "
            f"kurtosis {m.kurtosis:.2f}: {verdict}."
        ),
        diagnostics={
            "sharpe": m.sharpe,
            "sr_benchmark": sr_benchmark,
            "skew": m.skew,
            "kurtosis": m.kurtosis,
            "n_obs": m.n,
            "enough": bool(enough),
            "alpha": alpha,
        },
    )


# ------------------------------------------------------------ block-bootstrap Sharpe interval


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    """Per-row Sharpe ratio (ddof=1) of a ``(rows, n)`` matrix."""
    sd = x.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(x.mean(axis=1) / sd)


def _return_block_length(series: list[np.ndarray], kind: str) -> tuple[int, str]:
    """Politis-White block length for Sharpe ratios of ``series``.

    The Sharpe ratio depends on the mean *and* the variance, and returns are often
    uncorrelated while their squares are strongly correlated (volatility clustering). The
    length is therefore the longest suggested for the returns and for their squared deviations.
    """
    n = len(series[0])
    best, source = 1, "politis_white"
    for r in series:
        for x in (r, (r - r.mean()) ** 2):
            sug = suggest_block_length(x)
            best = max(best, sug.integer(kind, n))
            if sug.source != "politis_white":
                source = sug.source
    return best, source


def sharpe_block_bootstrap(
    returns: Any,
    rng: np.random.Generator,
    benchmark: Any | None = None,
    block_len: int | None = None,
    kind: str = "stationary",
    n_resamples: int = 4_999,
    alpha: float = 0.05,
    batch: int = 200,
) -> MethodResult:
    """Percentile block-bootstrap interval for a Sharpe ratio, or a Sharpe ratio difference.

    Without ``benchmark`` the statistic is the per-period Sharpe ratio of ``returns``. With a
    benchmark (for example buy-and-hold) returns and benchmark are resampled with the *same*
    block indices, which keeps their contemporaneous correlation, and the statistic is
    ``SR(returns) - SR(benchmark)``. Resampling blocks keeps the serial dependence of the
    series inside each block; an iid bootstrap (``block_len=1``) treats periods as independent
    and is too narrow whenever returns are autocorrelated.

    ``kind`` is ``stationary`` (geometric block lengths, mean ``block_len``) or ``circular``.
    Without ``block_len`` it is chosen automatically (see :func:`_return_block_length`) and the
    source is in ``diagnostics['block_len_source']``. Memory is O(batch * T).
    """
    if kind not in ("stationary", "circular"):
        raise ValueError("kind must be 'stationary' or 'circular'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    r = np.asarray(returns, dtype=np.float64)
    series = [r]
    if benchmark is not None:
        b = np.asarray(benchmark, dtype=np.float64)
        if b.shape != r.shape:
            raise ValueError("benchmark must have the same length as returns")
        series.append(b)
    for x in series:
        return_moments(x)  # validates: 1-D, finite, not constant, at least 3 observations
    n = len(r)
    if block_len is None:
        length, source = _return_block_length(series, kind)
    else:
        if not 1 <= block_len <= n:
            raise ValueError(f"block_len must be between 1 and the number of periods ({n})")
        length, source = int(block_len), "user"

    def stat(rows: np.ndarray) -> np.ndarray:
        sr = _sharpe_rows(r[rows])
        return sr if benchmark is None else sr - _sharpe_rows(series[1][rows])

    est = float(return_moments(r).sharpe)
    if benchmark is not None:
        est -= float(return_moments(series[1]).sharpe)
    draws = np.empty(n_resamples)
    step = _block(batch, n)
    for start in range(0, n_resamples, step):
        stop = min(start + step, n_resamples)
        draws[start:stop] = stat(_block_indices(kind, n, length, stop - start, rng))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    what = "Sharpe ratio" if benchmark is None else "Sharpe ratio difference versus the benchmark"
    how = "iid bootstrap (block length 1)" if length == 1 else f"{kind} block bootstrap"
    return MethodResult(
        method=f"{kind}_block_bootstrap_sharpe" if length > 1 else "iid_bootstrap_sharpe",
        estimate=est,
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=1 - alpha,
        p_value=None,
        statistic=None,
        df=None,
        n_effective=float(n),
        rationale=(
            f"Percentile {how} interval for the {what} ({n_resamples} resamples, mean block "
            f"length {length}, {source}): resampling blocks keeps the serial dependence of "
            "returns and of their volatility, which an iid bootstrap ignores."
        ),
        diagnostics={
            "se": float(draws.std(ddof=1)),
            "block_len": length,
            "block_len_source": source,
            "n_resamples": n_resamples,
            "kind": kind,
            "versus_benchmark": benchmark is not None,
        },
    )


# ------------------------------------------------ many strategies against buy-and-hold


def _loss_inputs(
    returns: Mapping[str, Any], benchmark: Any
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Losses (negative returns) so that a better strategy has a *lower* loss than the benchmark."""
    bench = -np.asarray(benchmark, dtype=np.float64)
    if len(returns) < 1:
        raise ValueError("need at least one strategy to compare with the benchmark")
    return bench, {str(k): -np.asarray(v, dtype=np.float64) for k, v in returns.items()}


def _as_return_result(res: MethodResult, note: str) -> MethodResult:
    """Re-express a loss-based result in returns: ``estimate`` = best mean excess return."""
    diag = dict(res.diagnostics)
    if "mean_loss_difference" in diag:
        diag["mean_excess_return"] = {k: -v for k, v in diag.pop("mean_loss_difference").items()}
    return dataclasses.replace(
        res,
        estimate=-res.estimate,
        rationale=f"{note} {res.rationale}",
        diagnostics=diag,
    )


def spa_vs_buy_and_hold(
    returns: Mapping[str, Any],
    benchmark: Any,
    rng: np.random.Generator,
    reps: int = 1000,
    block_len: int | None = None,
    alpha: float = 0.05,
) -> MethodResult:
    """Hansen's SPA: does ANY strategy beat the benchmark (for example buy-and-hold)?

    Runs :func:`~isbetter.methods.superior_predictive.spa_test` on losses defined as negative
    returns, so a strategy beats the benchmark when its mean return is higher. ``estimate`` is
    the best mean excess return over the benchmark; ``p_value`` is Hansen's consistent p-value,
    which accounts for having searched over all strategies in ``returns``.
    """
    bench, losses = _loss_inputs(returns, benchmark)
    res = spa_test(bench, losses, rng, reps=reps, block_len=block_len, alpha=alpha)
    return _as_return_result(
        res, "Strategies are scored by loss = -return, so lower loss means higher return."
    )


def stepm_vs_buy_and_hold(
    returns: Mapping[str, Any],
    benchmark: Any,
    rng: np.random.Generator,
    reps: int = 1000,
    block_len: int | None = None,
    size: float = 0.05,
) -> MethodResult:
    """Romano-Wolf StepM: which strategies beat the benchmark, controlling the family-wise error.

    Same loss convention as :func:`spa_vs_buy_and_hold`; ``diagnostics['superior_models']``
    lists the strategies that significantly out-earn the benchmark (often none).
    """
    bench, losses = _loss_inputs(returns, benchmark)
    res = stepm(bench, losses, rng, size=size, reps=reps, block_len=block_len)
    return _as_return_result(
        res, "Strategies are scored by loss = -return, so lower loss means higher return."
    )
