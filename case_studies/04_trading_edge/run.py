"""Case study 4: is the best moving-average rule an edge, or the luck of picking the best of 272?

    uv run python case_studies/04_trading_edge/run.py                    # synthetic stand-in
    uv run python case_studies/04_trading_edge/run.py --data prices.csv  # real daily ETF prices

Real ETF prices need a data download that was not reachable from the environment this was built
in, so by default two clearly labelled SYNTHETIC daily-price worlds are used: a ``luck`` world
whose returns are unpredictable (no moving-average rule has any true edge) and a ``trend`` world
whose expected return drifts persistently (trend-following has a true edge). Nothing here is a
statement about real markets. With ``--data`` (a CSV with columns ``date`` and ``close``) the same
analysis runs on one real price series.

Design. Every long/flat moving-average crossover (fast < slow, 272 variants) is evaluated by one
function decorated with ``TrialRegistry.track``, so every variant tried is logged without anyone
remembering to. The position held over day t + 1 is decided by prices up to day t. What is
tested is the *excess return over buy-and-hold*: a variant that is merely invested part of the
time in a rising market has no timing skill. The "best" variant is the one with the highest
Sharpe ratio, which is what an analyst would present.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from isbetter.adapters.strategy_returns import (
    TrialRegistry,
    prices_to_returns,
    resolve_deflation,
    returns_table,
)
from isbetter.core.config import load_config
from isbetter.core.plan import build_plan
from isbetter.core.runner import run
from isbetter.methods.finance import (
    annualize_sharpe,
    deflated_sharpe,
    min_track_record,
    probabilistic_sharpe,
    return_moments,
    sharpe_block_bootstrap,
    spa_vs_buy_and_hold,
    stepm_vs_buy_and_hold,
)
from isbetter.methods.pbo import pbo_cscv
from isbetter.report.markdown import render_markdown

HERE = Path(__file__).parent
PER_YEAR = 252
N_DAYS = 2520  # ten years of daily data
WARMUP = 250  # days before the first return that is scored (the longest slow average)
FAST = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 75, 90, 100]
SLOW = [20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 225, 250]
GRID = [(f, s) for s in SLOW for f in FAST if f < s]
WORLDS = ("luck", "trend")
N_REPEATS = 100  # simulated worlds per kind for the "how often is it fooled" table


# ------------------------------------------------------------------------------- data


def synthetic_returns(seed: int, kind: str, n: int = N_DAYS) -> np.ndarray:
    """SYNTHETIC daily returns: GARCH(1,1) with t(5) shocks, 1% daily volatility.

    ``luck``: no predictability, expected return 0. ``trend``: expected return follows a
    persistent AR(1) (autocorrelation 0.97, so a regime lasts about a month) with standard
    deviation 0.15% a day; being long while it is positive earns money.
    """
    rng = np.random.default_rng(seed)
    vol = 0.01
    z = rng.standard_t(5, n) * np.sqrt(3 / 5)
    omega = vol**2 * 0.05
    e = np.empty(n)
    s2, prev = vol**2, 0.0
    for t in range(n):
        s2 = omega + 0.10 * prev**2 + 0.85 * s2
        e[t] = prev = np.sqrt(s2) * z[t]
    if kind == "luck":
        return e
    if kind != "trend":
        raise ValueError("kind must be 'luck' or 'trend'")
    drift = np.zeros(n)
    eta = rng.normal(0.0, 0.0015 * np.sqrt(1 - 0.97**2), n)
    for t in range(1, n):
        drift[t] = 0.97 * drift[t - 1] + eta[t]
    return e + drift


def load_prices(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Daily closes from a CSV with ``date`` and ``close`` (or ``adj_close``) columns."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    col = "adj_close" if "adj_close" in rows[0] else "close"
    if "date" not in rows[0] or col not in rows[0]:
        raise ValueError(f"{path} needs 'date' and 'close' (or 'adj_close') columns")
    rows.sort(key=lambda r: r["date"])
    return np.array([float(r[col]) for r in rows]), [r["date"] for r in rows]


# ---------------------------------------------------------------------------- strategies


class MovingAverageGrid:
    """Long/flat crossover rules on one price series; scored from index ``WARMUP`` on."""

    def __init__(self, prices: np.ndarray, registry: TrialRegistry | None = None) -> None:
        self.returns = prices_to_returns(prices)  # returns[i] is the move from price i to i+1
        self.cum = np.concatenate([[0.0], np.cumsum(prices)])
        self.prices = prices
        self.buy_and_hold = self.returns[WARMUP - 1 :]
        # decorate here so that every call is logged: the evaluator cannot skip the registry
        self.excess_return = (
            self.compute if registry is None else registry.track(name="ma_crossover")(self.compute)
        )

    def _sma(self, window: int) -> np.ndarray:
        out = np.full(len(self.prices), np.nan)
        out[window - 1 :] = (self.cum[window:] - self.cum[:-window]) / window
        return out

    def compute(self, fast: int, slow: int) -> np.ndarray:
        """Long/flat rule return minus buy-and-hold, per day (position set at the prior close)."""
        signal = (self._sma(fast) > self._sma(slow)).astype(np.float64)
        pos = signal[WARMUP - 1 : len(self.returns)]  # position held over returns[WARMUP - 1:]
        return (pos - 1.0) * self.buy_and_hold


# --------------------------------------------------------------------------- analysis


@dataclass
class WorldAnalysis:
    label: str
    source: str
    n_days: int
    registry: TrialRegistry
    names: list[str]
    excess: np.ndarray  # (T, N) excess returns over buy-and-hold
    buy_and_hold: np.ndarray
    best: int
    sr: np.ndarray  # per-period Sharpe ratio of each variant's excess return


def evaluate(prices: np.ndarray, registry_path: Path, label: str, source: str) -> WorldAnalysis:
    registry_path.parent.mkdir(exist_ok=True)
    registry_path.unlink(missing_ok=True)  # one registry per run: stale trials would inflate N
    registry = TrialRegistry(registry_path)
    grid = MovingAverageGrid(prices, registry)
    cols = [grid.excess_return(fast=f, slow=s) for f, s in GRID]
    excess = np.column_stack(cols)
    sr = excess.mean(0) / excess.std(0, ddof=1)
    names = [f"ma_{f}_{s}" for f, s in GRID]
    return WorldAnalysis(
        label, source, len(prices), registry, names, excess, grid.buy_and_hold,
        int(sr.argmax()), sr,
    )  # fmt: skip


def naive_t(returns: np.ndarray) -> tuple[float, float]:
    """t-statistic and two-sided p-value of "mean return is zero", as if one rule had been tried."""
    t = float(returns.mean() / (returns.std(ddof=1) / np.sqrt(len(returns))))
    return t, float(2 * stats.t.sf(abs(t), len(returns) - 1))


def ann(sharpe: float) -> float:
    return annualize_sharpe(sharpe, PER_YEAR)


def fooled_rates(kind: str, seed: int, repeats: int) -> tuple[float, float, float, float]:
    """Over ``repeats`` fresh worlds: how often a naive t > 1.96 / DSR > 0.95 flags the best
    variant, and the mean PBO and mean best-variant t."""
    naive = dsr_pass = 0
    pbos, ts = [], []
    for i in range(repeats):
        prices = 100 * np.cumprod(1 + synthetic_returns(seed + 1_000 + i, kind))
        # no registry: n_trials and the Sharpe spread come straight from the grid
        grid = MovingAverageGrid(prices)
        x = np.column_stack([grid.excess_return(f, s) for f, s in GRID])
        sr = x.mean(0) / x.std(0, ddof=1)
        j = int(sr.argmax())
        t, _ = naive_t(x[:, j])
        res = deflated_sharpe(x[:, j], n_trials=len(GRID), sr_variance=float(sr.var(ddof=1)))
        naive += t > 1.96
        dsr_pass += res.diagnostics["dsr"] > 0.95
        pbos.append(pbo_cscv(x, 16).estimate)
        ts.append(t)
    return naive / repeats, dsr_pass / repeats, float(np.mean(pbos)), float(np.mean(ts))


def first_fooled_world(kind: str, seed: int, limit: int = 60) -> tuple[int, int]:
    """First seed >= ``seed`` whose best variant has a naive t > 1.96, and how many were scanned.

    This is a *conditional* illustration (the naive test is fooled) and is disclosed as such.
    """
    for i in range(limit):
        prices = 100 * np.cumprod(1 + synthetic_returns(seed + i, kind))
        grid = MovingAverageGrid(prices)
        x = np.column_stack([grid.excess_return(f, s) for f, s in GRID])
        sr = x.mean(0) / x.std(0, ddof=1)
        if naive_t(x[:, int(sr.argmax())])[0] > 1.96:
            return seed + i, i + 1
    raise RuntimeError(f"no world with a naive t > 1.96 within {limit} seeds")


def describe(w: WorldAnalysis, cfg_seed: int, base: Path) -> dict[str, Any]:
    """Every statistic the report needs for one world."""
    x_best = w.excess[:, w.best]
    T = len(x_best)
    t, p_naive = naive_t(x_best)
    inputs = resolve_deflation(w.registry)
    psr0 = probabilistic_sharpe(x_best)
    dsr = deflated_sharpe(x_best, n_trials=inputs.n_trials, sr_variance=inputs.sr_variance)
    sr_star = dsr.diagnostics["sr_star"]
    trl_zero = min_track_record(x_best, 0.0)
    trl_star = min_track_record(x_best, sr_star)
    rng = np.random.default_rng(cfg_seed)
    block = sharpe_block_bootstrap(x_best, rng, n_resamples=4999)
    iid = sharpe_block_bootstrap(x_best, rng, block_len=1, n_resamples=4999)
    # all variants against buy-and-hold (returns, not excess returns)
    gross = {n: w.excess[:, j] + w.buy_and_hold for j, n in enumerate(w.names)}
    spa = spa_vs_buy_and_hold(gross, w.buy_and_hold, rng, reps=500)
    sm = stepm_vs_buy_and_hold(gross, w.buy_and_hold, rng, reps=500)
    pbo = pbo_cscv(w.excess, 16, names=w.names)
    # the verdict an analyst gets from a standard test that ignores the search
    best_name = w.names[w.best]
    table = returns_table(
        {"best_variant": gross[best_name], "buy_and_hold": w.buy_and_hold},
        times=np.arange(T),
    )
    path = base / f"returns_{w.label}.csv"
    cols = table.columns
    with open(path, "w", newline="") as fh:
        cw = csv.writer(fh)
        names = ["unit_id", "pair_id", "variant", "metric", "value", "time"]
        cw.writerow(names)
        for i in range(len(table)):
            cw.writerow([cols[k][i] if k != "value" else f"{cols[k][i]:.10f}" for k in names])
    cfg = load_config(HERE / "config.yaml")
    plan = dataclasses.replace(build_plan(cfg), experiment=f"trading_edge_{w.label}")
    verdict = run(plan, table)
    return {
        "best_name": best_name, "T": T, "t": t, "p_naive": p_naive, "psr0": psr0, "dsr": dsr,
        "sr_star": sr_star, "trl_zero": trl_zero, "trl_star": trl_star, "block": block,
        "iid": iid, "spa": spa, "stepm": sm, "pbo": pbo, "inputs": inputs, "verdict": verdict,
        "sr_best": float(x_best.mean() / x_best.std(ddof=1)),
        "mean_best_ann": float(x_best.mean() * PER_YEAR),
        "bh_sharpe_ann": ann(return_moments(w.buy_and_hold).sharpe),
        "csv": path,
    }  # fmt: skip


# ----------------------------------------------------------------------------- report


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def years(n: float) -> str:
    return "never" if not np.isfinite(n) else f"{n / PER_YEAR:.1f} years"


def verdict_sentence(d: dict[str, Any]) -> str:
    dsr = d["dsr"].diagnostics["dsr"]
    if dsr >= 0.95:
        return (
            f"**Edge survives the search.** DSR = {dsr:.3f}: the best variant's excess Sharpe "
            f"ratio is above what the best of {d['inputs'].n_trials} unskilled variants would "
            "reach by luck, with 95% confidence."
        )
    return (
        f"**Cannot be told from luck.** DSR = {dsr:.3f} (< 0.95): after accounting for the "
        f"{d['inputs'].n_trials} variants tried, the best variant's Sharpe ratio is not "
        "distinguishable from the best of that many unskilled variants."
    )


def world_section(w: WorldAnalysis, d: dict[str, Any], heading: str, truth: str) -> str:
    n = d["inputs"].n_trials
    ci = {k: (d[k].ci_low, d[k].ci_high) for k in ("block", "iid")}
    assert all(v is not None for pair in ci.values() for v in pair)
    star = d["sr_star"]
    sup = d["stepm"].diagnostics["superior_models"]
    rows = [
        ("naive t-test", "mean excess return of the best rule = 0?",
         f"t = **{d['t']:.2f}**, p = {fmt_p(d['p_naive'])}"),
        ("PSR (no deflation)", "P(true Sharpe > 0), given skew and fat tails",
         f"{d['psr0'].diagnostics['psr']:.3f}"),
        ("**DSR**", f"P(true Sharpe > what the best of {n} unskilled variants reaches)",
         f"**{d['dsr'].diagnostics['dsr']:.3f}** (threshold SR* = {ann(star):.2f} a year, "
         f"observed {ann(d['sr_best']):.2f})"),
        ("MinTRL", "track record needed to clear 0 / to clear SR*",
         f"{years(d['trl_zero'].estimate)} / {years(d['trl_star'].estimate)} "
         f"(have {years(d['T'])})"),
        ("block-bootstrap CI", "95% interval of the annual excess Sharpe ratio",
         f"[{ann(ci['block'][0]):+.2f}, {ann(ci['block'][1]):+.2f}]"),
        ("iid-bootstrap CI", "same, wrongly assuming independent days",
         f"[{ann(ci['iid'][0]):+.2f}, {ann(ci['iid'][1]):+.2f}]"),
        ("SPA vs buy-and-hold", f"does *any* of the {n} rules beat buy-and-hold?",
         f"consistent p = {fmt_p(d['spa'].p_value)}"),
        ("StepM vs buy-and-hold", "rules that significantly beat it (family-wise 5%)",
         f"{len(sup)} of {n}"),
        ("**PBO** (CSCV, 16 blocks)", "how often the in-sample best is at or below the OOS median",
         f"**{d['pbo'].estimate:.2f}**"),
        ("standard DM test, ignoring the search", "verdict on `best_variant` vs buy-and-hold",
         f"`{d['verdict'].comparisons[0].verdict.label}`"),
    ]  # fmt: skip
    table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    return f"""### {heading}

*{w.source}.* Best variant by Sharpe ratio: **`{d["best_name"]}`**, scored on {d["T"]:,} days
({years(d["T"])}); its excess return over buy-and-hold averages {d["mean_best_ann"]:+.1%} a year.

| step | what it says | result |
|---|---|---|
{table}

{verdict_sentence(d)} The deflation is the difference between the PSR row and the DSR row.

{truth}
"""


TRUTH = {
    "luck": (
        "*Known only because the data are simulated:* no rule has any edge here, so the best "
        "variant's apparent out-performance is luck."
    ),
    "trend": (
        "*Known only because the data are simulated:* an edge does exist here, but "
        "{years} of data cannot confirm it once {n} variants have been tried: the price of "
        "deflating is power."
    ),
}


def truth_note(w: WorldAnalysis, d: dict[str, Any]) -> str:
    if w.label not in TRUTH:
        return ""
    note = TRUTH[w.label].format(years=years(d["T"]), n=d["inputs"].n_trials)
    if w.label == "trend" and d["dsr"].diagnostics["dsr"] >= 0.95:
        note = (
            "*Known only because the data are simulated:* an edge does exist here and DSR finds it."
        )
    return note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV of daily prices with columns date, close")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()

    cfg = load_config(HERE / "config.yaml")
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    worlds: list[tuple[WorldAnalysis, dict[str, Any], str]] = []
    scan_note = ""
    repeated = ""
    rates: dict[str, tuple[float, float, float]] = {}
    if args.data:
        prices, dates = load_prices(args.data)
        w = evaluate(prices, out / "trials_real.jsonl", "real", f"real prices from {args.data}")
        worlds.append((w, describe(w, cfg.seed, out), f"Real data ({dates[0]} to {dates[-1]})"))
        source = f"real prices from {args.data}"
    else:
        source = "SYNTHETIC daily prices -- NOT real market data"
        rows = []
        for kind, title in (
            ("luck", "World A: no edge exists (returns are unpredictable)"),
            ("trend", "World B: an edge exists (persistent trends)"),
        ):
            seed, scanned = first_fooled_world(kind, cfg.seed)
            prices = 100 * np.cumprod(1 + synthetic_returns(seed, kind))
            w = evaluate(prices, out / f"trials_{kind}.jsonl", kind, f"SYNTHETIC world '{kind}'")
            worlds.append((w, describe(w, cfg.seed, out), title))
            scan_note += (
                f"* {title.split(':')[0]}: seed {seed}, the first of {scanned} consecutive "
                "seeds (starting at the config seed) in which the best variant's naive t "
                "exceeds 1.96.\n"
            )
            naive_rate, dsr_rate, mean_pbo, mean_t = fooled_rates(kind, cfg.seed, args.repeats)
            rows.append(
                f"| {kind} | {mean_t:.2f} | {naive_rate:.0%} | {dsr_rate:.0%} | {mean_pbo:.2f} |"
            )
            rates[kind] = (naive_rate, dsr_rate, mean_pbo)
        repeated = "\n".join(rows)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    sections = "\n".join(world_section(w, d, title, truth_note(w, d)) for w, d, title in worlds)
    luck_d = worlds[0][1]
    verdict_md = render_markdown(luck_d["verdict"])
    n_trials = luck_d["inputs"].n_trials
    data_note = (
        "Real data were used."
        if args.data
        else "Real ETF prices could not be downloaded here; rerun with `--data prices.csv` "
        "for a real analysis. Nothing below is a claim about real markets."
    )
    table_md = (
        f"""
Unconditional behaviour over {args.repeats} fresh simulated worlds of each kind (the best of the
same {n_trials} variants in each):

| world | mean naive t of the best variant | naive t > 1.96 | DSR > 0.95 | mean PBO |
|---|---:|---:|---:|---:|
{repeated}

In the *luck* world every flagged result is a false positive: the naive test raises one in
{rates["luck"][0]:.0%} of worlds (a single pre-registered test would, by design, raise 5%), DSR
{rates["luck"][1]:.0%}. In the *trend* world the edge is real and DSR finds it less often than
the naive test does ({rates["trend"][1]:.0%} against {rates["trend"][0]:.0%}): the price of not
being fooled is power. PBO is {rates["luck"][2]:.2f} in the luck world (tuning the parameters
does not help, as it should be) and {rates["trend"][2]:.2f} in the trend world, only modestly
lower although the edge is real: neighbouring moving averages are near-duplicates, so *which* one
ranks first in-sample is largely chance even when they share a real edge. PBO answers "does
picking the best parameters help?", not "is there an edge?".
"""
        if repeated
        else ""
    )
    md = f"""# Case study 4: trading edge or luck? (moving-average crossovers)

> **Data:** {source}.
> {data_note}

## 1. Question

Search {n_trials} long/flat moving-average rules, keep the best one: is its out-performance over
buy-and-hold an *edge*, or what you would expect from the best of {n_trials} rules that have no
skill at all? A difference smaller than **{cfg.decision.min_effect:g} a day** (about
{cfg.decision.min_effect * PER_YEAR:.0%} a year) is treated as practically irrelevant.

## 2. Data and dependence structure

Daily returns ({N_DAYS:,} days, the first {WARMUP} used to warm up the averages). Daily returns
are serially dependent in their *size* (volatility clusters) and, for some strategies, in their
direction (a trend rule holds positions for weeks, so its daily returns are autocorrelated), hence
`dependence = timeseries`. The {n_trials} rules are evaluated on the same days, so they are
strongly correlated with each other, and every evaluation is appended to a `TrialRegistry`
(`out/trials_*.jsonl`), the source of the trial count and the Sharpe spread used by the DSR.

Every statistic below is about the **excess return over buy-and-hold** (rule return minus
buy-and-hold return, per day).

## 3. Naive analysis, and why it misleads

Pick the rule with the highest Sharpe ratio, and t-test whether its mean excess return is zero.
That test is valid for *one* rule chosen in advance. The best of {n_trials} is a maximum, so its
t-statistic is inflated even when no rule has any edge. Disclosure: the worlds shown below are
conditioned on the naive test being fooled, so the story is visible; how often that happens
without conditioning is in section 5.

{scan_note}
## 4. Correct analysis

{sections}
The standard verdict for world A, as `isbetter run` produces it for a single pre-chosen
comparison (it knows nothing about the search):

{verdict_md.split("## Results", 1)[1] if "## Results" in verdict_md else verdict_md}

## 5. How often is each tool fooled?
{table_md}
## 6. What would change the conclusion

* **Real data.** Equity indices trend only weakly and have positive drift; run with `--data`.
* **Transaction costs and slippage** are not modelled; they only make every rule worse.
* **Number of trials.** The DSR assumes independent trials; {n_trials} neighbouring averages are
  far fewer independent bets, but their Sharpe spread is correspondingly small. The registry
  count and spread are used as logged. Trials run outside the registry would raise the bar:
  pass an explicit `n_trials`.
* **A longer track record**: MinTRL above says how many years the best variant would need.
* Peak memory of this run: {peak_gb:.2f} GB.
"""
    (HERE / "REPORT.md").write_text(md)
    print(f"report written to {HERE / 'REPORT.md'}  (peak RSS {peak_gb:.2f} GB)")


if __name__ == "__main__":
    main()
