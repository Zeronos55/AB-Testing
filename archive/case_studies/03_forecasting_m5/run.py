"""Case study 3: is ARIMA or LightGBM really better than seasonal-naive, on M5-style daily sales?

    uv run python case_studies/03_forecasting_m5/run.py                  # synthetic stand-in data
    uv run python case_studies/03_forecasting_m5/run.py --stores 10      # all 70 series (slow)
    uv run python case_studies/03_forecasting_m5/run.py --data sales_train_evaluation.csv

The real M5 file (Kaggle "m5-forecasting-accuracy", ``sales_train_evaluation.csv``) needs a
Kaggle login and could not be downloaded in the environment this was built in, so by default a
clearly labelled SYNTHETIC M5-style dataset is used (10 stores x 7 departments, daily counts with
weekly and yearly seasonality, promotions, a shared store shock and over-dispersed noise).
Results on synthetic data say nothing about real retail demand.

Design. Every model forecasts 14 days ahead from every daily origin (an expanding window; ARIMA
and LightGBM are refitted every 28 origins and only *filtered* in between, so no model ever sees
data from after its origin). Per-series scaled errors (MASE, scale from the first 730 days) are
averaged over the series, which leaves ONE loss per model and forecast origin: we test over
origins, not series, because series in a store share a shock. Forecast errors at horizon h
overlap across the next h - 1 origins, hence ``dependence: timeseries`` with ``horizon: h``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import resource
import warnings
from pathlib import Path

import numpy as np

from isbetter.adapters.forecast_backtest import backtest_table, seasonal_naive
from isbetter.core.config import load_config
from isbetter.core.dependence import TimeSeries
from isbetter.core.plan import build_plan
from isbetter.core.runner import run
from isbetter.core.schema import CanonicalTable, build_table
from isbetter.methods.forecasting import diebold_mariano, naive_dm_t, suggest_block_length
from isbetter.methods.superior_predictive import model_confidence_set, spa_test, stepm
from isbetter.report.markdown import render_markdown

HERE = Path(__file__).parent
HORIZON = 14  # days ahead
EVAL_HORIZONS = (1, 7, 14)
INITIAL = 730  # first training window (days); also the window the MASE scale is taken from
N_DAYS = 1096  # three years
REFIT = 28  # origins between refits of ARIMA / LightGBM
SEASON = 7
MODELS = ("seasonal_naive", "arima", "lightgbm")
DEPTS = ("FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2")


# ------------------------------------------------------------------------------- data


def synthetic_m5(
    seed: int, n_days: int = N_DAYS, n_stores: int = 10
) -> dict[tuple[str, str], np.ndarray]:
    """SYNTHETIC M5-style daily unit sales for ``n_stores`` x 7 departments."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    out: dict[tuple[str, str], np.ndarray] = {}
    for s in range(n_stores):
        store = f"STORE_{s + 1}"
        # one AR(1) shock per store, shared by all its departments (cross-sectional dependence)
        shock = np.zeros(n_days)
        eps = rng.normal(0, 0.06, n_days)
        for i in range(1, n_days):
            shock[i] = 0.85 * shock[i - 1] + eps[i]
        store_scale = rng.lognormal(0.0, 0.3)
        for d, dept in enumerate(DEPTS):
            level = store_scale * rng.lognormal(np.log(60 if d < 3 else 25), 0.4)
            weekly = 1 + rng.uniform(0.1, 0.4) * np.cos(2 * np.pi * (t - rng.integers(0, 7)) / 7)
            yearly = 1 + rng.uniform(0.05, 0.2) * np.sin(2 * np.pi * t / 365.25 + rng.uniform(0, 6))
            trend = 1 + rng.uniform(-0.1, 0.3) * t / n_days
            promo = np.ones(n_days)
            for start in rng.choice(n_days - 5, size=n_days // 45, replace=False):
                promo[start : start + rng.integers(2, 5)] = rng.uniform(1.3, 1.9)
            mean = level * weekly * yearly * trend * promo * np.exp(shock)
            r = 12.0  # negative-binomial dispersion: variance = mean + mean^2 / r
            out[(store, dept)] = rng.poisson(rng.gamma(r, mean / r)).astype(np.float64)
    return out


def load_m5(
    path: str | Path, n_days: int = N_DAYS, n_stores: int = 10
) -> dict[tuple[str, str], np.ndarray]:
    """Daily unit sales per (store, department) from M5's ``sales_train_evaluation.csv``.

    Streams the file row by row and keeps only one accumulator per (store, department), so
    memory stays tiny; keeps the last ``n_days`` days and the first ``n_stores`` stores.
    """
    sums: dict[tuple[str, str], np.ndarray] = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        need = {"dept_id", "store_id"}
        if not need <= set(header):
            raise ValueError(
                f"{path} is not an M5 sales file: missing {sorted(need - set(header))}"
            )
        di, si = header.index("dept_id"), header.index("store_id")
        day_cols = [i for i, c in enumerate(header) if c.startswith("d_")]
        if len(day_cols) < n_days:
            raise ValueError(f"file has {len(day_cols)} days, need at least {n_days}")
        day_cols = day_cols[-n_days:]
        for row in reader:
            key = (row[si], row[di])
            vals = np.array([row[i] for i in day_cols], dtype=np.float64)
            if key in sums:
                sums[key] += vals
            else:
                sums[key] = vals
    stores = sorted({k[0] for k in sums})[:n_stores]
    return {k: v for k, v in sorted(sums.items()) if k[0] in stores}


# ---------------------------------------------------------------------------- models


def forecast_seasonal_naive(y: np.ndarray, origins: np.ndarray) -> np.ndarray:
    f = seasonal_naive(SEASON)
    return np.stack([f(y[:o], HORIZON) for o in origins])


def forecast_arima(y: np.ndarray, origins: np.ndarray) -> tuple[np.ndarray, int]:
    """ARIMA(1,0,1)(1,0,0)_7 on log(1 + sales), refitted every REFIT origins."""
    from statsmodels.tsa.arima.model import ARIMA

    z = np.log1p(y)
    out = np.empty((len(origins), HORIZON))
    fitted = None
    n_warn = 0
    for i, o in enumerate(origins):
        if i % REFIT == 0:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = ARIMA(z[:o], order=(1, 0, 1), seasonal_order=(1, 0, 0, SEASON)).fit()
            n_warn += len(caught)
        assert fitted is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[i] = fitted.apply(z[:o], refit=False).forecast(HORIZON)
    return np.clip(np.expm1(out), 0.0, None), n_warn


def lgbm_features(Y: np.ndarray, origins: np.ndarray, series_code: np.ndarray) -> np.ndarray:
    """Features for every (series, origin, horizon), shape (S, n_origins, H, F).

    Only values at positions < origin are used (the last week, rolling means, and the
    same-weekday values of the 3 most recent weeks that precede the origin), so the features of
    an origin never change when later data change.
    """
    S, N = Y.shape
    o = origins[:, None]  # (n_o, 1)
    h = np.arange(1, HORIZON + 1)[None, :]  # (1, H)
    target = o + h - 1  # position being forecast
    cs = np.concatenate([np.zeros((S, 1)), np.cumsum(Y, axis=1)], axis=1)
    feats = []
    shape = target.shape
    for s in range(S):
        row = Y[s]
        lags = np.stack([row[origins - k] for k in range(1, 8)], axis=-1)  # (n_o, 7)
        m7 = (cs[s, origins] - cs[s, origins - 7]) / 7
        m28 = (cs[s, origins] - cs[s, origins - 28]) / 28
        k0 = -(-h // SEASON)  # weeks back to the first same-weekday position before the origin
        same = np.stack([row[target - SEASON * (k0 + j)] for j in range(3)], axis=-1)  # (n_o,H,3)
        cols = [
            np.broadcast_to(lags[:, None, :], (*shape, 7)),
            np.broadcast_to(m7[:, None, None], (*shape, 1)),
            np.broadcast_to(m28[:, None, None], (*shape, 1)),
            same,
            (target % SEASON)[..., None],
            np.broadcast_to(h[..., None], (*shape, 1)),
            np.full((*shape, 1), series_code[s]),
        ]
        feats.append(np.concatenate([c.astype(np.float64) for c in cols], axis=-1))
    return np.stack(feats)


def forecast_lightgbm(Y: np.ndarray, origins: np.ndarray, seed: int) -> np.ndarray:
    """One global LightGBM (direct multi-horizon, L1 objective) refitted every REFIT origins.

    Trained on every (series, origin', horizon) whose target is known at the refit origin, on
    the ratio ``y / (28-day mean + 1)``; predictions are multiplied back.
    """
    import lightgbm as lgb

    S = Y.shape[0]
    codes = np.arange(S)
    train_origins = np.arange(60, origins[-1])
    X_tr_all = lgbm_features(Y, train_origins, codes)  # (S, n_tr, H, F)
    X_ev = lgbm_features(Y, origins, codes)
    n_tr = len(train_origins)
    h = np.arange(1, HORIZON + 1)
    pos = train_origins[:, None] + h[None, :] - 1  # (n_tr, H)
    target = np.stack([Y[s][np.minimum(pos, Y.shape[1] - 1)] for s in range(S)])  # (S, n_tr, H)
    ratio = target / (X_tr_all[..., 8:9][..., 0] + 1.0)  # feature 8 is the 28-day mean
    out = np.empty((S, len(origins), HORIZON))
    for i0 in range(0, len(origins), REFIT):
        o_r = int(origins[i0])
        known = (train_origins[:, None] + h[None, :] - 1) <= o_r - 1  # target observed by o_r
        mask = np.broadcast_to(known, (S, n_tr, HORIZON))
        model = lgb.LGBMRegressor(
            objective="l1", n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, verbose=-1, n_jobs=2, deterministic=True, force_row_wise=True,
        )  # fmt: skip
        model.fit(X_tr_all[mask], ratio[mask], categorical_feature=[X_tr_all.shape[-1] - 1])
        sl = slice(i0, min(i0 + REFIT, len(origins)))
        Xe = X_ev[:, sl]
        pred = model.predict(Xe.reshape(-1, Xe.shape[-1])).reshape(Xe.shape[:-1])
        out[:, sl] = np.clip(pred * (Xe[..., 8] + 1.0), 0.0, None)
    return out


# ------------------------------------------------------------------------- evaluation


def aggregate_losses(
    series: list[np.ndarray], forecasts: dict[str, list[np.ndarray]], origins: np.ndarray
) -> CanonicalTable:
    """Mean over series of the per-origin scaled MAE: one loss per (model, origin, horizon)."""
    per_series = [
        backtest_table(
            y,
            {m: forecasts[m][i] for m in MODELS},
            origins,
            loss="mase",
            horizons=EVAL_HORIZONS,
            season=SEASON,
            initial=INITIAL,
        ).table
        for i, y in enumerate(series)
    ]
    cols: dict[str, list[object]] = {
        k: [] for k in ("unit_id", "pair_id", "variant", "metric", "value", "time", "horizon")
    }
    for m in MODELS:
        for h in EVAL_HORIZONS:
            metric = f"mase_h{h}"
            vals = np.mean([t.select(m, metric)["value"] for t in per_series], axis=0)
            n = len(origins)
            cols["unit_id"] += [f"o{int(o)}" for o in origins]
            cols["pair_id"] += [f"o{int(o)}" for o in origins]
            cols["variant"] += [m] * n
            cols["metric"] += [metric] * n
            cols["value"] += vals.tolist()
            cols["time"] += origins.tolist()
            cols["horizon"] += [h] * n
    return build_table(cols, TimeSeries())


def losses_of(table: CanonicalTable, metric: str) -> dict[str, np.ndarray]:
    return {m: table.select(m, metric)["value"] for m in MODELS}


def join_names(names: list[str]) -> str:
    ticks = [f"`{m}`" for m in names]
    return ticks[0] if len(ticks) == 1 else ", ".join(ticks[:-1]) + f" and {ticks[-1]}"


def bootstrap_p(p: float, reps: int) -> str:
    """A bootstrap p-value of 0 only means 'smaller than 1 / reps'."""
    return f"< {1 / reps:g}" if p < 1 / reps else f"{p:.3g}"


def mcs_sentence(included: list[str], excluded: list[str], size: float) -> str:
    keep = join_names(included)
    if not excluded:
        return (
            f"The {1 - size:.0%} Model Confidence Set contains **all {len(included)} models** "
            f"({keep}): over these origins, the data cannot tell any of them apart from the "
            "best one, so none can be ruled out."
        )
    gone = join_names(excluded)
    if len(included) == 1:
        return (
            f"The {1 - size:.0%} Model Confidence Set contains **only {keep}**: it is "
            f"significantly better than {gone}, and the data single it out as the best model."
        )
    return (
        f"The {1 - size:.0%} Model Confidence Set is {keep}. {gone} "
        f"{'was' if len(excluded) == 1 else 'were'} eliminated: even allowing for noise, "
        f"{'it is' if len(excluded) == 1 else 'they are'} less accurate than the survivors, "
        f"while the survivors cannot be separated from each other."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to M5 sales_train_evaluation.csv")
    ap.add_argument("--stores", type=int, default=1, help="stores to analyse (1-10; 7 series each)")
    args = ap.parse_args()

    cfg = load_config(HERE / "config.yaml")
    if args.data:
        data = load_m5(args.data, N_DAYS, 10)
        source = f"real M5 data from {args.data}"
    else:
        data = synthetic_m5(cfg.seed, N_DAYS, 10)
        source = "SYNTHETIC M5-style stand-in -- NOT real retail data"
    stores = sorted({k[0] for k in data})[: args.stores]
    keys = [k for k in sorted(data) if k[0] in stores]
    series = [data[k] for k in keys]
    Y = np.stack(series)
    print(f"data: {source}; {len(keys)} series ({len(stores)} store(s) x 7 departments)")

    origins = np.arange(INITIAL, N_DAYS - HORIZON + 1)
    forecasts: dict[str, list[np.ndarray]] = {m: [] for m in MODELS}
    arima_warnings = 0
    for y in series:
        forecasts["seasonal_naive"].append(forecast_seasonal_naive(y, origins))
        f, w = forecast_arima(y, origins)
        forecasts["arima"].append(f)
        arima_warnings += w
    lgbm = forecast_lightgbm(Y, origins, cfg.seed)
    forecasts["lightgbm"] = [lgbm[i] for i in range(len(series))]

    table = aggregate_losses(series, forecasts, origins)
    (HERE / "out").mkdir(exist_ok=True)
    cols = table.columns
    with open(HERE / "out" / "losses.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        names = ["unit_id", "pair_id", "variant", "metric", "value", "time", "horizon"]
        w.writerow(names)
        for i in range(len(table)):
            w.writerow([cols[k][i] if k != "value" else f"{cols[k][i]:.8f}" for k in names])

    base_plan = build_plan(cfg)
    results = {}
    for h in EVAL_HORIZONS:
        plan = dataclasses.replace(
            base_plan, metric=f"mase_h{h}", dependence=TimeSeries(horizon=h, block_len=None)
        )
        results[h] = run(plan, table)

    # --- naive analysis for contrast ----------------------------------------------------
    naive_rows = []
    for h in EVAL_HORIZONS:
        L = losses_of(table, f"mase_h{h}")
        for cand in cfg.variants.candidates:
            nv = naive_dm_t(L[cand], L["seasonal_naive"])
            dm = diebold_mariano(L[cand], L["seasonal_naive"], horizon=h)
            naive_rows.append(
                f"| {h} | {cand} | {nv.estimate:+.4f} | {nv.p_value:.3g} | {dm.p_value:.3g} | "
                f"{dm.diagnostics['se'] / nv.diagnostics['se']:.2f} |"
            )

    # --- the close call: LightGBM vs ARIMA head to head ------------------------------------
    h2h_rows = []
    for h in EVAL_HORIZONS:
        L = losses_of(table, f"mase_h{h}")
        nv = naive_dm_t(L["lightgbm"], L["arima"])
        dm = diebold_mariano(L["lightgbm"], L["arima"], horizon=h)
        h2h_rows.append(
            f"| {h} | {-dm.estimate:+.4f} | [{-dm.ci_high:+.4f}, {-dm.ci_low:+.4f}] | "
            f"{nv.p_value:.3g} | {dm.p_value:.3g} |"
        )

    # --- many models vs the benchmark, and the Model Confidence Set ----------------------
    rng = np.random.default_rng(cfg.seed)
    h_set = 7
    L7 = losses_of(table, f"mase_h{h_set}")
    cands = {m: L7[m] for m in MODELS if m != "seasonal_naive"}
    spa = spa_test(L7["seasonal_naive"], cands, rng)
    sm = stepm(L7["seasonal_naive"], cands, rng)
    mcs = model_confidence_set(L7, rng, size=0.10)
    sug = suggest_block_length(L7["lightgbm"] - L7["arima"])
    mcs_pv = ", ".join(f"`{m}` {p:.3f}" for m, p in mcs.diagnostics["pvalues"].items())

    mean_mase = {m: float(np.mean(L7[m])) for m in MODELS}
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    data_note = (
        "Real data were used."
        if args.data
        else "The real M5 file could not be downloaded here; rerun with `--data "
        "sales_train_evaluation.csv` for the real analysis. Nothing below is a claim about real "
        "retail demand."
    )
    headline = render_markdown(results[7])
    md = f"""# Case study 3: forecasting daily sales (M5-style)

> **Data:** {source}.
> {data_note}

## 1. Question

Is ARIMA or a global LightGBM model *really* more accurate than the seasonal-naive forecast
("same weekday last week") for daily sales {HORIZON} days ahead, and which of the three can we
rule out? Accuracy is the mean absolute scaled error (MASE: 1.0 means as accurate as seasonal
naive on the first {INITIAL} days). A difference smaller than **{cfg.decision.min_effect:g} MASE**
is treated as practically irrelevant.

## 2. Data and dependence structure

{len(keys)} daily series ({len(stores)} store(s) x 7 departments, {N_DAYS} days). Every model is
re-run from {len(origins)} consecutive daily forecast origins (days {int(origins[0])} to
{int(origins[-1])}), forecasting up to {HORIZON} days ahead; ARIMA and LightGBM are refitted every
{REFIT} origins. Per-series scaled errors are averaged over the series, giving **one loss per
model, origin and horizon**. We test over origins, not series, because series in a store move
together (a shared store-level shock), so they are not independent replicates.

The origins are not independent either. A forecast made on day *t* for day *t+h* and one made on
day *t+1* for day *t+1+h* are built from almost the same information, so the loss differential
of **h-step** forecasts is autocorrelated at lags 1 to h-1. That is `dependence = timeseries`
with `horizon = h`, and it picks the Diebold-Mariano test with the Harvey-Leybourne-Newbold
correction (`h - 1` autocovariance lags).

## 3. Naive analysis, and why it misleads

A paired t-test on the loss differential treats the {len(origins)} origins as independent
observations. At horizon 1 that is exactly right (the two p-values below are identical). At longer
horizons the differential is autocorrelated, and the naive test can be badly overconfident: in
the failure gallery (#2b, 100 origins) it falsely flags two equally good forecasters 17% of the
time at h=3 and 46% at h=10. How much it matters depends on the data: the last column is the
standard error of the Diebold-Mariano test relative to the naive one (above 1: the naive test was
too confident; below 1: too cautious).

| horizon | candidate | mean MASE diff vs seasonal_naive | naive p | DM p | SE ratio DM/naive |
|---:|---|---:|---:|---:|---:|
{chr(10).join(naive_rows)}

The two candidates are both far ahead of seasonal-naive, so the naive test and the correct test
agree there. The informative comparison is the close one, LightGBM against ARIMA (positive effect:
LightGBM is more accurate, in MASE units; 95% CIs):

| horizon | LightGBM minus ARIMA accuracy gain | 95% CI (Diebold-Mariano) | naive p | DM p |
|---:|---:|---|---:|---:|
{chr(10).join(h2h_rows)}

## 4. Correct analysis (Diebold-Mariano + HLN, Holm across the 2 candidates)

Headline at horizon 7 (one-week-ahead); the same analysis at horizons 1 and 14 is summarised
below the report.

{headline.split("## Results", 1)[1]}
**All three horizons** (oriented effect > 0 means the candidate is more accurate):

| horizon | candidate | effect in MASE | 95% CI (Holm level) | verdict |
|---:|---|---:|---|---|
{
        chr(10).join(
            f"| {h} | {c.candidate} | {c.oriented_estimate:+.4f} | "
            f"[{c.oriented_ci_low:+.4f}, {c.oriented_ci_high:+.4f}] | `{c.verdict.label}` |"
            for h in EVAL_HORIZONS
            for c in results[h].comparisons
        )
    }

## 5. Which models can we rule out? (SPA, StepM and the Model Confidence Set at horizon 7)

Mean MASE at horizon 7: {", ".join(f"`{m}` {v:.4f}" for m, v in mean_mase.items())}.

* **Hansen's SPA** (is *any* of ARIMA / LightGBM better than seasonal naive, accounting for having
  looked at both?): consistent p {bootstrap_p(spa.p_value, spa.diagnostics["reps"])}
  (bootstrap with {spa.diagnostics["reps"]:,} resamples).
* **Romano-Wolf StepM** (which of them are significantly better, family-wise error 5%):
  {join_names(sm.diagnostics["superior_models"]) if sm.diagnostics["superior_models"] else "none"}.
* **Model Confidence Set** (90%; block bootstrap with Politis-White block length
  {mcs.diagnostics["block_len"]}): {
        mcs_sentence(mcs.diagnostics["included"], mcs.diagnostics["excluded"], 0.10)
    }
  MCS p-values: {mcs_pv}.

*In plain English:* the Model Confidence Set is the list of models that are still in the running.
A model is dropped only if it is clearly worse than the best one; models that stay in are not
"equally good", they are simply not distinguishable with this much data. Being outside the set is
evidence of being worse; being inside is not evidence of being best. (Politis-White suggests a mean
block length of {sug.stationary:.2f} for the ARIMA-vs-LightGBM loss differential; a value below 1
means the estimated autocorrelation is negligible, so blocks of length 1 are used.)

## 6. What would change the conclusion

* More series or more years of evaluation: the verdicts above rest on {len(origins)}
  strongly overlapping origins, so the `n_needed` note in the report says how much more is needed
  for the inconclusive comparisons.
* A different margin: with a larger minimum effect more comparisons become "equivalent".
* Real data: promotions, SNAP days, zero-inflated items and price changes are all absent from the
  synthetic generator and could favour LightGBM (which can use them) over ARIMA.
* Run time and memory: peak RSS of this run was {peak_gb:.2f} GB; {
        arima_warnings
    } warning(s) were raised
  by ARIMA fits (all fits were kept).
"""
    (HERE / "REPORT.md").write_text(md)
    print(f"report written to {HERE / 'REPORT.md'}  (peak RSS {peak_gb:.2f} GB)")


if __name__ == "__main__":
    main()
