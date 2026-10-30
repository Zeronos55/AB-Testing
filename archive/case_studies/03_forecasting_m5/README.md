# Case study 3: forecasting daily sales (M5-style)

**Question:** is ARIMA or a global LightGBM model really more accurate than the seasonal-naive
forecast ("same weekday last week") for daily sales up to 14 days ahead, and which of the three
models can the data rule out?

**Run it**

```bash
uv run python case_studies/03_forecasting_m5/run.py                  # synthetic stand-in, 1 store
uv run python case_studies/03_forecasting_m5/run.py --stores 10      # all 70 series (roughly 10x slower; not run here)
uv run python case_studies/03_forecasting_m5/run.py --data sales_train_evaluation.csv  # real M5
uv run isbetter run case_studies/03_forecasting_m5/config.yaml       # the h = 7 comparison alone
```

Output: [`REPORT.md`](REPORT.md) (full write-up) and `out/losses.csv` (per-origin losses, which
the `isbetter run` command above analyses). The default run takes under 2 minutes and peaks at
about 0.3 GB of memory (limit: 4 GB).

**Data caveat.** The real M5 file (Kaggle login required) was not reachable from the build
environment, so the committed report uses a **synthetic** M5-style dataset (10 stores x 7
departments, daily counts with weekly and yearly seasonality, promotions and a shared store
shock). It demonstrates the method, not real retail forecasting performance. The loader for the
real file (`--data`) sums item rows to store x department series; it is tested on a small file
in the documented M5 layout, not on the real file.

**How the data are handled.** All three models forecast 14 days ahead from every daily origin
(expanding window; ARIMA and LightGBM are refitted every 28 origins). Per-series MASE is averaged
over series, leaving one loss per model, origin and horizon. Errors of h-step forecasts overlap
across consecutive origins, so the analysis declares `dependence: timeseries` with `horizon: h`
and uses Diebold-Mariano with the Harvey-Leybourne-Newbold correction, a stationary block
bootstrap as a cross-check, and SPA / StepM / the Model Confidence Set for the many-models
question.

**Headline on the synthetic data.** Both ARIMA and LightGBM beat seasonal-naive by about 0.22
to 0.30 MASE at every horizon (far beyond the 0.02 practical margin), and the Model Confidence
Set at horizon 7 keeps only LightGBM. The naive and the correct test agree on those big gaps;
they disagree on the close one, LightGBM vs ARIMA at horizon 14, where the naive p-value
(9e-20) overstates the evidence by seven orders of magnitude compared with Diebold-Mariano
(1.5e-12).
