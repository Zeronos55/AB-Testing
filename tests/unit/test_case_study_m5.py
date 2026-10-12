"""Helpers of case_studies/03_forecasting_m5/run.py: data loaders, features and leak-freeness."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

RUN = Path(__file__).resolve().parents[2] / "case_studies" / "03_forecasting_m5" / "run.py"
spec = importlib.util.spec_from_file_location("m5_case_study", RUN)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)


def test_synthetic_data_shape_and_reproducibility():
    a = m5.synthetic_m5(1, n_days=120, n_stores=2)
    b = m5.synthetic_m5(1, n_days=120, n_stores=2)
    assert len(a) == 14 and {k[1] for k in a} == set(m5.DEPTS)
    for k, v in a.items():
        assert v.shape == (120,) and (v >= 0).all() and np.array_equal(v, b[k])
    assert not np.array_equal(
        a[("STORE_1", "FOODS_1")], m5.synthetic_m5(2, 120, 2)[("STORE_1", "FOODS_1")]
    )


def write_m5(path, n_days=10, rows=None):
    header = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    header += [f"d_{i}" for i in range(1, n_days + 1)]
    rows = rows or [
        ("i1", "FOODS_1", "FOODS", "CA_1", "CA", 1),
        ("i2", "FOODS_1", "FOODS", "CA_1", "CA", 2),
        ("i3", "HOBBIES_1", "HOBBIES", "CA_1", "CA", 5),
        ("i4", "FOODS_1", "FOODS", "TX_1", "TX", 7),
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for item, dept, cat, store, state, k in rows:
            w.writerow(
                [
                    f"{item}_{store}",
                    item,
                    dept,
                    cat,
                    store,
                    state,
                    *[k * d for d in range(1, n_days + 1)],
                ]
            )


def test_load_m5_sums_items_per_store_and_department(tmp_path):
    p = tmp_path / "sales.csv"
    write_m5(p)
    got = m5.load_m5(p, n_days=4, n_stores=10)
    assert set(got) == {("CA_1", "FOODS_1"), ("CA_1", "HOBBIES_1"), ("TX_1", "FOODS_1")}
    # last 4 of 10 days: d_7..d_10; items i1 (k=1) and i2 (k=2) of CA_1 / FOODS_1 are summed
    assert got[("CA_1", "FOODS_1")].tolist() == [3 * d for d in (7, 8, 9, 10)]
    assert got[("TX_1", "FOODS_1")].tolist() == [7 * d for d in (7, 8, 9, 10)]
    assert set(k[0] for k in m5.load_m5(p, n_days=4, n_stores=1)) == {"CA_1"}


def test_load_m5_validation(tmp_path):
    p = tmp_path / "sales.csv"
    write_m5(p, n_days=3)
    with pytest.raises(ValueError, match="at least 5"):
        m5.load_m5(p, n_days=5)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not an M5 sales file"):
        m5.load_m5(bad, n_days=1)


def small_panel(seed=0, S=3, N=260):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    mean = 40 * (1 + 0.3 * np.cos(2 * np.pi * t / 7))[None, :] * rng.uniform(0.5, 2, (S, 1))
    return rng.poisson(mean).astype(np.float64)


def test_lgbm_features_use_only_the_past():
    Y = small_panel()
    codes = np.arange(Y.shape[0])
    o = 150
    Y2 = Y.copy()
    Y2[:, o:] = np.random.default_rng(1).uniform(0, 500, Y2[:, o:].shape)  # change the future
    a = m5.lgbm_features(Y, np.array([o]), codes)
    b = m5.lgbm_features(Y2, np.array([o]), codes)
    assert a.shape == (3, 1, m5.HORIZON, 15)
    np.testing.assert_array_equal(a, b)
    # sanity on content: feature 0 is the last observed value, feature 13 the horizon
    np.testing.assert_array_equal(a[1, 0, :, 0], np.full(m5.HORIZON, Y[1, o - 1]))
    assert a[0, 0, :, 13].tolist() == list(range(1, m5.HORIZON + 1))


def test_lightgbm_forecasts_have_no_lookahead():
    Y = small_panel(1)
    origins = np.arange(180, 247)
    Y2 = Y.copy()
    Y2[:, 200:] = np.random.default_rng(2).uniform(0, 500, Y2[:, 200:].shape)
    a = m5.forecast_lightgbm(Y, origins, seed=3)
    b = m5.forecast_lightgbm(Y2, origins, seed=3)
    early = origins < 200  # these origins only ever see positions < 200
    np.testing.assert_allclose(a[:, early], b[:, early], atol=1e-9)
    assert not np.allclose(a[:, ~early], b[:, ~early])  # and the changed data do matter later
    assert a.shape == (3, len(origins), m5.HORIZON) and (a >= 0).all()


def test_arima_forecasts_have_no_lookahead():
    y = small_panel(2, S=1)[0]
    origins = np.arange(180, 247)
    y2 = y.copy()
    y2[200:] = np.random.default_rng(3).uniform(0, 500, len(y2[200:]))
    a, _ = m5.forecast_arima(y, origins)
    b, _ = m5.forecast_arima(y2, origins)
    early = origins < 200
    np.testing.assert_allclose(a[early], b[early], atol=1e-9)
    assert a.shape == (len(origins), m5.HORIZON) and (a >= 0).all()


def test_seasonal_naive_matrix_is_last_weeks_values():
    y = small_panel(3, S=1)[0]
    f = m5.forecast_seasonal_naive(y, np.array([100, 101]))
    np.testing.assert_array_equal(f[0, :7], y[93:100])
    np.testing.assert_array_equal(f[1, :7], y[94:101])
    np.testing.assert_array_equal(f[0, 7:14], f[0, :7])


def test_aggregate_losses_is_the_mean_over_series():
    rng = np.random.default_rng(4)
    series = [rng.poisson(30, 800).astype(float) + 1 for _ in range(3)]
    origins = np.arange(m5.INITIAL, 800 - m5.HORIZON + 1)
    fc = {
        m: [rng.normal(30, 3, (len(origins), m5.HORIZON)) + i for i in range(3)] for m in m5.MODELS
    }
    table = m5.aggregate_losses(series, fc, origins)
    got = table.select("arima", "mase_h7")
    assert got["time"].tolist() == origins.tolist()
    manual = []
    for i, y in enumerate(series):
        scale = np.mean(np.abs(y[7 : m5.INITIAL] - y[: m5.INITIAL - 7]))
        manual.append(np.abs(y[origins + 6] - fc["arima"][i][:, 6]) / scale)
    np.testing.assert_allclose(got["value"], np.mean(manual, axis=0))
    assert set(table.metrics()) == {"mase_h1", "mase_h7", "mase_h14"}


def test_text_helpers():
    assert m5.join_names(["a"]) == "`a`"
    assert m5.join_names(["a", "b", "c"]) == "`a`, `b` and `c`"
    assert m5.bootstrap_p(0.0, 1000) == "< 0.001"
    assert m5.bootstrap_p(0.0312, 1000) == "0.0312"
    assert "all 3 models" in m5.mcs_sentence(["a", "b", "c"], [], 0.1)
    assert "only `a`" in m5.mcs_sentence(["a"], ["b", "c"], 0.1)
    both = m5.mcs_sentence(["a", "b"], ["c"], 0.1)
    assert "`c` was eliminated" in both and "`a` and `b`" in both
