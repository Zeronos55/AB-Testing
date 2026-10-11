import numpy as np
import pytest

from isbetter.adapters.forecast_backtest import (
    backtest_table,
    rolling_origin_backtest,
    seasonal_naive,
    seasonal_scale,
)
from isbetter.core.dependence import TimeSeries
from isbetter.core.schema import SchemaError, validate

Y = np.array([10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0])


def test_seasonal_naive_repeats_last_season():
    f = seasonal_naive(3)(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 7)
    assert f.tolist() == [3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0]
    assert seasonal_naive(1)(np.array([1.0, 9.0]), 3).tolist() == [9.0, 9.0, 9.0]
    with pytest.raises(ValueError, match="at least 3"):
        seasonal_naive(3)(np.array([1.0, 2.0]), 1)
    with pytest.raises(ValueError, match="season"):
        seasonal_naive(0)


def test_seasonal_scale_hand_computed():
    # diffs of [10, 12, 11, 13]: 2, -1, 2  -> mean abs 5/3, mean square 3
    assert seasonal_scale(Y[:4]) == pytest.approx(5 / 3)
    assert seasonal_scale(Y[:4], squared=True) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="more than"):
        seasonal_scale(Y[:2], season=2)
    with pytest.raises(ValueError, match="zero"):
        seasonal_scale(np.ones(5))


def test_losses_hand_computed_across_horizons():
    # origins 4 and 5, H = 2: targets Y[4:6] = [15, 14] and Y[5:7] = [14, 16]
    fa = np.array([[14.0, 14.0], [15.0, 15.0]])
    fb = np.array([[15.0, 13.0], [12.0, 17.0]])
    out = backtest_table(Y, {"a": fa, "b": fb}, [4, 5], loss="mae")
    assert out.metrics == ["mae_h1", "mae_h2"]
    sel = out.table.select("a", "mae_h2")
    # |14-14|=0 at origin 4 (target Y[5]); |16-15|=1 at origin 5 (target Y[6])
    assert sel["value"].tolist() == [0.0, 1.0]
    assert out.table.select("b", "mae_h1")["value"].tolist() == [0.0, 2.0]
    assert out.table.select("b", "mae_h2")["value"].tolist() == [1.0, 1.0]
    assert sel["time"].tolist() == [4, 5]


def test_mase_and_rmsse_scaling():
    fa = np.array([[14.0], [15.0]])
    fb = np.array([[15.0], [12.0]])
    mase = backtest_table(Y, {"a": fa, "b": fb}, [4, 5], loss="mase")
    # scale from Y[:4] (first training window) = 5/3
    assert mase.scale == pytest.approx(5 / 3)
    assert mase.table.select("b", "mase_h1")["value"].tolist() == pytest.approx([0.0, 2 / (5 / 3)])
    rmsse = backtest_table(Y, {"a": fa, "b": fb}, [4, 5], loss="rmsse", initial=4)
    assert rmsse.scale == pytest.approx(3.0)
    # origin 5 for b: error 14 - 12 = 2, squared 4, scaled by 3
    assert rmsse.table.select("b", "rmsse_h1")["value"][1] == pytest.approx(4 / 3)


def test_horizon_subset_and_custom_times():
    f = np.zeros((2, 3)) + 13.0
    out = backtest_table(
        Y, {"a": f, "b": f + 1}, [3, 4], loss="mae", horizons=[1, 3], times=np.arange(8) * 7
    )
    assert out.metrics == ["mae_h1", "mae_h3"]
    assert out.table.select("a", "mae_h3")["time"].tolist() == [21, 28]


def test_table_passes_schema_for_timeseries():
    out = backtest_table(Y, {"a": np.zeros((2, 1)), "b": np.ones((2, 1))}, [4, 5], loss="mae")
    validate(out.table, TimeSeries())  # time monotone within each (variant, metric)


def test_rolling_origin_backtest_is_paired_and_leak_free():
    seen: list[int] = []

    def spy(history: np.ndarray, h: int) -> np.ndarray:
        seen.append(len(history))
        return np.full(h, history[-1])

    y = np.arange(20, dtype=float) ** 1.5
    out = rolling_origin_backtest(
        y, {"naive": seasonal_naive(1), "spy": spy}, initial=10, horizon=3, step=2, loss="mae"
    )
    assert seen == [10, 12, 14, 16]  # each forecaster sees only y[:origin]
    assert out.origins.tolist() == [10, 12, 14, 16]  # 16 + 3 <= 20; 18 + 3 > 20
    a, b = out.table.select("naive", "mae_h2"), out.table.select("spy", "mae_h2")
    assert a["pair_id"].tolist() == b["pair_id"].tolist()
    np.testing.assert_allclose(a["value"], b["value"])  # same forecaster, same losses
    np.testing.assert_allclose(out.forecasts["naive"][1], np.full(3, y[11]))
    assert out.actuals.shape == (4, 3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"loss": "mape"}, "loss must be"),
        ({"origins": [4]}, "at least 2"),
        ({"origins": [5, 4]}, "increasing"),
        ({"origins": [4, 8]}, "inside the series"),
        ({"horizons": [2]}, "horizons must be"),
        ({"horizons": [1, 1]}, "horizons must be"),
        ({"times": np.arange(3)}, "times must"),
    ],
)
def test_backtest_table_validation(kwargs, match):
    f = {"a": np.zeros((2, 1)), "b": np.ones((2, 1))}
    args = {"origins": [4, 5], "loss": "mae"} | kwargs
    with pytest.raises(ValueError, match=match):
        backtest_table(Y, f, **args)


def test_backtest_table_rejects_bad_forecasts_and_series():
    good = np.zeros((2, 1))
    with pytest.raises(ValueError, match="at least two variants"):
        backtest_table(Y, {"a": good}, [4, 5], loss="mae")
    with pytest.raises(ValueError, match="shape"):
        backtest_table(Y, {"a": good, "b": np.zeros((3, 1))}, [4, 5], loss="mae")
    with pytest.raises(ValueError, match="NaN"):
        backtest_table(Y, {"a": good, "b": np.array([[np.nan], [1.0]])}, [4, 5], loss="mae")
    with pytest.raises(ValueError, match="NaN"):
        backtest_table(np.array([1.0, np.nan, 3.0, 4.0]), {"a": good, "b": good}, [1, 2], "mae")


def test_non_monotone_times_fail_loudly():
    f = {"a": np.zeros((2, 1)), "b": np.ones((2, 1))}
    times = np.array([0, 1, 2, 3, 5, 4, 6, 7])  # origin stamps 5 then 4
    with pytest.raises(SchemaError, match="strictly increasing"):
        backtest_table(Y, f, [4, 5], loss="mae", times=times)


def test_rolling_origin_validation():
    with pytest.raises(ValueError, match="must not exceed"):
        rolling_origin_backtest(Y, {"a": seasonal_naive(1)}, initial=7, horizon=3)
    with pytest.raises(ValueError, match="at least 2 forecast origins"):
        rolling_origin_backtest(Y, {"a": seasonal_naive(1), "b": seasonal_naive(1)}, 6, 2)

    def bad(history: np.ndarray, h: int) -> np.ndarray:
        return np.zeros(h + 1)

    with pytest.raises(ValueError, match="returned shape"):
        rolling_origin_backtest(Y, {"a": bad, "b": bad}, initial=3, horizon=2)
