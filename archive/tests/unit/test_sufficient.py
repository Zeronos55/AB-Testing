import numpy as np
import pytest

from isbetter.core.sufficient import AggregatedTable, SufficientStats


def test_from_arrays_matches_numpy():
    rng = np.random.default_rng(0)
    y, x = rng.normal(5, 2, 500), rng.normal(1, 3, 500)
    s = SufficientStats.from_arrays(y, x)
    assert s.n == 500
    assert s.mean == pytest.approx(y.mean())
    assert s.var == pytest.approx(y.var(ddof=1))
    assert s.mean_x == pytest.approx(x.mean())
    assert s.var_x == pytest.approx(x.var(ddof=1))
    assert s.cov_xy == pytest.approx(np.cov(x, y, ddof=1)[0, 1])


def test_merge_equals_whole():
    rng = np.random.default_rng(1)
    y, x = rng.normal(size=300), rng.normal(size=300)
    whole = SufficientStats.from_arrays(y, x)
    merged = SufficientStats.from_arrays(y[:120], x[:120]) + SufficientStats.from_arrays(
        y[120:], x[120:]
    )
    for f in ("n", "sum", "sum_sq", "sum_x", "sum_x_sq", "sum_xy"):
        assert getattr(merged, f) == pytest.approx(getattr(whole, f))


def test_merge_requires_matching_x():
    a = SufficientStats.from_arrays(np.arange(3.0))
    b = SufficientStats.from_arrays(np.arange(3.0), np.arange(3.0))
    with pytest.raises(ValueError, match="with and without"):
        a + b


def test_x_properties_need_x():
    with pytest.raises(ValueError, match="no x moments"):
        _ = SufficientStats.from_arrays(np.arange(4.0)).var_x


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        (dict(n=0, sum=0.0, sum_sq=0.0), "n >= 1"),
        (dict(n=3, sum=float("nan"), sum_sq=1.0), "not finite"),
        (dict(n=3, sum=1.0, sum_sq=-1.0), "non-negative"),
        (dict(n=2, sum=10.0, sum_sq=1.0), "inconsistent"),
        (dict(n=3, sum=3.0, sum_sq=5.0, sum_x=1.0), "together"),
    ],
)
def test_invalid_moments_rejected(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        SufficientStats(**kwargs)


def test_from_arrays_rejects_bad_input():
    with pytest.raises(ValueError, match="NaN"):
        SufficientStats.from_arrays(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="same length"):
        SufficientStats.from_arrays(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="non-empty"):
        SufficientStats.from_arrays(np.array([]))


def test_aggregated_table_lookup():
    s = SufficientStats(2, 1.0, 1.0)
    t = AggregatedTable({("a", "m"): s}, {"a": 2})
    assert t.get("a", "m") is s
    assert t.variants() == ["a"] and t.metrics() == ["m"]
    with pytest.raises(KeyError, match="no aggregated data"):
        t.get("b", "m")
