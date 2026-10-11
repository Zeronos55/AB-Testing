import numpy as np
import pytest

from isbetter.adapters.ab_table import AggSpec, aggregate_ab_table, read_ab_table
from isbetter.core.dependence import Clustered, Iid
from isbetter.core.schema import SchemaError
from isbetter.core.sufficient import SufficientStats
from isbetter.simulate.ab_data import hillstrom_like, write_criteo_like_csv, write_csv


@pytest.fixture()
def hillstrom_csv(tmp_path):
    data = hillstrom_like(np.random.default_rng(3), counts=(300, 310, 290))
    path = tmp_path / "h.csv"
    write_csv(path, data)
    return path, data


ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]


def test_unit_level_long_table(hillstrom_csv):
    path, data = hillstrom_csv
    t = read_ab_table(
        path,
        variant_col="segment",
        value_cols=["visit", "spend"],
        covariate_cols=["history"],
        dependence=Iid(),
    )
    n = len(data["visit"])
    assert len(t) == 2 * n
    assert t.metrics() == ["spend", "visit"]
    assert t.variants() == sorted(ARMS)
    r = t.select("Mens E-Mail", "spend")
    mask = data["segment"] == "Mens E-Mail"
    assert np.allclose(np.sort(r["value"]), np.sort(data["spend"][mask]))
    assert "cov_history" in t.columns


def test_stats_path_matches_unit_path_exactly(hillstrom_csv):
    path, data = hillstrom_csv
    specs = [AggSpec("visit", "visit"), AggSpec("spend", "spend", "history")]
    agg = aggregate_ab_table(path, variant_col="segment", specs=specs, variants=ARMS)
    for arm in ARMS:
        m = data["segment"] == arm
        want = SufficientStats.from_arrays(data["spend"][m], data["history"][m])
        got = agg.get(arm, "spend")
        for f in ("n", "sum", "sum_sq", "sum_x", "sum_x_sq", "sum_xy"):
            assert getattr(got, f) == pytest.approx(getattr(want, f), rel=1e-12), f
        assert agg.get(arm, "visit").sum == data["visit"][m].sum()
        assert agg.units[arm] == int(m.sum())


def test_cluster_column_written_under_canonical_name(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("arm,y,user\na,1,u1\na,2,u1\na,3,u2\nb,4,u3\nb,5,u3\nb,6,u4\n")
    t = read_ab_table(
        p,
        variant_col="arm",
        value_cols=["y"],
        cluster_col="user",
        cluster_name="cluster_id",
        dependence=Clustered(),
    )
    assert set(t.columns["cluster_id"]) == {"u1", "u2", "u3", "u4"}


def test_parquet_roundtrip(tmp_path, hillstrom_csv):
    import polars as pl

    path, data = hillstrom_csv
    pq = tmp_path / "h.parquet"
    pl.read_csv(path).write_parquet(pq)
    agg = aggregate_ab_table(
        pq, variant_col="segment", specs=[AggSpec("visit", "visit")], variants=ARMS
    )
    assert sum(agg.units.values()) == len(data["visit"])


def test_missing_column_fails_loudly(hillstrom_csv):
    path, _ = hillstrom_csv
    with pytest.raises(SchemaError, match="nope"):
        read_ab_table(path, variant_col="segment", value_cols=["nope"])
    with pytest.raises(SchemaError):
        aggregate_ab_table(path, variant_col="segment", specs=[AggSpec("m", "nope")], variants=ARMS)


def test_nulls_fail_loudly_on_both_paths(tmp_path):
    p = tmp_path / "n.csv"
    p.write_text("arm,y\na,1\na,\nb,3\nb,4\n")
    with pytest.raises(SchemaError, match="missing"):
        read_ab_table(p, variant_col="arm", value_cols=["y"])
    with pytest.raises(SchemaError, match="missing"):
        aggregate_ab_table(p, variant_col="arm", specs=[AggSpec("y", "y")], variants=["a", "b"])


def test_unlisted_and_absent_arms_fail_loudly(hillstrom_csv):
    path, _ = hillstrom_csv
    spec = [AggSpec("visit", "visit")]
    with pytest.raises(SchemaError, match="not named in the config"):
        aggregate_ab_table(path, variant_col="segment", specs=spec, variants=ARMS[:2])
    with pytest.raises(SchemaError, match="no rows"):
        aggregate_ab_table(path, variant_col="segment", specs=spec, variants=[*ARMS, "ghost"])


def test_non_numeric_value_fails_loudly(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("arm,y\na,x\nb,2\n")
    with pytest.raises(SchemaError):
        aggregate_ab_table(p, variant_col="arm", specs=[AggSpec("y", "y")], variants=["a", "b"])
    with pytest.raises(SchemaError, match="numeric"):
        read_ab_table(p, variant_col="arm", value_cols=["y"])


def test_criteo_like_aggregation_exact(tmp_path):
    p = tmp_path / "c.csv"
    tot = write_criteo_like_csv(p, 50_000, np.random.default_rng(5), chunk=7_000)
    agg = aggregate_ab_table(
        p,
        variant_col="treatment",
        specs=[AggSpec("visit", "visit"), AggSpec("conversion", "conversion")],
        variants=["0", "1"],
    )
    assert agg.units == {"0": tot["n0"], "1": tot["n1"]}
    assert agg.get("1", "visit").sum == tot["visit1"]
    assert agg.get("0", "conversion").sum == tot["conv0"]
    assert agg.get("1", "conversion").sum_sq == tot["conv1"]  # binary: y^2 == y
