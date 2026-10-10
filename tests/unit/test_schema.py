import pytest

from isbetter.core.dependence import Clustered, CVOverlap, Paired, TimeSeries
from isbetter.core.schema import SchemaError, build_table


def base(**extra):
    d = {
        "unit_id": [1, 2, 1, 2],
        "variant": ["a", "a", "b", "b"],
        "metric": ["m"] * 4,
        "value": [1.0, 2.0, 1.5, 2.5],
    }
    d.update(extra)
    return d


def test_valid_table():
    t = build_table(base())
    assert len(t) == 4
    assert t.variants() == ["a", "b"]
    assert t.select("a", "m")["value"].tolist() == [1.0, 2.0]


def test_missing_required_column():
    d = base()
    del d["metric"]
    with pytest.raises(SchemaError, match="missing required"):
        build_table(d)


def test_nan_value():
    with pytest.raises(SchemaError, match="NaN"):
        build_table(base(value=[1.0, float("nan"), 1.0, 1.0]))


def test_non_numeric_value():
    with pytest.raises(SchemaError, match="numeric"):
        build_table(base(value=["x", "y", "z", "w"]))


def test_length_mismatch():
    with pytest.raises(SchemaError, match="same length"):
        build_table(base(pair_id=[1, 2]))


def test_duplicate_pair_rows():
    with pytest.raises(SchemaError, match="duplicate"):
        build_table(base(pair_id=[1, 1, 1, 2]))


def test_paired_ok():
    build_table(base(pair_id=[1, 2, 1, 2]), Paired())


def test_paired_missing_partner():
    d = base(pair_id=[1, 2, 1, 3])
    with pytest.raises(SchemaError, match="missing"):
        build_table(d, Paired())


def test_paired_requires_pair_id():
    with pytest.raises(SchemaError, match="pair_id"):
        build_table(base(), Paired())


def test_clusters_need_two():
    with pytest.raises(SchemaError, match="cluster"):
        build_table(base(cluster_id=[1, 1, 1, 2]), Clustered())
    build_table(base(cluster_id=[1, 2, 1, 2]), Clustered())


def test_clustered_requires_column():
    with pytest.raises(SchemaError, match="cluster_id"):
        build_table(base(), Clustered())


def test_time_must_increase():
    with pytest.raises(SchemaError, match="strictly increasing"):
        build_table(base(time=[1, 1, 1, 2]), TimeSeries())
    build_table(base(time=[1, 2, 1, 2]), TimeSeries())


def test_cv_overlap_validates_args():
    with pytest.raises(ValueError):
        CVOverlap(n_train=0, n_test=1, repeats=1, folds=2)
