import numpy as np
import pytest

from isbetter.adapters.llm_eval import (
    CacheMiss,
    EvalItem,
    ItemScore,
    ResponseCache,
    cache_key,
    item_scores_table,
    read_item_scores,
    records_from_mapping,
    scores_from_cache,
    write_item_scores,
)
from isbetter.core.dependence import Clustered, Paired
from isbetter.core.schema import SchemaError


def _recs(n=6, clustered=True):
    out = []
    for m in ("a", "b"):
        for i in range(n):
            out.append(ItemScore(m, f"q{i}", float(i % 2), f"p{i // 2}" if clustered else None))
    return out


def test_table_pairs_by_item_and_validates_pairing():
    t = item_scores_table(_recs(), "acc", Paired())
    assert t.variants() == ["a", "b"] and t.metrics() == ["acc"]
    assert (t.columns["pair_id"] == t.columns["unit_id"]).all()
    assert "cluster_id" in t


def test_missing_partner_fails_loudly():
    recs = _recs()[:-1]  # model b lacks q5
    with pytest.raises(SchemaError, match="missing"):
        item_scores_table(recs, "acc", Paired())


def test_cluster_column_is_named_after_dependence_and_partial_clusters_rejected():
    t = item_scores_table(_recs(), "acc", Clustered("passage"))
    assert "passage" in t and "cluster_id" not in t
    recs = _recs()
    recs[0] = ItemScore("a", "q0", 0.0, None)
    with pytest.raises(SchemaError, match="no cluster_id"):
        item_scores_table(recs, "acc", Clustered())


def test_clustered_without_any_cluster_id_fails():
    with pytest.raises(SchemaError, match="cluster_id"):
        item_scores_table(_recs(clustered=False), "acc", Clustered())


def test_roundtrip_through_jsonl_and_booleans(tmp_path):
    p = tmp_path / "scores.jsonl"
    write_item_scores(p, _recs())
    t = read_item_scores(p, Paired(), metric="acc")
    assert len(t) == 12 and set(t.columns["value"]) == {0.0, 1.0}
    p.write_text(
        '{"model":"a","item_id":1,"score":true}\n\n{"model":"b","item_id":1,"score":false}\n'
    )
    t = read_item_scores(p, Paired())
    assert t.metrics() == ["score"]
    assert sorted(t.columns["value"]) == [0.0, 1.0]


@pytest.mark.parametrize(
    ("line", "msg"),
    [
        ("not json", ":1: not valid JSON"),
        ('{"model":"a","item_id":"x"}', "missing field"),
        ('{"model":"a","item_id":"x","score":"high"}', "must be a number"),
        ('["a"]', "expected a JSON object"),
        ('{"model":"a","item_id":"x","score":NaN}', "NaN"),
    ],
)
def test_bad_lines_report_file_and_line(tmp_path, line, msg):
    p = tmp_path / "s.jsonl"
    p.write_text(line + "\n")
    with pytest.raises(SchemaError, match=msg):
        read_item_scores(p)


def test_missing_file_and_empty_file(tmp_path):
    with pytest.raises(SchemaError, match="cannot read"):
        read_item_scores(tmp_path / "nope.jsonl")
    (tmp_path / "e.jsonl").write_text("\n")
    with pytest.raises(SchemaError, match="no item scores"):
        read_item_scores(tmp_path / "e.jsonl")


def test_records_from_mapping():
    scores = {"a": {"q1": 1, "q2": 0, "q3": 1}, "b": {"q1": 0, "q2": 0, "q3": 1}}
    recs = records_from_mapping(scores, {"q1": "p1", "q2": "p1", "q3": "p2"})
    t = item_scores_table(recs, "acc", Clustered())
    assert len(t) == 6 and set(t.columns["cluster_id"]) == {"p1", "p2"}


# -- cache ----------------------------------------------------------------------------------
def test_cache_key_depends_on_every_component():
    base = cache_key("m", "prompt", "q1", 0)
    assert base == cache_key("m", "prompt", "q1", 0)
    for other in (
        cache_key("m2", "prompt", "q1", 0),
        cache_key("m", "prompt2", "q1", 0),
        cache_key("m", "prompt", "q2", 0),
        cache_key("m", "prompt", "q1", 1),
    ):
        assert other != base
    # components cannot be shifted into one another
    assert cache_key("m", "ab", "c", 0) != cache_key("m", "a", "bc", 0)


def test_cache_calls_once_persists_and_counts(tmp_path):
    path = tmp_path / "c" / "cache.jsonl"
    calls = []

    def make(text):
        def call():
            calls.append(text)
            return text

        return call

    c = ResponseCache(path)
    assert c.get_or_call("m", "p", "q1", 0, make("A")) == "A"
    assert c.get_or_call("m", "p", "q1", 0, make("B")) == "A"  # hit: not called again
    assert c.get_or_call("m", "p", "q1", 1, make("C")) == "C"  # other seed is another key
    assert calls == ["A", "C"] and (c.hits, c.misses) == (1, 2) and len(c) == 2

    again = ResponseCache(path, offline=True)  # a new process reads the file
    assert len(again) == 2
    assert again.get_or_call("m", "p", "q1", 0) == "A"
    with pytest.raises(CacheMiss, match="offline"):
        again.get_or_call("m", "p", "q9", 0, make("never"))
    assert "never" not in calls


def test_cache_without_call_raises_and_latest_put_wins(tmp_path):
    c = ResponseCache(tmp_path / "c.jsonl")
    with pytest.raises(CacheMiss, match="no `call`"):
        c.get_or_call("m", "p", "q", 0)
    c.put("m", "p", "q", 0, "old")
    c.put("m", "p", "q", 0, "new")
    assert ResponseCache(tmp_path / "c.jsonl").get("m", "p", "q", 0) == "new"
    assert len((tmp_path / "c.jsonl").read_text().splitlines()) == 2  # append-only
    with pytest.raises(TypeError):
        c.put("m", "p", "q", 0, 3)  # type: ignore[arg-type]


def test_cache_stores_prompt_hash_not_prompt(tmp_path):
    c = ResponseCache(tmp_path / "c.jsonl")
    c.put("m", "SECRET PROMPT TEXT", "q", 0, "r")
    text = (tmp_path / "c.jsonl").read_text()
    assert "SECRET PROMPT TEXT" not in text and "prompt_sha256" in text


def test_corrupt_cache_line_is_reported(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"key": "k", "response": "r"}\n{oops\n')
    with pytest.raises(ValueError, match=r"c\.jsonl:2"):
        ResponseCache(p)


def test_scores_from_cache_scores_and_refuses_incomplete_cache(tmp_path):
    c = ResponseCache(tmp_path / "c.jsonl")
    items = [EvalItem("q1", "2+2?", "p1"), EvalItem("q2", "3+3?", "p1")]
    for m in ("a", "b"):
        c.put(m, "2+2?", "q1", 0, "4")
        c.put(m, "3+3?", "q2", 0, "6" if m == "a" else "7")

    def scorer(model, item, response):
        return float(response == {"q1": "4", "q2": "6"}[item.item_id])

    recs = scores_from_cache(c, ["a", "b"], items, scorer, metric="acc")
    assert {(r.model, r.item_id): r.score for r in recs} == {
        ("a", "q1"): 1.0,
        ("a", "q2"): 1.0,
        ("b", "q1"): 1.0,
        ("b", "q2"): 0.0,
    }
    with pytest.raises(CacheMiss):
        scores_from_cache(c, ["a", "c"], items, scorer)
    with pytest.raises(ValueError, match="duplicate"):
        scores_from_cache(c, ["a"], items + items[:1], scorer)


def test_end_to_end_cli_runs_llm_eval_adapter(tmp_path):
    from typer.testing import CliRunner

    from isbetter.cli import app

    rng = np.random.default_rng(0)
    recs = []
    for i in range(60):
        base = rng.random() < 0.6
        recs.append(ItemScore("base", f"q{i}", float(base), f"p{i // 3}"))
        recs.append(ItemScore("cand", f"q{i}", float(base or rng.random() < 0.3), f"p{i // 3}"))
    write_item_scores(tmp_path / "s.jsonl", recs)
    (tmp_path / "c.yaml").write_text(
        "experiment: llm_toy\nadapter: llm_eval\ndata: s.jsonl\n"
        "variants: {baseline: base, candidates: [cand]}\n"
        "metric: {name: acc}\ndependence: {type: paired}\n"
        "decision: {min_effect: 0.02}\nseed: 1\n"
    )
    res = CliRunner().invoke(
        app, ["run", str(tmp_path / "c.yaml"), "--out", str(tmp_path / "R.md")]
    )
    assert res.exit_code == 0, res.output
    assert "cand vs base" in res.output
