import json

import numpy as np
import pytest

from isbetter.adapters.strategy_returns import (
    MissingTrialsError,
    TrialRegistry,
    make_panel,
    prices_to_returns,
    read_returns_csv,
    resolve_deflation,
    returns_table,
)
from isbetter.core.schema import SchemaError


def _r(seed, n=100, mu=0.0005):
    return np.random.default_rng(seed).normal(mu, 0.01, n)


def test_log_appends_one_json_line_per_call_and_never_rewrites(tmp_path):
    reg = TrialRegistry(tmp_path / "sub" / "trials.jsonl")
    r1 = reg.log("a", _r(1), {"w": 5})
    first = (tmp_path / "sub" / "trials.jsonl").read_text()
    reg.log("b", _r(2), {"w": 10})
    text = (tmp_path / "sub" / "trials.jsonl").read_text()
    assert text.startswith(first) and len(text.splitlines()) == 2
    rec = json.loads(text.splitlines()[0])
    assert rec["name"] == "a" and rec["params"] == {"w": 5} and rec["n_obs"] == 100
    assert rec["sharpe"] == pytest.approx(r1.sharpe)
    x = _r(1)
    assert r1.sharpe == pytest.approx(x.mean() / x.std(ddof=1))


def test_rerunning_a_variant_does_not_inflate_the_trial_count(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    for _ in range(3):
        reg.log("a", _r(1), {"w": 5})
    reg.log("a", _r(1), {"w": 6})  # different parameters: a different variant
    reg.log("b", _r(2))
    assert len(reg.records()) == 5
    assert reg.n_trials() == 3


def test_a_new_registry_object_reads_what_an_old_one_wrote(tmp_path):
    TrialRegistry(tmp_path / "t.jsonl").log("a", _r(1))
    reg = TrialRegistry(tmp_path / "t.jsonl")
    reg.log("b", _r(2))
    assert reg.n_trials() == 2


def test_sharpe_variance_is_the_sample_variance_across_distinct_trials(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    series = [_r(s, mu=0.0002 * s) for s in range(1, 6)]
    for i, s in enumerate(series):
        reg.log(f"v{i}", s)
    srs = [s.mean() / s.std(ddof=1) for s in series]
    assert reg.sharpe_variance() == pytest.approx(np.var(srs, ddof=1))


def test_decorator_logs_every_call_with_its_arguments(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")

    @reg.track
    def ma(fast, slow=50):
        return _r(fast + slow)

    out = ma(5)
    ma(fast=10, slow=20)
    assert isinstance(out, np.ndarray)
    trials = list(reg.trials().values())
    assert reg.n_trials() == 2
    assert {t.name for t in trials} == {"ma"}
    assert {(t.params["fast"], t.params["slow"]) for t in trials} == {(5, 50), (10, 20)}


def test_decorator_with_explicit_name(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")

    @reg.track(name="momentum")
    def f(k):
        return _r(k)

    f(3)
    assert next(iter(reg.trials().values())).name == "momentum"


def test_context_manager_logs_on_clean_exit(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with reg.trial("a", w=5) as t:
        t.returns = _r(1)
    assert reg.n_trials() == 1 and reg.records()[0].params == {"w": 5}


def test_context_manager_logs_nothing_when_the_block_raises(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with pytest.raises(RuntimeError), reg.trial("a") as t:
        t.returns = _r(1)
        raise RuntimeError("boom")
    assert reg.n_trials() == 0


def test_context_manager_without_returns_is_an_error(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match="without setting"), reg.trial("a"):
        pass


@pytest.mark.parametrize(
    "bad", [[0.1, 0.2], [0.1, np.nan, 0.2, 0.3], np.zeros(10), np.ones((4, 2))]
)
def test_log_rejects_unusable_returns(tmp_path, bad):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with pytest.raises(ValueError):
        reg.log("a", bad)
    assert reg.n_trials() == 0


def test_corrupt_registry_line_is_reported_with_its_number(tmp_path):
    p = tmp_path / "t.jsonl"
    TrialRegistry(p).log("a", _r(1))
    with open(p, "a") as fh:
        fh.write("not json\n")
    with pytest.raises(ValueError, match="line 2"):
        TrialRegistry(p).records()


def test_dsr_inputs_refuse_without_registry_or_n_trials():
    with pytest.raises(MissingTrialsError, match="how many strategy variants"):
        resolve_deflation()
    with pytest.raises(MissingTrialsError, match="sr_variance"):
        resolve_deflation(n_trials=100)
    with pytest.raises(MissingTrialsError):
        resolve_deflation(n_trials=0, sr_variance=0.1)


def test_dsr_inputs_from_explicit_values():
    d = resolve_deflation(n_trials=300, sr_variance=0.002)
    assert (d.n_trials, d.sr_variance, d.source) == (300, 0.002, "explicit")


def test_dsr_inputs_from_registry_and_overrides(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    for s in range(1, 5):
        reg.log(f"v{s}", _r(s))
    d = resolve_deflation(reg)
    assert d.n_trials == 4 and d.sr_variance == pytest.approx(reg.sharpe_variance())
    assert resolve_deflation(reg, n_trials=50).n_trials == 50  # tried more outside the registry
    assert resolve_deflation(reg, sr_variance=0.01).sr_variance == 0.01


def test_single_trial_registry_cannot_supply_a_variance(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    reg.log("a", _r(1))
    with pytest.raises(MissingTrialsError, match="at least 2"):
        resolve_deflation(reg)
    assert resolve_deflation(reg, sr_variance=0.01).n_trials == 1


def test_empty_registry_has_no_trials(tmp_path):
    assert TrialRegistry(tmp_path / "missing.jsonl").n_trials() == 0


def test_returns_table_is_a_valid_timeseries_table():
    table = returns_table({"a": _r(1, 50), "bh": _r(2, 50)}, times=np.arange(50) + 100)
    assert table.variants() == ["a", "bh"] and table.metrics() == ["return"]
    rows = table.select("a", "return")
    assert len(rows["value"]) == 50 and rows["time"][0] == 100
    np.testing.assert_allclose(rows["value"], _r(1, 50))


def test_returns_table_fails_loudly_on_non_monotone_time():
    t = np.arange(20)
    t[5] = 2
    with pytest.raises(SchemaError, match="strictly increasing"):
        returns_table({"a": _r(1, 20), "b": _r(2, 20)}, times=t)


def test_make_panel_validates_alignment():
    with pytest.raises(ValueError, match="different lengths"):
        make_panel({"a": _r(1, 20), "b": _r(2, 21)})
    with pytest.raises(ValueError, match="one entry per period"):
        make_panel({"a": _r(1, 20)}, times=np.arange(19))
    p = make_panel({"a": _r(1, 20), "b": _r(2, 20)})
    assert p.matrix.shape == (20, 2)
    np.testing.assert_allclose(p.series("b"), _r(2, 20))


def test_prices_to_returns():
    p = np.array([100.0, 110.0, 99.0])
    np.testing.assert_allclose(prices_to_returns(p), [0.1, -0.1])
    np.testing.assert_allclose(prices_to_returns(p, "log"), np.log([1.1, 0.9]))
    with pytest.raises(ValueError, match="positive"):
        prices_to_returns([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="kind"):
        prices_to_returns(p, "pct")


def test_read_returns_csv(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("date,a,bh\n2020-01-01,0.01,0.02\n2020-01-02,-0.01,0.00\n2020-01-03,0.02,0.01\n")
    panel = read_returns_csv(p)
    assert panel.names == ["a", "bh"] and panel.matrix.shape == (3, 2)
    assert list(panel.times) == ["2020-01-01", "2020-01-02", "2020-01-03"]
    (tmp_path / "bad.csv").write_text("date,a\n2020-01-01,x\n")
    with pytest.raises(SchemaError, match="non-numeric"):
        read_returns_csv(tmp_path / "bad.csv")
    (tmp_path / "nodate.csv").write_text("a\n0.1\n")
    with pytest.raises(SchemaError, match="no 'date'"):
        read_returns_csv(tmp_path / "nodate.csv")
