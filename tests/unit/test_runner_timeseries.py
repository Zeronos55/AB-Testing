import numpy as np
import pytest

from isbetter.adapters.forecast_backtest import (
    backtest_table,
    rolling_origin_backtest,
    seasonal_naive,
)
from isbetter.core.config import parse_config
from isbetter.core.dependence import TimeSeries
from isbetter.core.plan import build_plan
from isbetter.core.runner import run
from isbetter.core.schema import CanonicalTable, SchemaError, build_table
from isbetter.methods.forecasting import diebold_mariano
from isbetter.report.markdown import render_markdown


def make_cfg(horizon=1, metric="mae_h1", higher=False, **dep):
    return parse_config(
        {
            "experiment": "ts",
            "data": "d.csv",
            "variants": {"baseline": "naive", "candidates": ["smart"]},
            "metric": {"name": metric, "higher_is_better": higher, "min_effect": 0.0},
            "dependence": {"type": "timeseries", "horizon": horizon, **dep},
            "seed": 7,
        }
    )


def two_forecasters(n=240, h=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.standard_normal(n)) + 0.5 * np.sin(np.arange(n) / 3)
    return rolling_origin_backtest(
        y,
        {"naive": seasonal_naive(1), "smart": lambda hist, k: np.full(k, hist[-3:].mean())},
        initial=60,
        horizon=h,
        loss="mae",
    )


def test_end_to_end_matches_direct_dm():
    out = two_forecasters(h=3)
    cfg = make_cfg(horizon=3, metric="mae_h3")
    plan = build_plan(cfg)
    res = run(plan, out.table)
    comp = res.comparisons[0]
    la = out.table.select("smart", "mae_h3")["value"]
    lb = out.table.select("naive", "mae_h3")["value"]
    ref = diebold_mariano(la, lb, horizon=3)
    assert comp.primary.method == "diebold_mariano_hln"
    assert comp.primary.estimate == pytest.approx(ref.estimate)
    assert comp.primary.p_value == pytest.approx(ref.p_value)
    assert comp.cross_checks[0].method == "stationary_block_bootstrap"
    # lower loss is better: a negative candidate-baseline estimate is a win
    assert comp.oriented_estimate == pytest.approx(-ref.estimate)


def test_block_length_diagnostics_default_and_user_set():
    out = two_forecasters()
    auto = run(build_plan(make_cfg(metric="mae_h1", horizon=1)), out.table)
    diag = auto.comparisons[0].cross_checks[0].diagnostics
    assert diag["block_len_source"] in ("politis_white", "rule_of_thumb")
    assert set(diag["suggested_block_len"]) == {"stationary", "circular", "source"}
    fixed = run(build_plan(make_cfg(metric="mae_h1", horizon=1, block_len=9)), out.table)
    diag = fixed.comparisons[0].cross_checks[0].diagnostics
    assert diag["block_len"] == 9 and diag["block_len_source"] == "user"


def test_report_shows_dm_rationale_and_block_length():
    out = two_forecasters()
    res = run(build_plan(make_cfg(metric="mae_h1", horizon=1)), out.table)
    text = render_markdown(res)
    assert "Diebold-Mariano" in text and "Block length" in text and "automatic suggestion" in text
    assert "diebold_mariano_hln" in text and "stationary_block_bootstrap" in text


def test_result_json_round_trip():
    from isbetter.core.result import Result

    out = two_forecasters()
    res = run(build_plan(make_cfg(metric="mae_h1", horizon=1)), out.table)
    again = Result.from_json(res.to_json())
    assert again.comparisons[0].primary.p_value == res.comparisons[0].primary.p_value


def test_declared_horizon_must_match_the_data():
    out = two_forecasters(h=3)
    plan = build_plan(make_cfg(horizon=1, metric="mae_h3"))
    with pytest.raises(SchemaError, match="horizon"):
        run(plan, out.table)


def test_variants_must_share_origins():
    cols = {
        "unit_id": ["a", "b", "c", "a", "b", "d"],
        "variant": ["naive"] * 3 + ["smart"] * 3,
        "metric": ["m"] * 6,
        "value": [1.0, 2.0, 1.5, 0.5, 2.5, 1.0],
        "time": [1, 2, 3, 1, 2, 4],
    }
    table = build_table(cols, TimeSeries())
    with pytest.raises(SchemaError, match="same forecast origins"):
        run(build_plan(make_cfg(metric="m")), table)


def test_origins_are_ordered_by_time_not_by_row_or_id_order():
    out = two_forecasters(h=3)
    perm = np.random.default_rng(0).permutation(len(out.table))
    shuffled = {k: v[perm] for k, v in out.table.columns.items()}
    with pytest.raises(SchemaError, match="strictly increasing"):  # the schema contract
        build_table(shuffled, TimeSeries())
    # and the runner re-orders by `time` even for a table that skipped validation
    plan = build_plan(make_cfg(horizon=3, metric="mae_h3"))
    a, b = run(plan, out.table), run(plan, CanonicalTable(shuffled))
    assert a.comparisons[0].primary.p_value == pytest.approx(b.comparisons[0].primary.p_value)


def test_backtest_table_feeds_the_runner_directly():
    y = np.random.default_rng(1).standard_normal(120).cumsum()
    f1 = np.repeat(y[39:99, None], 2, axis=1)
    f2 = f1 + 0.3
    out = backtest_table(y, {"naive": f1, "smart": f2}, np.arange(40, 100), loss="mase")
    res = run(build_plan(make_cfg(metric="mase_h1")), out.table)
    assert res.comparisons[0].n_pairs == 60
