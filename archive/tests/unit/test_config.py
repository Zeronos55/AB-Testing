import pytest
import yaml

from isbetter.core.config import ConfigError, parse_config
from isbetter.core.dependence import CVOverlap, Paired

EXAMPLE = """
experiment: fraud_models_v1
adapter: sklearn_cv
data: results/fraud_cv_scores.parquet
variants: {baseline: logreg, candidates: [lightgbm, rf]}
metric: {name: pr_auc, higher_is_better: true}
dependence: {type: cv_overlap, n_train: 900, n_test: 100, repeats: 10, folds: 10}
decision: {min_effect: 0.01, alpha: 0.05, correction: holm}
seed: 20261010
report: {format: [markdown, html], include_simulation_check: true}
"""


def test_example_parses():
    cfg = parse_config(yaml.safe_load(EXAMPLE))
    assert cfg.decision.correction == "holm"
    assert cfg.to_dependence() == CVOverlap(900, 100, 10, 10)


def test_paired_minimal():
    cfg = parse_config(
        {
            "experiment": "x",
            "data": "d.csv",
            "variants": {"baseline": "a", "candidates": ["b"]},
            "metric": {"name": "m"},
            "dependence": {"type": "paired"},
        }
    )
    assert cfg.to_dependence() == Paired()


def test_bad_config_message_is_readable():
    with pytest.raises(ConfigError) as e:
        parse_config({"experiment": "x", "dependence": {"type": "nope"}})
    msg = str(e.value)
    assert "dependence" in msg and "data" in msg


def test_alpha_range_and_unknown_key():
    base = {
        "experiment": "x",
        "data": "d.csv",
        "variants": {"baseline": "a", "candidates": ["b"]},
        "metric": {"name": "m"},
        "dependence": {"type": "iid"},
    }
    with pytest.raises(ConfigError, match="alpha"):
        parse_config({**base, "decision": {"alpha": 2}})
    with pytest.raises(ConfigError, match="bogus"):
        parse_config({**base, "bogus": 1})


def test_baseline_not_candidate():
    with pytest.raises(ConfigError):
        parse_config(
            {
                "experiment": "x",
                "data": "d.csv",
                "variants": {"baseline": "a", "candidates": ["a"]},
                "metric": {"name": "m"},
                "dependence": {"type": "iid"},
            }
        )
