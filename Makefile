# Local equivalents of the CI jobs. Personal, unpublished project: nothing here uploads anything.
.PHONY: check calibrate gallery cases all

check:  ## what CI runs on every push: lint, strict types, fast tests
	uv run ruff check .
	uv run mypy src/isbetter/core src/isbetter/methods src/isbetter/decision
	uv run pytest

calibrate:  ## slow Type I error / coverage tests (10,000 replications; takes a while)
	uv run pytest -m slow

gallery:  ## failure-gallery scripts (write figures and *_results.md under validation/failure_gallery/)
	uv run python validation/failure_gallery/01_naive_cv_ttest.py
	uv run python validation/failure_gallery/02_iid_bootstrap_sharpe.py
	uv run python validation/failure_gallery/02b_dm_horizon.py
	uv run python validation/failure_gallery/03_clt_small_n.py
	uv run python validation/failure_gallery/04_peeking.py
	uv run python validation/failure_gallery/05_ignored_clustering.py

cases:  ## the five synthetic case studies (outputs go to case_studies/*/out/, not committed)
	uv run python case_studies/01_fraud_classifiers/run.py
	uv run python case_studies/02_ab_hillstrom_criteo/run.py
	uv run python case_studies/03_forecasting_m5/run.py
	uv run python case_studies/04_trading_edge/run.py
	uv run python case_studies/05_llm_eval/run.py

all: check calibrate gallery
