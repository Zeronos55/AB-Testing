# CLAUDE.md

A hands-on A/B testing project. Goal: learn to compare A vs B properly, see the results
ourselves, and present the findings and conclusions in a Power BI dashboard.
The previous `isbetter` toolkit is archived under `archive/` (git tag `archive/isbetter-v1`).

## How we work (most important)
- Claude runs all analyses (the user's Windows laptop is low on resources). The user reviews
  results and gives insights. We decide each next step together.
- One small step at a time. Before running: say in plain English what the step does and why,
  and ask the user for a prediction. After running: show the numbers and a chart, compare them
  to the prediction, say what they mean, then stop and ask for the user's reading. Do not start
  the next step until the user has responded.
- Record what we found and what the user concluded in `notes/` (dated, short).
- Never present a result as settled if its check has not been run.

## Teaching style
- Explain like a sharp, down-to-earth MIT lecturer: simple language, everyday examples, an
  unconventional angle when it helps. Intuition first, then the exact definition in one sentence.
- Define any statistics term on first use. Name where each analogy stops working.
- Use 80/20, game theory, habit-building and similar ideas only where they genuinely fit.
- Accuracy beats style: if a vivid explanation and a correct one conflict, the correct one wins.

## Layout
- `data/` raw datasets (gitignored, never committed)
- `analysis/` one script per step, numbered (`01_explore.py`, `02_two_proportion.py` ...)
- `results/` CSV tables written by the analyses; these feed Power BI
- `dashboard/` Power BI page and measure spec (`SPEC.md`, DAX) and data-model notes
- `notes/` our findings and the user's conclusions
- `archive/` the old toolkit, read-only reference; do not edit or import from it

## Commands
```bash
uv sync
uv run python analysis/<step>.py
uv run pytest
uv run ruff check .
```

## Rules
1. Randomness goes through a seeded `numpy.random.Generator`; write the seed next to the result.
   Never use `np.random.seed` or global state.
2. Every hand-written test or interval is checked against a trusted library (SciPy,
   statsmodels). Each step prints the side-by-side comparison so the user sees it; turn it into
   an automatic test later as a safety net. Prefer calling the library over rewriting it.
3. Trust check before conclusions: for each method, run an A/A simulation (two identical
   groups) and report how often it wrongly finds a difference.
4. Don't weaken a test tolerance to make it pass; find the cause.
5. Keep memory light. Aggregate big tables first; Power BI only sees summary CSVs.

## Dashboard contract
- Each analysis writes tidy CSVs to `results/` (one row per comparison or check, stable column
  names). Do not change a column name without updating `dashboard/SPEC.md`.
- Power BI Desktop (Windows) is the dashboard. Claude writes the CSVs and the spec; the user
  builds the `.pbix` and writes the conclusions.

## Scope
Personal and local. No PyPI publishing, hosting, docs site or web apps.

## Workflow
- Small commits, one step each. Commit messages are a plain-English summary of the change.
  Open a PR only when the user asks.
- Don't commit `data/` or large generated outputs.

## Gotchas
- Name the dataset, its source, licence and known caveats in `notes/` before using it
  (for example, whether rows are users or impressions).
