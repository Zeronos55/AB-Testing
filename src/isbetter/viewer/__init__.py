"""Streamlit viewer for saved results (``isbetter run ... --json-out result.json``).

The viewer reads only the saved Result / ExperimentResult JSON. It never loads raw data and
never runs a test, so it opens in milliseconds and cannot disagree with the report.
"""
