"""Ticket 2.1 DoD: a Criteo-sized table (about 14M rows) aggregates with peak RSS < 1 GB."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from isbetter.simulate.ab_data import write_criteo_like_csv

ROWS = 14_000_000
LIMIT_MB = 1024

CHILD = textwrap.dedent(
    """
    import resource, sys
    from isbetter.adapters.ab_table import AggSpec, aggregate_ab_table
    agg = aggregate_ab_table(
        sys.argv[1], variant_col="treatment",
        specs=[AggSpec("visit", "visit"), AggSpec("conversion", "conversion")],
        variants=["0", "1"],
    )
    print(sum(agg.units.values()), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """
)


@pytest.mark.slow
def test_criteo_scale_aggregation_stays_under_1gb(tmp_path):
    path = tmp_path / "criteo_like.csv"
    tot = write_criteo_like_csv(path, ROWS, np.random.default_rng(11))
    out = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    rows, peak_kb = int(out[0]), int(out[1])
    assert rows == ROWS == tot["rows"]
    assert peak_kb / 1024 < LIMIT_MB, f"peak RSS {peak_kb / 1024:.0f} MB"
