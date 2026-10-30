"""Synthetic A/B datasets shaped like the public ones used in the case study.

These are STAND-INS. The real Hillstrom and Criteo files could not be downloaded where this
was built, so the rates below are assumptions chosen to be plausible, not measurements.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HILLSTROM_ARMS = ("No E-Mail", "Mens E-Mail", "Womens E-Mail")
HILLSTROM_COUNTS = (21_306, 21_307, 21_387)  # arm sizes quoted for the real data (control first)
# Assumed per-arm visit and conversion probabilities (control, mens, womens).
_VISIT = (0.106, 0.183, 0.151)
_CONV_GIVEN_VISIT = (0.054, 0.068, 0.058)


def hillstrom_like(
    rng: np.random.Generator, counts: tuple[int, int, int] = HILLSTROM_COUNTS
) -> dict[str, np.ndarray]:
    """64,000-row email experiment: columns ``segment, history, visit, conversion, spend``.

    ``history`` is last year's spend, a pre-experiment covariate that predicts who converts,
    so CUPED has something to work with. Outcomes are zero-inflated, as in real e-commerce.
    """
    seg, hist, visit, conv, spend = [], [], [], [], []
    for arm, n in zip(HILLSTROM_ARMS, counts, strict=True):
        k = HILLSTROM_ARMS.index(arm)
        h = np.round(rng.lognormal(mean=4.6, sigma=0.9, size=n), 2)
        z = (np.log(h) - 4.6) / 0.9  # standardised log history
        p_visit = 1 / (1 + np.exp(-(np.log(_VISIT[k] / (1 - _VISIT[k])) + 0.45 * z)))
        v = rng.random(n) < p_visit
        p_conv = 1 / (
            1 + np.exp(-(np.log(_CONV_GIVEN_VISIT[k] / (1 - _CONV_GIVEN_VISIT[k])) + 0.6 * z))
        )
        c = v & (rng.random(n) < p_conv)
        amount = np.round(rng.lognormal(mean=3.9 + 0.35 * z, sigma=0.7, size=n), 2)
        seg.append(np.full(n, arm))
        hist.append(h)
        visit.append(v.astype(np.int8))
        conv.append(c.astype(np.int8))
        spend.append(np.where(c, amount, 0.0))
    order = rng.permutation(sum(counts))
    cat = lambda xs: np.concatenate(xs)[order]  # noqa: E731
    return {
        "segment": cat(seg),
        "history": cat(hist),
        "visit": cat(visit),
        "conversion": cat(conv),
        "spend": cat(spend),
    }


def write_csv(path: str | Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    n = len(columns[names[0]])
    with open(path, "w", newline="") as fh:
        fh.write(",".join(names) + "\n")
        for i in range(n):
            fh.write(",".join(str(columns[k][i]) for k in names) + "\n")


def write_criteo_like_csv(
    path: str | Path,
    rows: int,
    rng: np.random.Generator,
    treated_share: float = 0.85,
    visit_rate: tuple[float, float] = (0.040, 0.047),
    conv_given_visit: tuple[float, float] = (0.045, 0.060),
    chunk: int = 1_000_000,
) -> dict[str, int]:
    """Write a ``treatment,visit,conversion`` CSV in chunks (memory O(chunk)).

    Mimics the shape of the Criteo uplift data: about 85% treated, rare binary outcomes.
    Returns exact counts of rows and successes per arm for checking the aggregation.
    """
    tot = {"rows": 0, "n0": 0, "n1": 0, "visit0": 0, "visit1": 0, "conv0": 0, "conv1": 0}
    with open(path, "w", newline="") as fh:
        fh.write("treatment,visit,conversion\n")
        done = 0
        while done < rows:
            m = min(chunk, rows - done)
            t = (rng.random(m) < treated_share).astype(np.int8)
            pv = np.where(t == 1, visit_rate[1], visit_rate[0])
            v = (rng.random(m) < pv).astype(np.int8)
            pc = np.where(t == 1, conv_given_visit[1], conv_given_visit[0])
            c = (v.astype(bool) & (rng.random(m) < pc)).astype(np.int8)
            np.savetxt(fh, np.column_stack([t, v, c]), fmt="%d", delimiter=",")
            for arm in (0, 1):
                sel = t == arm
                tot[f"n{arm}"] += int(sel.sum())
                tot[f"visit{arm}"] += int(v[sel].sum())
                tot[f"conv{arm}"] += int(c[sel].sum())
            tot["rows"] += m
            done += m
    return tot
