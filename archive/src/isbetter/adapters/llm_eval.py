"""Per-item LLM evaluation results as a canonical table, plus an on-disk response cache.

An LLM eval is a set of items (questions, grouped into passages or tasks) scored for each model.
The two models answer the *same* items, so the scores are paired by item, and items sharing a
source passage are clustered. This adapter keeps that structure:

* :func:`read_item_scores` / :func:`item_scores_table` / :func:`write_item_scores`: per-item JSON
  Lines (``model``, ``item_id``, ``score`` and optionally ``cluster_id``, ``metric``) to the
  long :class:`~isbetter.core.schema.CanonicalTable` with ``pair_id = item_id``.
* :class:`ResponseCache`: append-only JSON Lines cache of model responses keyed by
  ``(model, prompt, item_id, seed)``, so re-running an analysis never pays for a call twice and
  can run fully offline (``offline=True`` raises :class:`CacheMiss` instead of calling).
* :func:`scores_from_cache`: score cached responses into item records.

No model is run here and none is imported: scores come from public per-item results or from
responses obtained through a caller-supplied ``call`` function (an API client the caller owns).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from isbetter.core.dependence import Clustered, Dependence
from isbetter.core.schema import CanonicalTable, SchemaError, build_table

DEFAULT_METRIC = "score"


class CacheMiss(KeyError):
    """A response is not in the cache and the cache is offline (or no ``call`` was given)."""


@dataclass(frozen=True)
class ItemScore:
    """One model's score on one item. ``cluster_id`` groups items (passage, task, source)."""

    model: str
    item_id: str
    score: float
    cluster_id: str | None = None
    metric: str | None = None


@dataclass(frozen=True)
class EvalItem:
    """An item to be answered: its id, the prompt sent to the model and its cluster."""

    item_id: str
    prompt: str
    cluster_id: str | None = None


# -- per-item scores ----------------------------------------------------------------------------
def item_scores_table(
    records: Iterable[ItemScore],
    metric: str = DEFAULT_METRIC,
    dependence: Dependence | None = None,
) -> CanonicalTable:
    """Records to a validated long table (``unit_id = pair_id = item_id``, ``variant = model``).

    Records without their own ``metric`` get the ``metric`` argument. A ``cluster_id`` column is
    written only if every record has one; with ``dependence=Clustered`` a partly missing cluster
    id is an error rather than a silently smaller table. The column is named after
    ``Clustered.cluster_col`` so the runner finds it.
    """
    recs = list(records)
    if not recs:
        raise SchemaError("no item scores")
    have = [r.cluster_id is not None for r in recs]
    if any(have) and not all(have):
        raise SchemaError(
            f"{len(recs) - sum(have)} of {len(recs)} item scores have no cluster_id; give every "
            "item a cluster or none"
        )
    cols: dict[str, list[Any]] = {
        "unit_id": [r.item_id for r in recs],
        "pair_id": [r.item_id for r in recs],
        "variant": [r.model for r in recs],
        "metric": [r.metric or metric for r in recs],
        "value": [r.score for r in recs],
    }
    if all(have):
        name = dependence.cluster_col if isinstance(dependence, Clustered) else "cluster_id"
        cols[name] = [r.cluster_id for r in recs]
    return build_table(cols, dependence)


def _parse_line(line: str, where: str) -> ItemScore:
    try:
        d = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{where}: not valid JSON ({exc.msg})") from exc
    if not isinstance(d, dict):
        raise SchemaError(f"{where}: expected a JSON object")
    missing = [k for k in ("model", "item_id", "score") if k not in d]
    if missing:
        raise SchemaError(f"{where}: missing field(s) {', '.join(missing)}")
    score = d["score"]
    if isinstance(score, bool):
        score = float(score)  # a correct/incorrect flag is a 0/1 score
    if not isinstance(score, int | float):
        raise SchemaError(f"{where}: 'score' must be a number or a boolean, got {score!r}")
    cluster = d.get("cluster_id")
    metric = d.get("metric")
    return ItemScore(
        model=str(d["model"]),
        item_id=str(d["item_id"]),
        score=float(score),
        cluster_id=None if cluster is None else str(cluster),
        metric=None if metric is None else str(metric),
    )


def read_item_scores(
    path: str | Path, dependence: Dependence | None = None, metric: str = DEFAULT_METRIC
) -> CanonicalTable:
    """Read a per-item JSON Lines file (one ``{"model", "item_id", "score", ...}`` per line)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaError(f"cannot read data file {path}: {exc}") from exc
    recs = [
        _parse_line(line, f"{path}:{i}")
        for i, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]
    return item_scores_table(recs, metric, dependence)


def write_item_scores(path: str | Path, records: Iterable[ItemScore]) -> int:
    """Write per-item scores as JSON Lines (fields that are ``None`` are omitted)."""
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            d = {
                "model": r.model,
                "item_id": r.item_id,
                "score": r.score,
                "cluster_id": r.cluster_id,
                "metric": r.metric,
            }
            fh.write(json.dumps({k: v for k, v in d.items() if v is not None}) + "\n")
            n += 1
    return n


# -- response cache -----------------------------------------------------------------------------
def cache_key(model: str, prompt: str, item_id: str, seed: int) -> str:
    """Stable identity of one model call: SHA-256 of the JSON of (model, prompt, item, seed)."""
    payload = json.dumps([model, prompt, str(item_id), int(seed)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Append-only JSON Lines cache of model responses, keyed by (model, prompt, item, seed).

    ``put`` appends one line and never rewrites the file; a key stored twice resolves to the
    latest response. Lines hold a hash of the prompt, not the prompt, so the file stays small
    and does not duplicate a benchmark's text. With ``offline=True`` a miss raises
    :class:`CacheMiss` and ``call`` is never used, which is how a CI run proves it made no calls.
    """

    def __init__(self, path: str | Path, offline: bool = False) -> None:
        self.path = Path(path)
        self.offline = offline
        self.hits = 0
        self.misses = 0
        self._index: dict[str, str] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self._index[rec["key"]] = rec["response"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"corrupt cache line {self.path}:{i}") from exc

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: object) -> bool:
        return key in self._index

    def get(self, model: str, prompt: str, item_id: str, seed: int = 0) -> str | None:
        """The cached response, or ``None``. Counts a hit or a miss."""
        out = self._index.get(cache_key(model, prompt, item_id, seed))
        if out is None:
            self.misses += 1
        else:
            self.hits += 1
        return out

    def put(self, model: str, prompt: str, item_id: str, seed: int, response: str) -> None:
        if not isinstance(response, str):
            raise TypeError("a cached response must be a string")
        key = cache_key(model, prompt, item_id, seed)
        rec = {
            "key": key,
            "model": model,
            "item_id": str(item_id),
            "seed": int(seed),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response": response,
            "logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._index[key] = response

    def get_or_call(
        self,
        model: str,
        prompt: str,
        item_id: str,
        seed: int,
        call: Callable[[], str] | None = None,
    ) -> str:
        """Return the cached response or obtain it with ``call()`` and cache it."""
        hit = self.get(model, prompt, item_id, seed)
        if hit is not None:
            return hit
        if self.offline or call is None:
            why = "the cache is offline" if self.offline else "no `call` was given"
            raise CacheMiss(
                f"no cached response for model={model!r}, item={item_id!r}, seed={seed} and {why}"
            )
        response = call()
        self.put(model, prompt, item_id, seed, response)
        return response


def scores_from_cache(
    cache: ResponseCache,
    models: Sequence[str],
    items: Sequence[EvalItem],
    scorer: Callable[[str, EvalItem, str], float],
    seed: int = 0,
    metric: str | None = None,
) -> list[ItemScore]:
    """Score every (model, item) response in the cache; a missing response raises CacheMiss.

    ``scorer(model, item, response)`` returns the item's score (for example 1.0 if the answer
    matches the key). Nothing is called: an incomplete cache is an error, not a smaller eval.
    """
    if len({it.item_id for it in items}) != len(items):
        raise ValueError("duplicate item_id in items")
    out: list[ItemScore] = []
    for model in models:
        for it in items:
            response = cache.get_or_call(model, it.prompt, it.item_id, seed)
            out.append(
                ItemScore(model, it.item_id, scorer(model, it, response), it.cluster_id, metric)
            )
    return out


def records_from_mapping(
    scores: Mapping[str, Mapping[str, float]],
    clusters: Mapping[str, str] | None = None,
    metric: str | None = None,
) -> list[ItemScore]:
    """``{model: {item_id: score}}`` (public per-item results) to records."""
    return [
        ItemScore(m, str(i), float(s), None if clusters is None else str(clusters[i]), metric)
        for m, per_item in scores.items()
        for i, s in per_item.items()
    ]
