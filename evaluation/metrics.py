"""Retrieval metrics. ``retrieved`` is the ranked list returned by a system (best first),
``relevant`` the set of chunks that answer the question.

* Recall@k    = |relevant ∩ top k| / |relevant|
* Precision@k = |relevant ∩ top k| / k
* MRR@k       = mean over questions of 1 / rank of the first relevant chunk in the
                top k (0 when there is none) - ``reciprocal_rank_at_k`` is the
                per-question term.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence


def _check(k: int, relevant: Collection[str]) -> None:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant:
        raise ValueError("a question needs at least one relevant chunk")


def _hits(retrieved: Sequence[str], relevant: Collection[str], k: int) -> int:
    # A system returning the same chunk twice must not be counted twice.
    return len(set(retrieved[:k]) & set(relevant))


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(k, relevant)
    return _hits(retrieved, relevant, k) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(k, relevant)
    return _hits(retrieved, relevant, k) / k


def reciprocal_rank_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(k, relevant)
    wanted = set(relevant)
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in wanted:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
