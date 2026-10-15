"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

Each retriever contributes ``weight / (k + rank)`` for every document it returns,
where ``rank`` starts at 1. Only ranks matter, so BM25 scores and cosine similarities
(which live on different scales) can be combined without any normalisation:

    score(d) = sum over rankings r containing d of  w_r / (k + rank_r(d))
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.types import SearchHit


@dataclass(frozen=True, slots=True)
class FusedHit:
    chunk_id: uuid.UUID
    score: float
    # Rank (1-based) of the chunk in each input ranking, None when absent.
    ranks: tuple[int | None, ...]


class ReciprocalRankFusion:
    def __init__(self, k: int = 60, max_candidates: int | None = 60) -> None:
        if k <= 0:
            raise ValueError("k must be > 0")
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be > 0")
        self.k = k
        self.max_candidates = max_candidates

    def fuse(
        self,
        rankings: Sequence[Sequence[SearchHit]],
        weights: Sequence[float] | None = None,
    ) -> list[FusedHit]:
        if weights is None:
            weights = [1.0] * len(rankings)
        if len(weights) != len(rankings):
            raise ValueError("weights and rankings must have the same length")

        scores: dict[uuid.UUID, float] = {}
        ranks: dict[uuid.UUID, list[int | None]] = {}
        first_seen: dict[uuid.UUID, int] = {}
        for position, (ranking, weight) in enumerate(zip(rankings, weights, strict=True)):
            seen: set[uuid.UUID] = set()
            rank = 0
            for hit in ranking:
                if hit.chunk_id in seen:  # a retriever must not vote twice for a chunk
                    continue
                seen.add(hit.chunk_id)
                rank += 1
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (self.k + rank)
                ranks.setdefault(hit.chunk_id, [None] * len(rankings))[position] = rank
                first_seen.setdefault(hit.chunk_id, len(first_seen))

        # Ties: best individual rank first, then order of first appearance (deterministic).
        def sort_key(chunk_id: uuid.UUID) -> tuple[float, int, int]:
            best_rank = min(r for r in ranks[chunk_id] if r is not None)
            return (-scores[chunk_id], best_rank, first_seen[chunk_id])

        ordered = sorted(scores, key=sort_key)
        if self.max_candidates is not None:
            ordered = ordered[: self.max_candidates]
        return [FusedHit(c, scores[c], tuple(ranks[c])) for c in ordered]
