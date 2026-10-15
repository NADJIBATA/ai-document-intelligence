from __future__ import annotations

import uuid

import pytest

from app.rag.fusion import ReciprocalRankFusion
from app.rag.types import SearchHit

A, B, C, D = (uuid.UUID(int=i) for i in range(1, 5))


def hits(*chunk_ids: uuid.UUID) -> list[SearchHit]:
    # Scores are deliberately meaningless: RRF must only look at ranks.
    return [SearchHit(chunk_id, score=1000.0 / (i + 1)) for i, chunk_id in enumerate(chunk_ids)]


def test_scores_follow_the_rrf_formula() -> None:
    fused = ReciprocalRankFusion(k=60).fuse([hits(A, B), hits(B, C)])
    by_id = {hit.chunk_id: hit for hit in fused}
    assert by_id[A].score == pytest.approx(1 / 61)
    assert by_id[B].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[C].score == pytest.approx(1 / 62)
    assert by_id[B].ranks == (2, 1)
    assert by_id[A].ranks == (1, None)


def test_chunks_found_by_both_retrievers_come_first() -> None:
    fused = ReciprocalRankFusion().fuse([hits(A, B, C), hits(D, B)])
    assert [hit.chunk_id for hit in fused] == [B, A, D, C]


def test_scores_of_the_retrievers_are_ignored() -> None:
    lexical = [SearchHit(A, 42.0), SearchHit(B, 0.001)]
    dense = [SearchHit(B, 0.99), SearchHit(A, 0.98)]
    fused = ReciprocalRankFusion().fuse([lexical, dense])
    assert fused[0].score == pytest.approx(fused[1].score)


def test_ties_are_deterministic() -> None:
    # A and D are both ranked first by one retriever: first appearance wins.
    fused = ReciprocalRankFusion().fuse([hits(A), hits(D)])
    assert [hit.chunk_id for hit in fused] == [A, D]


def test_max_candidates() -> None:
    many = [uuid.UUID(int=i) for i in range(1, 41)]
    other = [uuid.UUID(int=i) for i in range(100, 140)]
    fused = ReciprocalRankFusion(max_candidates=60).fuse([hits(*many), hits(*other)])
    assert len(fused) == 60
    assert len(ReciprocalRankFusion(max_candidates=None).fuse([hits(*many), hits(*other)])) == 80


def test_duplicates_inside_a_ranking_count_once() -> None:
    fused = ReciprocalRankFusion(k=60).fuse([hits(A, A, B)])
    by_id = {hit.chunk_id: hit for hit in fused}
    assert by_id[A].score == pytest.approx(1 / 61)
    assert by_id[B].score == pytest.approx(1 / 62)  # rank 2, not 3


def test_weights() -> None:
    fused = ReciprocalRankFusion(k=60).fuse([hits(A), hits(B)], weights=[1.0, 2.0])
    assert [hit.chunk_id for hit in fused] == [B, A]
    with pytest.raises(ValueError, match="same length"):
        ReciprocalRankFusion().fuse([hits(A)], weights=[1.0, 2.0])


def test_empty_inputs() -> None:
    assert ReciprocalRankFusion().fuse([]) == []
    assert ReciprocalRankFusion().fuse([[], []]) == []
    assert [hit.chunk_id for hit in ReciprocalRankFusion().fuse([[], hits(A)])] == [A]


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="k must be"):
        ReciprocalRankFusion(k=0)
    with pytest.raises(ValueError, match="max_candidates"):
        ReciprocalRankFusion(max_candidates=0)
