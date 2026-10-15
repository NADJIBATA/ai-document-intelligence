"""The retrieval pipeline with in-memory collaborators (no database, no model server)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from app.rag.bm25 import BM25Retriever
from app.rag.dense import DenseRetriever
from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.fusion import ReciprocalRankFusion
from app.rag.pipeline import RAGPipeline, RetrievalConfig, preprocess_query
from app.rag.reranker import LexicalOverlapReranker, PassthroughReranker, RerankerError
from app.rag.types import CorpusEntry, RetrievedChunk, SearchHit
from app.rag.vector_store import InMemoryVectorStore, VectorPoint, VectorStoreError

ALICE = uuid.UUID(int=1)
BOB = uuid.UUID(int=2)
DOC_ALICE = uuid.UUID(int=101)
DOC_ALICE_2 = uuid.UUID(int=102)
DOC_BOB = uuid.UUID(int=201)


class World:
    """A tiny multi-user corpus exposed through the protocols the pipeline expects."""

    def __init__(self) -> None:
        self.embedder = HashingEmbedder(dimension=128)
        self.vector_store = InMemoryVectorStore()
        self.chunks: dict[uuid.UUID, tuple[uuid.UUID, RetrievedChunk]] = {}

    def add(self, number: int, owner: uuid.UUID, document_id: uuid.UUID, text: str) -> uuid.UUID:
        chunk_id = uuid.UUID(int=1000 + number)
        self.chunks[chunk_id] = (
            owner,
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                filename=f"{document_id.int}.pdf",
                text=text,
                page=1,
                section=None,
                chunk_index=number,
            ),
        )
        self.vector_store.upsert(
            [
                VectorPoint(
                    chunk_id, self.embedder.embed_documents([text])[0], owner, document_id, {}
                )
            ]
        )
        return chunk_id

    # CorpusSource
    def signature(self, user_id: uuid.UUID) -> str:
        return str(sorted(str(cid) for cid, (owner, _) in self.chunks.items() if owner == user_id))

    def load(self, user_id: uuid.UUID) -> list[CorpusEntry]:
        return [
            CorpusEntry(chunk_id, chunk.document_id, chunk.text)
            for chunk_id, (owner, chunk) in self.chunks.items()
            if owner == user_id
        ]

    # ChunkStore
    def get_many(
        self, user_id: uuid.UUID, chunk_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, RetrievedChunk]:
        found: dict[uuid.UUID, RetrievedChunk] = {}
        for chunk_id in chunk_ids:
            owner, chunk = self.chunks.get(chunk_id, (None, None))
            if chunk is not None and owner == user_id:
                found[chunk_id] = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    text=chunk.text,
                    page=chunk.page,
                    section=chunk.section,
                    chunk_index=chunk.chunk_index,
                )
        return found


@pytest.fixture
def world() -> World:
    world = World()
    world.add(1, ALICE, DOC_ALICE, "The contract may be terminated with thirty days notice.")
    world.add(2, ALICE, DOC_ALICE, "Invoices are payable within sixty days of receipt.")
    world.add(3, ALICE, DOC_ALICE_2, "Late payment penalties are three times the legal rate.")
    world.add(4, ALICE, DOC_ALICE_2, "The supplier keeps all customer data confidential.")
    world.add(5, BOB, DOC_BOB, "Bob's contract may be terminated with ninety days notice.")
    world.add(6, BOB, DOC_BOB, "Bob's salary is confidential.")
    return world


def build(world: World, **overrides: object) -> RAGPipeline:
    options: dict[str, object] = {
        "lexical": BM25Retriever(world),
        "dense": DenseRetriever(world.embedder, world.vector_store),
        "fusion": ReciprocalRankFusion(k=60, max_candidates=60),
        "reranker": LexicalOverlapReranker(),
        "chunk_store": world,
        "config": RetrievalConfig(bm25_top_k=30, dense_top_k=30, rerank_top_k=5),
    }
    options.update(overrides)
    return RAGPipeline(**options)  # type: ignore[arg-type]


def numbers(chunks: list[RetrievedChunk]) -> list[int]:
    return [chunk.chunk_id.int - 1000 for chunk in chunks]


# ------------------------------------------------------------------ preprocessing
def test_preprocess_query() -> None:
    assert preprocess_query("  What is\tthe\nnotice   period?\x00 ") == "What is the notice period?"
    assert preprocess_query("a" * 50, max_chars=10) == "a" * 10
    assert preprocess_query(" \n\t ") == ""


# ----------------------------------------------------------------------- pipeline
def test_hybrid_retrieval_returns_the_best_passage_with_stage_details(world: World) -> None:
    result = build(world).retrieve("contract terminated notice", ALICE)

    assert numbers(result.chunks)[0] == 1
    best = result.chunks[0]
    assert best.bm25_rank == 1
    assert best.dense_rank == 1
    assert best.rrf_score == pytest.approx(2 / 61)
    assert best.rerank_score == pytest.approx(1.0)
    assert best.filename == "101.pdf"
    assert result.degraded == []
    assert result.candidate_count >= 1
    assert {"bm25", "dense", "fusion", "hydration", "rerank", "total"} <= set(result.timings_ms)


def test_user_isolation_on_every_path(world: World) -> None:
    pipeline = build(world)
    for options in (
        {},
        {"use_dense": False},  # BM25 only
        {"use_lexical": False},  # dense only
        {"use_reranker": False},
    ):
        alice = pipeline.retrieve("contract terminated notice confidential", ALICE, **options)
        bob = pipeline.retrieve("contract terminated notice confidential", BOB, **options)
        assert alice.chunks and bob.chunks
        assert set(numbers(alice.chunks)) <= {1, 2, 3, 4}, options
        assert set(numbers(bob.chunks)) <= {5, 6}, options

    stranger = pipeline.retrieve("contract notice", uuid.UUID(int=999))
    assert stranger.chunks == []


def test_candidates_not_owned_by_the_user_are_dropped_at_hydration(world: World) -> None:
    """Defence in depth: even a retriever returning foreign ids cannot leak a passage."""

    class LeakyRetriever:
        def search(
            self,
            query: str,
            user_id: uuid.UUID,
            *,
            limit: int,
            document_ids: Sequence[uuid.UUID] | None = None,
        ) -> list[SearchHit]:
            return [SearchHit(uuid.UUID(int=1005), 9.9), SearchHit(uuid.UUID(int=1001), 1.0)]

    result = build(world, lexical=LeakyRetriever(), dense=LeakyRetriever()).retrieve(
        "contract notice", ALICE
    )
    assert numbers(result.chunks) == [1]
    assert result.candidate_count == 2


def test_document_filter(world: World) -> None:
    pipeline = build(world)
    result = pipeline.retrieve("payment days", ALICE, document_ids=[DOC_ALICE_2])
    assert result.chunks
    assert {chunk.document_id for chunk in result.chunks} == {DOC_ALICE_2}
    # Another user's document id gives nothing, on both retrieval paths.
    assert pipeline.retrieve("contract notice", ALICE, document_ids=[DOC_BOB]).chunks == []
    assert pipeline.retrieve("contract notice", ALICE, document_ids=[]).chunks == []


def test_top_k(world: World) -> None:
    pipeline = build(world)
    assert len(pipeline.retrieve("contract payment days data", ALICE, top_k=2).chunks) == 2
    assert len(pipeline.retrieve("contract payment days data", ALICE).chunks) <= 5


def test_at_most_max_candidates_reach_the_reranker() -> None:
    world = World()
    for number in range(1, 81):
        world.add(number, ALICE, DOC_ALICE, f"clause number {number} about the contract")
    seen: list[int] = []

    class CountingReranker(PassthroughReranker):
        def rerank(
            self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
        ) -> list[RetrievedChunk]:
            seen.append(len(chunks))
            return super().rerank(query, chunks, top_k)

    pipeline = build(
        world,
        reranker=CountingReranker(),
        fusion=ReciprocalRankFusion(k=60, max_candidates=60),
    )
    result = pipeline.retrieve("contract clause", ALICE)
    assert result.candidate_count <= 60
    assert seen == [result.candidate_count]
    assert len(result.chunks) == 5


def test_empty_query_returns_nothing(world: World) -> None:
    result = build(world).retrieve("   \n ", ALICE)
    assert result.chunks == []
    assert result.query == ""


def test_no_match(world: World) -> None:
    result = build(world).retrieve("zzzunknownzzz", ALICE, use_dense=False)
    assert result.chunks == []
    assert result.candidate_count == 0


# -------------------------------------------------------------------- degradation
class BrokenEmbedder(HashingEmbedder):
    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingError("embeddings: ConnectError")


class BrokenVectorStore(InMemoryVectorStore):
    def search(self, *args: object, **kwargs: object) -> list[SearchHit]:
        raise VectorStoreError("qdrant: ConnectError")


class BrokenReranker:
    model_name = "broken"

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        raise RerankerError("reranker: HTTP 503")


def test_embedding_server_down_falls_back_to_lexical_search(world: World) -> None:
    pipeline = build(world, dense=DenseRetriever(BrokenEmbedder(128), world.vector_store))
    result = pipeline.retrieve("contract terminated notice", ALICE)
    assert numbers(result.chunks)[0] == 1
    assert result.degraded == ["dense"]
    assert all(chunk.dense_rank is None for chunk in result.chunks)


def test_vector_store_down_falls_back_to_lexical_search(world: World) -> None:
    pipeline = build(world, dense=DenseRetriever(world.embedder, BrokenVectorStore()))
    result = pipeline.retrieve("contract terminated notice", ALICE)
    assert numbers(result.chunks)[0] == 1
    assert result.degraded == ["dense"]


def test_reranker_down_keeps_the_fusion_order(world: World) -> None:
    result = build(world, reranker=BrokenReranker()).retrieve(
        "contract terminated notice", ALICE, top_k=3
    )
    assert numbers(result.chunks)[0] == 1
    assert len(result.chunks) <= 3
    assert result.degraded == ["rerank"]
    assert all(chunk.rerank_score is None for chunk in result.chunks)
    scores = [chunk.rrf_score for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_everything_down_still_answers_from_bm25(world: World) -> None:
    pipeline = build(
        world,
        dense=DenseRetriever(BrokenEmbedder(128), world.vector_store),
        reranker=BrokenReranker(),
    )
    result = pipeline.retrieve("confidential customer data", ALICE)
    assert numbers(result.chunks)[0] == 4
    assert result.degraded == ["dense", "rerank"]


# ------------------------------------------------------------------------ options
def test_evaluation_switches(world: World) -> None:
    pipeline = build(world)
    lexical_only = pipeline.retrieve("termination notice", ALICE, use_dense=False)
    assert "dense" not in lexical_only.timings_ms
    assert all(chunk.dense_rank is None for chunk in lexical_only.chunks)

    dense_only = pipeline.retrieve("termination notice", ALICE, use_lexical=False)
    assert "bm25" not in dense_only.timings_ms
    assert all(chunk.bm25_rank is None for chunk in dense_only.chunks)

    no_rerank = pipeline.retrieve("termination notice", ALICE, use_reranker=False)
    assert "rerank" not in no_rerank.timings_ms
    assert all(chunk.rerank_score is None for chunk in no_rerank.chunks)


def test_min_rerank_score_discards_weak_passages(world: World) -> None:
    config = RetrievalConfig(rerank_top_k=5, min_rerank_score=0.6)
    result = build(world, config=config).retrieve("contract terminated notice", ALICE)
    assert numbers(result.chunks) == [1]  # the only passage containing the three terms
    assert result.candidate_count > 1


def test_long_queries_are_truncated(world: World) -> None:
    config = RetrievalConfig(max_query_chars=20)
    result = build(world, config=config).retrieve("contract " * 50, ALICE)
    assert len(result.query) <= 20
