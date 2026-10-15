"""The four retrieval systems compared by the evaluation.

All of them run through the production ``RAGPipeline``; only stages are switched off,
so the evaluation measures exactly the code that serves the users:

* ``bm25``          - lexical search only
* ``dense``         - embeddings + vector search only
* ``hybrid_rrf``    - both, merged with Reciprocal Rank Fusion
* ``hybrid_rerank`` - both, RRF, then the cross-encoder reranker (production setting)

Two ways to build the pipeline:

* offline - a local corpus indexed in memory (in-memory BM25 and vector store), with
  the embedding model and reranker configured in .env, or deterministic fakes;
* live    - the PostgreSQL / Qdrant data of an existing user of a deployment.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.config import Settings
from app.rag.bm25 import BM25Retriever
from app.rag.dense import DenseRetriever
from app.rag.embeddings import EmbeddingProvider
from app.rag.factory import build_analyzer, build_embedder, build_reranker
from app.rag.fusion import ReciprocalRankFusion
from app.rag.pipeline import RAGPipeline, RetrievalConfig
from app.rag.reranker import Reranker
from app.rag.types import CorpusEntry, RetrievedChunk
from app.rag.vector_store import InMemoryVectorStore, VectorPoint
from app.services.indexing_service import IndexableChunk, embedding_text
from evaluation.corpus import Corpus

SYSTEMS: tuple[str, ...] = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")

SWITCHES: dict[str, dict[str, bool]] = {
    "bm25": {"use_lexical": True, "use_dense": False, "use_reranker": False},
    "dense": {"use_lexical": False, "use_dense": True, "use_reranker": False},
    "hybrid_rrf": {"use_lexical": True, "use_dense": True, "use_reranker": False},
    "hybrid_rerank": {"use_lexical": True, "use_dense": True, "use_reranker": True},
}

# The offline corpus belongs to a single pseudo-user.
OFFLINE_USER = uuid.UUID(int=0)


@dataclass(frozen=True, slots=True)
class Retrieval:
    chunk_ids: list[str]
    latency_ms: float
    degraded: list[str] = field(default_factory=list)


def run_system(
    pipeline: RAGPipeline, user_id: uuid.UUID, system: str, question: str, top_k: int
) -> Retrieval:
    if system not in SWITCHES:
        raise ValueError(f"unknown system '{system}' (expected one of {', '.join(SYSTEMS)})")
    started = time.perf_counter()
    switches = SWITCHES[system]
    result = pipeline.retrieve(
        question,
        user_id,
        top_k=top_k,
        use_lexical=switches["use_lexical"],
        use_dense=switches["use_dense"],
        use_reranker=switches["use_reranker"],
    )
    latency = (time.perf_counter() - started) * 1000
    return Retrieval(
        chunk_ids=[str(chunk.chunk_id) for chunk in result.chunks],
        latency_ms=latency,
        degraded=list(result.degraded),
    )


def retrieval_config(settings: Settings, top_k: int) -> RetrievalConfig:
    return RetrievalConfig(
        bm25_top_k=max(settings.bm25_top_k, top_k),
        dense_top_k=max(settings.dense_top_k, top_k),
        rerank_top_k=top_k,
        max_query_chars=settings.max_question_chars,
        # The evaluation ranks, it does not filter: a threshold would hide results.
        min_rerank_score=0.0,
    )


# ------------------------------------------------------------------- offline mode
class _MemoryCorpus:
    """CorpusSource + ChunkStore over a local corpus (single pseudo-user)."""

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus

    def signature(self, user_id: uuid.UUID) -> str:
        return "static"

    def load(self, user_id: uuid.UUID) -> list[CorpusEntry]:
        if user_id != OFFLINE_USER:
            return []
        return [
            CorpusEntry(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                # Same text as the production index: section title + passage.
                text=f"{chunk.section}\n{chunk.text}" if chunk.section else chunk.text,
            )
            for chunk in self._corpus.chunks
        ]

    def get_many(
        self, user_id: uuid.UUID, chunk_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, RetrievedChunk]:
        found: dict[uuid.UUID, RetrievedChunk] = {}
        if user_id != OFFLINE_USER:
            return found
        for chunk_id in chunk_ids:
            chunk = self._corpus.get(str(chunk_id))
            if chunk is not None:
                # A new object every time: the pipeline writes the scores into it.
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


def build_offline_pipeline(
    corpus: Corpus,
    settings: Settings,
    *,
    top_k: int,
    embedder: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
) -> RAGPipeline:
    """Index ``corpus`` in memory and return the pipeline (embeds every passage once)."""
    embedder = embedder if embedder is not None else build_embedder(settings)
    reranker = reranker if reranker is not None else build_reranker(settings)
    vector_store = InMemoryVectorStore()
    vector_store.ensure_collection(embedder.dimension)
    texts = [
        embedding_text(
            chunk.filename,
            IndexableChunk(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
            ),
        )
        for chunk in corpus.chunks
    ]
    vectors = embedder.embed_documents(texts)
    vector_store.upsert(
        [
            VectorPoint(chunk.chunk_id, vector, OFFLINE_USER, chunk.document_id, {})
            for chunk, vector in zip(corpus.chunks, vectors, strict=True)
        ]
    )
    memory = _MemoryCorpus(corpus)
    return RAGPipeline(
        lexical=BM25Retriever(memory, build_analyzer(settings)),
        dense=DenseRetriever(embedder, vector_store),
        fusion=ReciprocalRankFusion(k=settings.rrf_k, max_candidates=settings.rrf_max_candidates),
        reranker=reranker,
        chunk_store=memory,
        config=retrieval_config(settings, top_k),
    )
