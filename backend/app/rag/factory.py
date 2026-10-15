"""Builds the retrieval components from the configuration.

Changing the embedding model, the reranker or the lexical engine is a configuration
change (or one new class implementing a protocol): nothing else in the code moves.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.rag.analyzer import TextAnalyzer
from app.rag.bm25 import BM25Retriever
from app.rag.dense import DenseRetriever
from app.rag.embeddings import EmbeddingProvider, HashingEmbedder, OpenAICompatibleEmbedder
from app.rag.fusion import ReciprocalRankFusion
from app.rag.pipeline import RAGPipeline, RetrievalConfig
from app.rag.reranker import (
    LexicalOverlapReranker,
    PassthroughReranker,
    RerankApiReranker,
    Reranker,
)
from app.rag.vector_store import QdrantVectorStore, VectorStore, collection_name
from app.repositories.corpus import PostgresChunkStore, PostgresCorpusSource


def build_analyzer(settings: Settings) -> TextAnalyzer:
    return TextAnalyzer(languages=settings.bm25_languages)


def build_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "fake":
        return HashingEmbedder(dimension=settings.embedding_dimension)
    api_key = settings.embedding_api_key
    return OpenAICompatibleEmbedder(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        api_key=api_key.get_secret_value() if api_key else None,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.embedding_timeout_seconds,
        # Only instruction-aware models served locally expect the query prefix.
        query_instruction=(
            settings.embedding_query_instruction if settings.embedding_provider == "vllm" else ""
        ),
    )


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker_provider == "none":
        return PassthroughReranker()
    if settings.reranker_provider == "fake":
        return LexicalOverlapReranker(build_analyzer(settings))
    api_key = settings.reranker_api_key
    return RerankApiReranker(
        base_url=settings.reranker_base_url,
        model=settings.reranker_model,
        api_key=api_key.get_secret_value() if api_key else None,
        timeout_seconds=settings.reranker_timeout_seconds,
    )


def build_vector_store(settings: Settings, embedder: EmbeddingProvider) -> VectorStore:
    api_key = settings.qdrant_api_key
    return QdrantVectorStore(
        url=settings.qdrant_url,
        collection=collection_name(
            settings.qdrant_collection_prefix, embedder.model_name, embedder.dimension
        ),
        api_key=api_key.get_secret_value() if api_key else None,
    )


@dataclass(slots=True)
class RetrievalComponents:
    embedder: EmbeddingProvider
    vector_store: VectorStore
    reranker: Reranker
    lexical: BM25Retriever
    pipeline: RAGPipeline

    def close(self) -> None:
        self.pipeline.close()


def build_retrieval(
    settings: Settings,
    session_factory: sessionmaker[Session],
    *,
    embedder: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    reranker: Reranker | None = None,
) -> RetrievalComponents:
    """Assemble the hybrid pipeline. Components can be injected (tests, evaluation)."""
    # "is None", not truthiness: an empty in-memory store has a length of 0.
    if embedder is None:
        embedder = build_embedder(settings)
    if vector_store is None:
        vector_store = build_vector_store(settings, embedder)
    if reranker is None:
        reranker = build_reranker(settings)
    lexical = BM25Retriever(
        PostgresCorpusSource(session_factory),
        build_analyzer(settings),
        max_cached_users=settings.bm25_cache_max_users,
    )
    pipeline = RAGPipeline(
        lexical=lexical,
        dense=DenseRetriever(embedder, vector_store),
        fusion=ReciprocalRankFusion(k=settings.rrf_k, max_candidates=settings.rrf_max_candidates),
        reranker=reranker,
        chunk_store=PostgresChunkStore(session_factory),
        config=RetrievalConfig(
            bm25_top_k=settings.bm25_top_k,
            dense_top_k=settings.dense_top_k,
            rerank_top_k=settings.rerank_top_k,
            max_query_chars=settings.max_question_chars,
            min_rerank_score=settings.rerank_min_score,
        ),
    )
    return RetrievalComponents(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        lexical=lexical,
        pipeline=pipeline,
    )
