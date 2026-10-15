"""Per-process resources of the worker (created lazily, after the prefork fork).

Creating the database engine or HTTP pools *before* Celery forks its child processes
would share sockets between processes; lazy creation avoids it.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.session import create_sync_db_engine, create_sync_session_factory
from app.document_processing.pipeline import DocumentProcessor
from app.llm.base import LLMProvider
from app.llm.factory import build_llm
from app.rag.embeddings import EmbeddingProvider
from app.rag.factory import RetrievalComponents, build_embedder, build_retrieval, build_vector_store
from app.rag.vector_store import VectorStore
from app.repositories.corpus import PostgresChunkStore
from app.services.comparison_service import ComparisonJobService, build_document_comparator
from app.services.indexing_service import ChunkIndexer
from app.services.ingestion_service import IngestionService, build_document_processor
from app.services.intelligence_service import IntelligenceService, build_insight_extractor
from app.services.storage_service import MinioStorage, ObjectStorage
from app.services.task_dispatcher import CeleryTaskDispatcher, TaskDispatcher


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    settings = get_settings()
    return create_sync_session_factory(create_sync_db_engine(settings, pool_size=5))


@lru_cache(maxsize=1)
def get_storage() -> ObjectStorage:
    return MinioStorage(get_settings())


@lru_cache(maxsize=1)
def get_processor() -> DocumentProcessor:
    return build_document_processor(get_settings())


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    return build_embedder(get_settings())


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return build_vector_store(get_settings(), get_embedder())


@lru_cache(maxsize=1)
def get_indexer() -> ChunkIndexer:
    return ChunkIndexer(get_embedder(), get_vector_store())


def get_ingestion_service() -> IngestionService:
    return IngestionService(get_session_factory(), get_storage(), get_processor(), get_indexer())


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    return build_llm(get_settings())


@lru_cache(maxsize=1)
def get_retrieval() -> RetrievalComponents:
    """Hybrid search used by the comparison job (same pipeline as the API)."""
    return build_retrieval(
        get_settings(),
        get_session_factory(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_dispatcher() -> TaskDispatcher:
    return CeleryTaskDispatcher()


def get_intelligence_service() -> IntelligenceService:
    settings = get_settings()
    return IntelligenceService(get_session_factory(), build_insight_extractor(get_llm(), settings))


def get_comparison_service() -> ComparisonJobService:
    settings = get_settings()
    comparator = build_document_comparator(
        get_llm(),
        get_retrieval().pipeline,
        PostgresChunkStore(get_session_factory()),
        settings,
    )
    return ComparisonJobService(get_session_factory(), comparator)
