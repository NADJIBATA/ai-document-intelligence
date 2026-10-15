"""Document ingestion, executed by the Celery worker (synchronous code).

    uploaded -> processing -> extract -> clean -> chunk
                           -> embed + upsert into the vector store
                           -> persist chunks in PostgreSQL -> indexed
                                \\-> failed (with a user-safe message)

Vectors are written *before* the PostgreSQL transaction that makes the document
visible: ``indexed`` therefore guarantees that both the lexical and the dense indexes
contain the document. Vectors left behind by a failed run are never served (search
results are re-checked against PostgreSQL) and are replaced by the next run.

The job is idempotent: running it twice for the same document replaces the chunks and
the vectors. Transient infrastructure errors are re-raised so that Celery retries the
job; on the last attempt the document is marked as failed instead.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import structlog
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.document_processing.chunking import RecursiveTextChunker
from app.document_processing.cleaning import TextCleaner
from app.document_processing.pipeline import DocumentProcessor
from app.document_processing.registry import default_registry
from app.document_processing.types import Chunk, DocumentProcessingError
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.rag.guardrails import is_suspicious
from app.services.indexing_service import ChunkIndexer, IndexableChunk, IndexingError
from app.services.storage_service import ObjectNotFoundError, ObjectStorage, StorageError

logger = structlog.get_logger(__name__)

# Deterministic chunk ids: the same (document, index) always gives the same id, which
# makes re-indexing idempotent (the Qdrant point id is the chunk id).
CHUNK_ID_NAMESPACE = uuid.UUID("6f1c7a52-6b8e-4b8e-9d0e-2f7a1c3d9e41")
UNEXPECTED_ERROR_MESSAGE = "An unexpected error occurred while processing the document."
STORAGE_ERROR_MESSAGE = "The file could not be read from storage."
INDEXING_ERROR_MESSAGE = (
    "The document could not be indexed because the search services are unavailable. "
    "Please retry later."
)


class TransientIngestionError(Exception):
    """Temporary failure (storage, database...): the job should be retried."""


class IngestionOutcome(StrEnum):
    INDEXED = "indexed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: uuid.UUID
    outcome: IngestionOutcome
    chunk_count: int = 0
    duration_seconds: float = 0.0
    reason: str | None = None


def chunk_id_for(document_id: uuid.UUID, chunk_index: int) -> uuid.UUID:
    return uuid.uuid5(CHUNK_ID_NAMESPACE, f"{document_id}:{chunk_index}")


def build_document_processor(settings: Settings) -> DocumentProcessor:
    return DocumentProcessor(
        registry=default_registry(),
        cleaner=TextCleaner(),
        chunker=RecursiveTextChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            min_size=settings.chunk_min_size,
        ),
    )


class IngestionService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage: ObjectStorage,
        processor: DocumentProcessor,
        indexer: ChunkIndexer,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._processor = processor
        self._indexer = indexer

    def process(self, document_id: uuid.UUID, *, final_attempt: bool = True) -> IngestionResult:
        started = time.perf_counter()
        log = logger.bind(document_id=str(document_id))

        # 1. Claim the document.
        with self._session_factory() as session:
            document = session.get(Document, document_id, with_for_update=True)
            if document is None:
                log.info("ingestion_skipped", reason="document_deleted")
                return IngestionResult(document_id, IngestionOutcome.SKIPPED, reason="deleted")
            if document.status == DocumentStatus.INDEXED:
                log.info("ingestion_skipped", reason="already_indexed")
                return IngestionResult(document_id, IngestionOutcome.SKIPPED, reason="indexed")
            document.status = DocumentStatus.PROCESSING
            document.error_message = None
            document.processing_started_at = datetime.now(UTC)
            storage_key, content_type, filename = (
                document.storage_key,
                document.content_type,
                document.filename,
            )
            owner_id = document.owner_id
            session.commit()
        log.info("ingestion_started")

        # 2. Read, extract, clean, chunk.
        try:
            data = self._storage.get_object(storage_key)
            processed = self._processor.process(data, content_type, filename)
        except ObjectNotFoundError:
            return self._fail(document_id, STORAGE_ERROR_MESSAGE, started)
        except StorageError as exc:
            if not final_attempt:
                log.warning("ingestion_transient_error", error=str(exc))
                raise TransientIngestionError(str(exc)) from exc
            return self._fail(document_id, STORAGE_ERROR_MESSAGE, started)
        except DocumentProcessingError as exc:
            return self._fail(document_id, exc.user_message, started)
        except Exception:
            log.exception("ingestion_unexpected_error")
            return self._fail(document_id, UNEXPECTED_ERROR_MESSAGE, started)

        # 3. Embeddings + vector store.
        try:
            self._indexer.index_document(
                document_id=document_id,
                owner_id=owner_id,
                filename=filename,
                chunks=[
                    IndexableChunk(
                        chunk_id=chunk_id_for(document_id, chunk.index),
                        chunk_index=chunk.index,
                        text=chunk.text,
                        page=chunk.page,
                        section=chunk.section,
                    )
                    for chunk in processed.chunks
                ],
            )
        except IndexingError as exc:
            if not final_attempt:
                log.warning("ingestion_transient_error", error=str(exc))
                raise TransientIngestionError(str(exc)) from exc
            log.error("ingestion_indexing_failed", error=str(exc))
            return self._fail(document_id, INDEXING_ERROR_MESSAGE, started)
        except Exception:
            log.exception("ingestion_unexpected_error")
            return self._fail(document_id, UNEXPECTED_ERROR_MESSAGE, started)

        # 4. Persist chunks and mark the document as indexed (single transaction).
        try:
            chunk_count = self._persist(
                document_id, owner_id, processed.chunks, processed.extracted.page_count
            )
        except IntegrityError:
            # The document was deleted while it was being processed.
            log.info("ingestion_skipped", reason="document_deleted_during_processing")
            self._delete_vectors_quietly(document_id)
            return IngestionResult(document_id, IngestionOutcome.SKIPPED, reason="deleted")
        except Exception as exc:
            if not final_attempt:
                log.warning("ingestion_transient_error", error=type(exc).__name__)
                raise TransientIngestionError(type(exc).__name__) from exc
            log.exception("ingestion_persist_failed")
            return self._fail(document_id, UNEXPECTED_ERROR_MESSAGE, started)

        duration = time.perf_counter() - started
        log.info("ingestion_completed", chunk_count=chunk_count, duration_s=round(duration, 3))
        return IngestionResult(
            document_id,
            IngestionOutcome.INDEXED,
            chunk_count=chunk_count,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------ helpers
    def _persist(
        self,
        document_id: uuid.UUID,
        owner_id: uuid.UUID,
        chunks: list[Chunk],
        page_count: int | None,
    ) -> int:
        now = datetime.now(UTC)
        # Passages that look like instructions addressed to an AI are flagged: they stay
        # searchable but are marked as such in the prompt (see app.rag.guardrails).
        flagged = {chunk.index for chunk in chunks if is_suspicious(chunk.text)}
        if flagged:
            logger.warning(
                "prompt_injection_suspected", document_id=str(document_id), chunks=len(flagged)
            )
        with self._session_factory() as session:
            session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
            session.add_all(
                DocumentChunk(
                    id=chunk_id_for(document_id, chunk.index),
                    document_id=document_id,
                    owner_id=owner_id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    page=chunk.page,
                    section=chunk.section[:512] if chunk.section else None,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    flagged_injection=chunk.index in flagged,
                    meta={},
                )
                for chunk in chunks
            )
            document = session.get(Document, document_id, with_for_update=True)
            if document is None:
                raise IntegrityError("document deleted", params=None, orig=Exception())
            document.status = DocumentStatus.INDEXED
            document.chunk_count = len(chunks)
            document.page_count = page_count
            document.embedding_model = self._indexer.model_name[:255]
            document.indexed_at = now
            document.processing_completed_at = now
            session.commit()
        return len(chunks)

    def _delete_vectors_quietly(self, document_id: uuid.UUID) -> None:
        try:
            self._indexer.delete_document(document_id)
        except IndexingError as exc:
            logger.warning("vector_cleanup_failed", document_id=str(document_id), error=str(exc))

    def _fail(self, document_id: uuid.UUID, message: str, started: float) -> IngestionResult:
        duration = time.perf_counter() - started
        with self._session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                return IngestionResult(document_id, IngestionOutcome.SKIPPED, reason="deleted")
            document.status = DocumentStatus.FAILED
            document.error_message = message
            document.processing_completed_at = datetime.now(UTC)
            session.commit()
        logger.warning("ingestion_failed", document_id=str(document_id), reason=message)
        return IngestionResult(
            document_id, IngestionOutcome.FAILED, duration_seconds=duration, reason=message
        )
