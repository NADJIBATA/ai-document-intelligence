"""Document tasks: ingestion and cleanup of deleted documents."""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from celery import Task

from app.core.config import get_settings
from app.rag.vector_store import VectorStoreError
from app.services.ingestion_service import IngestionOutcome, TransientIngestionError
from app.services.intelligence_service import schedule_insights
from app.services.storage_service import StorageError
from app.services.task_dispatcher import CLEANUP_DOCUMENT_TASK, PROCESS_DOCUMENT_TASK
from app.workers.celery_app import celery_app
from app.workers.resources import (
    get_dispatcher,
    get_ingestion_service,
    get_intelligence_service,
    get_storage,
    get_vector_store,
)

logger = structlog.get_logger(__name__)

MAX_RETRIES = 3


def _retry_delay(retries: int) -> int:
    """Exponential backoff: 10 s, 20 s, 40 s... capped at 2 minutes."""
    return int(min(10 * 2**retries, 120))


@celery_app.task(name=PROCESS_DOCUMENT_TASK, bind=True, max_retries=MAX_RETRIES)
def process_document(self: Task, document_id: str) -> dict[str, Any]:
    structlog.contextvars.bind_contextvars(document_id=document_id)
    final_attempt = self.request.retries >= MAX_RETRIES
    try:
        result = get_ingestion_service().process(
            uuid.UUID(document_id), final_attempt=final_attempt
        )
    except TransientIngestionError as exc:
        raise self.retry(exc=exc, countdown=_retry_delay(self.request.retries)) from exc
    if result.outcome == IngestionOutcome.INDEXED and get_settings().intelligence_enabled:
        # Document Intelligence runs as a separate job: the document is searchable
        # right away, and a slow or failing model never affects the indexing.
        schedule_insights(uuid.UUID(document_id), get_intelligence_service(), get_dispatcher())
    return {
        "document_id": document_id,
        "outcome": result.outcome.value,
        "chunk_count": result.chunk_count,
        "duration_seconds": round(result.duration_seconds, 3),
    }


@celery_app.task(name=CLEANUP_DOCUMENT_TASK, bind=True, max_retries=5)
def cleanup_document(self: Task, document_id: str, storage_key: str) -> dict[str, str]:
    """Remove the artifacts of a document already deleted from PostgreSQL.

    Removes the original file from object storage and the vectors from Qdrant (both
    operations are idempotent). Leftovers are harmless meanwhile: search results are
    always re-checked against PostgreSQL, so orphan data can never be served.
    """
    structlog.contextvars.bind_contextvars(document_id=document_id)
    try:
        get_storage().delete_object(storage_key)
        get_vector_store().delete_document(uuid.UUID(document_id))
    except (StorageError, VectorStoreError) as exc:
        raise self.retry(exc=exc, countdown=_retry_delay(self.request.retries)) from exc
    logger.info("document_artifacts_cleaned")
    return {"document_id": document_id, "status": "cleaned"}
