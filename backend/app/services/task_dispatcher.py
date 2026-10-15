"""Publishing background jobs.

The API only needs to *enqueue* jobs: it uses Celery's ``send_task`` with the task
name, so the API process never imports the worker code (nor its heavy dependencies).
Tests use an in-process implementation of the same protocol.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from app.core.context import get_request_id

PROCESS_DOCUMENT_TASK = "documents.process"
CLEANUP_DOCUMENT_TASK = "documents.cleanup"
EXTRACT_INSIGHTS_TASK = "intelligence.extract"
COMPARE_DOCUMENTS_TASK = "comparisons.run"
INGESTION_QUEUE = "ingestion"
# LLM jobs have their own queue: a long comparison never delays the indexing of uploads.
ANALYSIS_QUEUE = "analysis"


class TaskDispatchError(Exception):
    """The job could not be published (broker unavailable)."""


class TaskDispatcher(Protocol):
    def process_document(self, document_id: uuid.UUID) -> str | None:
        """Enqueue the ingestion of a document. Returns the task id when known."""
        ...

    def cleanup_document(self, document_id: uuid.UUID, storage_key: str) -> None:
        """Enqueue the removal of a deleted document's artifacts (file, vectors)."""
        ...

    def extract_insights(self, document_id: uuid.UUID) -> None:
        """Enqueue the extraction of structured information from an indexed document."""
        ...

    def compare_documents(self, comparison_id: uuid.UUID) -> None:
        """Enqueue a document comparison."""
        ...


class CeleryTaskDispatcher:
    def __init__(self) -> None:
        from app.workers.celery_app import celery_app

        self._celery = celery_app

    def _send(self, name: str, args: list[str], queue: str = INGESTION_QUEUE) -> str:
        try:
            result = self._celery.send_task(
                name,
                args=args,
                queue=queue,
                headers={"request_id": get_request_id()},
            )
        except Exception as exc:  # kombu raises various connection errors
            raise TaskDispatchError(f"Cannot publish task {name}: {type(exc).__name__}") from exc
        return str(result.id)

    def process_document(self, document_id: uuid.UUID) -> str | None:
        return self._send(PROCESS_DOCUMENT_TASK, [str(document_id)])

    def cleanup_document(self, document_id: uuid.UUID, storage_key: str) -> None:
        self._send(CLEANUP_DOCUMENT_TASK, [str(document_id), storage_key])

    def extract_insights(self, document_id: uuid.UUID) -> None:
        self._send(EXTRACT_INSIGHTS_TASK, [str(document_id)], ANALYSIS_QUEUE)

    def compare_documents(self, comparison_id: uuid.UUID) -> None:
        self._send(COMPARE_DOCUMENTS_TASK, [str(comparison_id)], ANALYSIS_QUEUE)
