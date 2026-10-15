"""In-process test doubles for external services (object storage, job queue)."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from typing import BinaryIO

from app.rag.vector_store import VectorStore
from app.services.comparison_service import ComparisonJobService
from app.services.ingestion_service import IngestionOutcome, IngestionService
from app.services.intelligence_service import IntelligenceService, schedule_insights
from app.services.storage_service import STREAM_CHUNK_SIZE, ObjectNotFoundError, StorageError
from app.services.task_dispatcher import TaskDispatchError


class InMemoryStorage:
    """Thread-safe dictionary implementing the ``ObjectStorage`` protocol."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.available = True
        self._lock = threading.Lock()

    def _check(self) -> None:
        if not self.available:
            raise StorageError("storage down")

    def ensure_bucket(self) -> None:
        self._check()

    def ping(self) -> None:
        self._check()

    def put_object(self, key: str, data: BinaryIO, length: int, content_type: str) -> None:
        self._check()
        payload = data.read()
        assert len(payload) == length
        with self._lock:
            self.objects[key] = (payload, content_type)

    def get_object(self, key: str) -> bytes:
        self._check()
        with self._lock:
            if key not in self.objects:
                raise ObjectNotFoundError(key)
            return self.objects[key][0]

    def iter_object(self, key: str, chunk_size: int = STREAM_CHUNK_SIZE) -> Iterator[bytes]:
        data = self.get_object(key)
        for start in range(0, len(data), chunk_size):
            yield data[start : start + chunk_size]

    def delete_object(self, key: str) -> None:
        self._check()
        with self._lock:
            self.objects.pop(key, None)


class RecordingDispatcher:
    """Records the jobs that would have been published to Celery."""

    def __init__(self) -> None:
        self.processed: list[uuid.UUID] = []
        self.cleaned: list[tuple[uuid.UUID, str]] = []
        self.insights: list[uuid.UUID] = []
        self.comparisons: list[uuid.UUID] = []
        self.fail = False

    def process_document(self, document_id: uuid.UUID) -> str | None:
        if self.fail:
            raise TaskDispatchError("broker down")
        self.processed.append(document_id)
        return f"task-{document_id}"

    def cleanup_document(self, document_id: uuid.UUID, storage_key: str) -> None:
        if self.fail:
            raise TaskDispatchError("broker down")
        self.cleaned.append((document_id, storage_key))

    def extract_insights(self, document_id: uuid.UUID) -> None:
        if self.fail:
            raise TaskDispatchError("broker down")
        self.insights.append(document_id)

    def compare_documents(self, comparison_id: uuid.UUID) -> None:
        if self.fail:
            raise TaskDispatchError("broker down")
        self.comparisons.append(comparison_id)


class InlineDispatcher(RecordingDispatcher):
    """Runs the jobs immediately, like ``CELERY_TASK_ALWAYS_EAGER`` would."""

    def __init__(
        self,
        ingestion: IngestionService,
        storage: InMemoryStorage,
        vector_store: VectorStore,
        intelligence: IntelligenceService,
        comparisons: ComparisonJobService,
    ) -> None:
        super().__init__()
        self._ingestion = ingestion
        self._storage = storage
        self._vector_store = vector_store
        self._intelligence = intelligence
        self._comparison_jobs = comparisons
        # Mirrors INTELLIGENCE_ENABLED in the worker. Off by default so that tests about
        # something else do not pay for (nor see) an extraction after every upload.
        self.auto_insights = False

    def process_document(self, document_id: uuid.UUID) -> str | None:
        task_id = super().process_document(document_id)
        result = self._ingestion.process(document_id)
        if self.auto_insights and result.outcome == IngestionOutcome.INDEXED:
            schedule_insights(document_id, self._intelligence, self)
        return task_id

    def extract_insights(self, document_id: uuid.UUID) -> None:
        super().extract_insights(document_id)
        self._intelligence.extract(document_id)

    def compare_documents(self, comparison_id: uuid.UUID) -> None:
        super().compare_documents(comparison_id)
        self._comparison_jobs.run(comparison_id)

    def cleanup_document(self, document_id: uuid.UUID, storage_key: str) -> None:
        super().cleanup_document(document_id, storage_key)
        self._storage.delete_object(storage_key)
        self._vector_store.delete_document(document_id)
