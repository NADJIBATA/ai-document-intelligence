"""Celery wiring: task names, dispatcher and task bodies (executed eagerly, no broker)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytest.importorskip("celery")

from app.rag.embeddings import HashingEmbedder
from app.rag.vector_store import InMemoryVectorStore, VectorPoint
from app.services.comparison_service import ComparisonJobService
from app.services.ingestion_service import IngestionService
from app.services.intelligence_service import IntelligenceService
from app.services.task_dispatcher import (
    ANALYSIS_QUEUE,
    CLEANUP_DOCUMENT_TASK,
    COMPARE_DOCUMENTS_TASK,
    EXTRACT_INSIGHTS_TASK,
    INGESTION_QUEUE,
    PROCESS_DOCUMENT_TASK,
    CeleryTaskDispatcher,
)
from app.workers import resources
from app.workers.celery_app import celery_app
from app.workers.tasks import comparisons as comparison_tasks
from app.workers.tasks import documents as document_tasks
from app.workers.tasks import intelligence as intelligence_tasks
from tests.fakes import InMemoryStorage, RecordingDispatcher
from tests.fixtures.builders import simple_pdf
from tests.integration.conftest import signup, upload


class _WorkerSettings:
    """Stands in for ``get_settings()`` inside the task module."""

    def __init__(self, *, intelligence_enabled: bool) -> None:
        self.intelligence_enabled = intelligence_enabled


def test_tasks_are_registered_under_stable_names() -> None:
    for name in (
        PROCESS_DOCUMENT_TASK,
        CLEANUP_DOCUMENT_TASK,
        EXTRACT_INSIGHTS_TASK,
        COMPARE_DOCUMENTS_TASK,
    ):
        assert name in celery_app.tasks
    routes = celery_app.conf.task_routes
    assert routes["documents.*"] == {"queue": INGESTION_QUEUE}
    assert routes["intelligence.*"] == {"queue": ANALYSIS_QUEUE}
    assert routes["comparisons.*"] == {"queue": ANALYSIS_QUEUE}
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_dispatcher_publishes_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    class _Result:
        id = "task-123"

    def fake_send_task(name: str, **kwargs: Any) -> _Result:
        sent.update(name=name, **kwargs)
        return _Result()

    monkeypatch.setattr(celery_app, "send_task", fake_send_task)
    document_id = uuid.uuid4()
    assert CeleryTaskDispatcher().process_document(document_id) == "task-123"
    assert sent["name"] == PROCESS_DOCUMENT_TASK
    assert sent["args"] == [str(document_id)]
    assert sent["queue"] == INGESTION_QUEUE

    CeleryTaskDispatcher().extract_insights(document_id)
    assert (sent["name"], sent["queue"]) == (EXTRACT_INSIGHTS_TASK, ANALYSIS_QUEUE)
    CeleryTaskDispatcher().compare_documents(document_id)
    assert (sent["name"], sent["queue"]) == (COMPARE_DOCUMENTS_TASK, ANALYSIS_QUEUE)
    assert sent["args"] == [str(document_id)]


def test_process_task_runs_the_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Any,
    ingestion_service: IngestionService,
) -> None:
    recorder = RecordingDispatcher()  # upload only records the job...
    client = make_client(task_dispatcher=recorder)
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "a.pdf", simple_pdf("Celery test")).json()["id"]

    # ...which is then executed by the real task function, eagerly.
    monkeypatch.setattr(document_tasks, "get_ingestion_service", lambda: ingestion_service)
    monkeypatch.setattr(
        document_tasks, "get_settings", lambda: _WorkerSettings(intelligence_enabled=False)
    )
    result = document_tasks.process_document.apply(args=[document_id]).get()
    assert result["outcome"] == "indexed"
    insights = client.get(f"/api/documents/{document_id}/insights", headers=headers)
    assert insights.status_code == 404  # INTELLIGENCE_ENABLED=false: nothing scheduled
    status = client.get(f"/api/documents/{document_id}/status", headers=headers).json()
    assert status["status"] == "indexed"


def test_cleanup_task_deletes_the_file_and_the_vectors(
    monkeypatch: pytest.MonkeyPatch, storage: InMemoryStorage, vector_store: InMemoryVectorStore
) -> None:
    document_id, other_document_id = uuid.uuid4(), uuid.uuid4()
    storage.objects["users/x/documents/y/original.pdf"] = (b"data", "application/pdf")
    vector = HashingEmbedder(dimension=8).embed_query("some text")
    vector_store.upsert(
        [
            VectorPoint(uuid.uuid4(), vector, uuid.uuid4(), document_id, {}),
            VectorPoint(uuid.uuid4(), vector, uuid.uuid4(), other_document_id, {}),
        ]
    )
    monkeypatch.setattr(document_tasks, "get_storage", lambda: storage)
    monkeypatch.setattr(document_tasks, "get_vector_store", lambda: vector_store)
    result = document_tasks.cleanup_document.apply(
        args=[str(document_id), "users/x/documents/y/original.pdf"]
    ).get()
    assert result["status"] == "cleaned"
    assert storage.objects == {}
    assert len(vector_store) == 1  # only the vectors of the deleted document are removed
    assert resources  # module importable without a running broker


def test_process_task_schedules_the_insights_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    make_client: Any,
    ingestion_service: IngestionService,
    intelligence_service: IntelligenceService,
) -> None:
    upload_recorder = RecordingDispatcher()
    client = make_client(task_dispatcher=upload_recorder)
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "a.pdf", simple_pdf("Celery test")).json()["id"]

    worker_dispatcher = RecordingDispatcher()  # what the worker would publish
    monkeypatch.setattr(document_tasks, "get_ingestion_service", lambda: ingestion_service)
    monkeypatch.setattr(document_tasks, "get_intelligence_service", lambda: intelligence_service)
    monkeypatch.setattr(document_tasks, "get_dispatcher", lambda: worker_dispatcher)

    monkeypatch.setattr(
        document_tasks, "get_settings", lambda: _WorkerSettings(intelligence_enabled=True)
    )
    document_tasks.process_document.apply(args=[document_id]).get()

    assert worker_dispatcher.insights == [uuid.UUID(document_id)]
    insight = client.get(f"/api/documents/{document_id}/insights", headers=headers).json()
    assert insight["status"] == "pending"

    # The extraction task then completes it.
    monkeypatch.setattr(
        intelligence_tasks, "get_intelligence_service", lambda: intelligence_service
    )
    result = intelligence_tasks.extract_insights.apply(args=[document_id]).get()
    assert result == {"document_id": document_id, "outcome": "completed"}
    insight = client.get(f"/api/documents/{document_id}/insights", headers=headers).json()
    assert insight["status"] == "completed"


def test_comparison_task_runs_the_job(
    monkeypatch: pytest.MonkeyPatch, make_client: Any, comparison_jobs: ComparisonJobService
) -> None:
    setup = make_client()  # inline jobs: both documents get indexed
    headers = signup(setup, "alice@docintel.dev")
    a = upload(setup, headers, "a.pdf", simple_pdf("The fee is 100 EUR.")).json()["id"]
    b = upload(setup, headers, "b.pdf", simple_pdf("The fee is 120 EUR.")).json()["id"]

    recorder = RecordingDispatcher()
    client = make_client(task_dispatcher=recorder)
    comparison_id = client.post(
        "/api/documents/compare", headers=headers, json={"document_a_id": a, "document_b_id": b}
    ).json()["id"]
    assert recorder.comparisons == [uuid.UUID(comparison_id)]

    monkeypatch.setattr(comparison_tasks, "get_comparison_service", lambda: comparison_jobs)
    result = comparison_tasks.compare_documents.apply(args=[comparison_id]).get()
    assert result == {"comparison_id": comparison_id, "outcome": "completed"}
    body = client.get(f"/api/comparisons/{comparison_id}", headers=headers).json()
    assert body["status"] == "completed"
