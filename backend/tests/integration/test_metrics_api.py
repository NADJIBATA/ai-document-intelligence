"""GET /metrics: HTTP, retrieval and chat metrics, values read at scrape time, job
metrics written by the worker, and no user data in the labels."""

from __future__ import annotations

import re
import uuid
from typing import Any

import redis
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.metrics import JOB_BUCKET_KEY, JOB_COUNT_KEY, JOB_SUM_KEY, JobMetricsStore
from app.services.metrics_service import Collector
from tests.fixtures.corpus import service_agreement
from tests.integration.conftest import signup, upload

# Label values may contain braces (route templates): quoted strings are skipped whole.
SAMPLE = re.compile(
    r'^(?P<series>[a-zA-Z_:][\w:]*(?:\{(?:[^"}]|"(?:[^"\\]|\\.)*")*\})?) (?P<value>\S+)$'
)


def scrape(client: TestClient) -> dict[str, float]:
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    samples: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line)
        assert match, f"invalid exposition line: {line!r}"
        samples[match.group("series")] = float(match.group("value"))
    return samples


def test_metrics_cover_http_retrieval_chat_and_documents(client: TestClient) -> None:
    before = scrape(client)
    headers = signup(client, "alice@docintel.dev")
    document = upload(client, headers, "service-agreement.txt", service_agreement()).json()
    assert client.get(f"/api/documents/{document['id']}", headers=headers).status_code == 200
    question = {"query": "What is the notice period?"}
    assert client.post("/api/search", headers=headers, json=question).status_code == 200
    chat = client.post(
        "/api/chat", headers=headers, json={"question": "What is the notice period?"}
    )
    assert chat.status_code == 200, chat.text

    after = scrape(client)

    def delta(series: str) -> float:
        return after.get(series, 0.0) - before.get(series, 0.0)

    # HTTP: route templates, never raw identifiers.
    assert (
        delta('docintel_http_requests_total{method="POST",route="/api/documents",status="202"}')
        == 1
    )
    assert (
        delta(
            'docintel_http_requests_total{method="GET",route="/api/documents/{document_id}",'
            'status="200"}'
        )
        == 1
    )
    assert (
        delta('docintel_http_request_duration_seconds_count{method="POST",route="/api/chat"}') == 1
    )
    # Retrieval and chat.
    assert delta('docintel_retrievals_total{origin="search"}') == 1
    assert delta('docintel_retrievals_total{origin="chat"}') == 1
    assert delta('docintel_retrieval_stage_duration_seconds_count{stage="total"}') == 2
    assert delta('docintel_chat_answers_total{outcome="answered"}') == 1
    assert delta('docintel_llm_request_duration_seconds_count{operation="chat"}') == 1
    # Read at scrape time.
    assert after['docintel_documents{status="indexed"}'] == 1
    assert after['docintel_documents{status="failed"}'] == 0
    assert after["docintel_users"] == 1
    assert 'docintel_queue_length{queue="ingestion"}' in after
    assert 'docintel_queue_length{queue="analysis"}' in after

    text = client.get("/metrics").text
    assert document["id"] not in text
    assert "alice" not in text
    assert "notice period" not in text
    assert "/metrics" not in text  # the scrapes themselves are not counted


def test_unknown_paths_share_one_series(client: TestClient) -> None:
    before = scrape(client)
    for _ in range(3):
        client.get(f"/no-such-page/{uuid.uuid4()}")
    after = scrape(client)
    series = 'docintel_http_requests_total{method="GET",route="unmatched",status="404"}'
    assert after[series] - before.get(series, 0.0) == 3


def test_job_metrics_written_by_the_worker_are_exposed(
    client: TestClient, integration_settings: Settings
) -> None:
    connection = redis.Redis.from_url(integration_settings.redis_url)
    try:
        store = JobMetricsStore(connection)
        store.record("documents.process", "SUCCESS", 3.0)
        store.record("documents.process", "FAILURE", 0.2)
        samples = scrape(client)
    finally:
        connection.delete(JOB_COUNT_KEY, JOB_SUM_KEY, JOB_BUCKET_KEY)
        connection.close()

    assert samples['docintel_jobs_total{task="documents.process",state="success"}'] == 1
    assert samples['docintel_jobs_total{task="documents.process",state="failure"}'] == 1
    assert samples['docintel_job_duration_seconds_count{task="documents.process"}'] == 2
    assert samples['docintel_job_duration_seconds_bucket{task="documents.process",le="5"}'] == 2


def test_a_failing_source_does_not_break_the_scrape(client: TestClient) -> None:
    async def broken() -> list[Any]:
        raise ConnectionError("database down")

    collectors: dict[str, Collector] = client.app.state.metrics_collectors  # type: ignore[attr-defined]
    collectors["broken"] = broken
    try:
        samples = scrape(client)
        samples = scrape(client)
    finally:
        del collectors["broken"]
    assert samples['docintel_metrics_collector_errors_total{collector="broken"}'] >= 2
    assert 'docintel_documents{status="indexed"}' in samples


def test_metrics_can_be_disabled(make_client: Any) -> None:
    client = make_client(metrics_enabled=False)
    assert client.get("/metrics").status_code == 404
