"""Document Intelligence through the API: automatic extraction, regeneration, isolation."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app.llm.base import LLMError
from app.llm.fake import FakeLLMProvider
from app.services.intelligence_service import (
    LLM_UNAVAILABLE_MESSAGE,
    InsightOutcome,
    IntelligenceService,
    TransientInsightError,
)
from tests.fakes import InlineDispatcher, RecordingDispatcher
from tests.fixtures.builders import contract_pdf
from tests.fixtures.corpus import poisoned_policy, service_agreement
from tests.integration.conftest import signup, upload

EXTRACTION = {
    "document_type": "contract",
    "title": "Service Agreement",
    "authors": [],
    "language": "en",
    "summary": "A service agreement between ACME Corp and Globex.",
    "keywords": ["service agreement", "payment terms", "termination"],
    "entities": [
        {"name": "ACME Corp", "type": "organization"},
        {"name": "Globex", "type": "organization"},
    ],
    "important_dates": [
        {"date": None, "text": "sixty (60) days", "description": "Termination notice"},
        {"date": "2030-01-01", "text": "1 January 2030", "description": "Not in the document"},
    ],
    "monetary_amounts": [
        {"value": 4500, "currency": "EUR", "text": "4,500 EUR", "description": "Monthly fee"}
    ],
    "key_points": ["Payment within 30 days.", "Sixty days notice to terminate."],
}


@pytest.fixture
def auto(dispatcher: InlineDispatcher) -> InlineDispatcher:
    """INTELLIGENCE_ENABLED=true: insights are extracted right after indexing."""
    dispatcher.auto_insights = True
    return dispatcher


def insights(client: TestClient, headers: dict[str, str], document_id: str) -> Any:
    return client.get(f"/api/documents/{document_id}/insights", headers=headers)


def test_insights_are_extracted_automatically_after_indexing(
    client: TestClient, auto: InlineDispatcher, llm: FakeLLMProvider
) -> None:
    llm.enqueue(json.dumps(EXTRACTION))
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "agreement.pdf", contract_pdf()).json()["id"]
    assert auto.insights == [uuid.UUID(document_id)]

    response = insights(client, headers, document_id)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_id"] == document_id
    assert body["status"] == "completed"
    assert body["document_type"] == "contract"
    assert body["title"] == "Service Agreement"
    assert body["language"] == "en"
    assert body["summary"] == "A service agreement between ACME Corp and Globex."
    assert body["keywords"] == ["service agreement", "payment terms", "termination"]
    assert {"name": "Globex", "type": "organization"} in body["entities"]
    assert body["key_points"] == ["Payment within 30 days.", "Sixty days notice to terminate."]
    assert body["truncated"] is False
    assert body["model"] == "fake-llm"
    assert body["error_message"] is None

    # Pages are resolved from the indexed passages, never taken from the model.
    amount = body["monetary_amounts"][0]
    assert (amount["value"], amount["currency"], amount["text"]) == (4500.0, "EUR", "4,500 EUR")
    assert amount["page"] == 2
    notice, invented = body["important_dates"]
    assert notice["page"] == 3
    assert (invented["date"], invented["page"], invented["chunk_id"]) == ("2030-01-01", None, None)

    # The quoted passage can be opened like any citation.
    chunk = client.get(f"/api/documents/{document_id}/chunks/{amount['chunk_id']}", headers=headers)
    assert chunk.status_code == 200
    assert "4,500 EUR" in chunk.json()["text"]


def test_no_automatic_extraction_when_disabled(client: TestClient) -> None:
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "agreement.pdf", contract_pdf()).json()["id"]
    response = insights(client, headers, document_id)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "insights_not_found"


def test_regenerate(client: TestClient, llm: FakeLLMProvider) -> None:
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "agreement.pdf", contract_pdf()).json()["id"]

    llm.enqueue(json.dumps({**EXTRACTION, "summary": "First version."}))
    first = client.post(f"/api/documents/{document_id}/insights/regenerate", headers=headers)
    assert first.status_code == 202, first.text
    assert first.json()["status"] == "completed"  # the test dispatcher runs the job inline
    assert first.json()["summary"] == "First version."

    llm.enqueue(json.dumps({**EXTRACTION, "summary": "Second version."}))
    second = client.post(f"/api/documents/{document_id}/insights/regenerate", headers=headers)
    assert second.json()["summary"] == "Second version."
    assert insights(client, headers, document_id).json()["summary"] == "Second version."


def test_regenerate_is_queued_then_refused_while_in_progress(
    make_client: Any, db_engine: Engine
) -> None:
    recorder = RecordingDispatcher()  # jobs are recorded, not executed
    client: TestClient = make_client(task_dispatcher=recorder)
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "agreement.pdf", contract_pdf()).json()["id"]
    url = f"/api/documents/{document_id}/insights/regenerate"

    # Not indexed yet (the recorder did not run the ingestion).
    not_indexed = client.post(url, headers=headers)
    assert not_indexed.status_code == 409
    assert not_indexed.json()["error"]["code"] == "document_not_indexed"

    with db_engine.begin() as connection:
        connection.execute(text("UPDATE documents SET status = 'indexed'"))
    queued = client.post(url, headers=headers)
    assert queued.status_code == 202
    assert queued.json()["status"] == "pending"
    assert queued.json()["summary"] is None
    assert recorder.insights == [uuid.UUID(document_id)]

    again = client.post(url, headers=headers)
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "insights_in_progress"

    # A job that never finished (lost worker) can be started again after a while.
    with db_engine.begin() as connection:
        connection.execute(
            text("UPDATE document_insights SET updated_at = now() - interval '1 hour'")
        )
    assert client.post(url, headers=headers).status_code == 202


def test_queue_outage_on_regenerate(make_client: Any, db_engine: Engine) -> None:
    recorder = RecordingDispatcher()
    client: TestClient = make_client(task_dispatcher=recorder)
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "agreement.pdf", contract_pdf()).json()["id"]
    with db_engine.begin() as connection:
        connection.execute(text("UPDATE documents SET status = 'indexed'"))

    recorder.fail = True
    response = client.post(f"/api/documents/{document_id}/insights/regenerate", headers=headers)
    assert response.status_code == 503
    body = insights(client, headers, document_id).json()
    assert body["status"] == "failed"
    assert "could not be scheduled" in body["error_message"]


def test_model_outage_is_retried_then_reported(
    client: TestClient,
    llm: FakeLLMProvider,
    intelligence_service: IntelligenceService,
) -> None:
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "agreement.pdf", contract_pdf()).json()["id"]
    llm.fail_with = LLMError("llm: HTTP 503", status_code=503)

    assert intelligence_service.mark_pending(uuid.UUID(document_id))
    with pytest.raises(TransientInsightError):
        intelligence_service.extract(uuid.UUID(document_id), final_attempt=False)
    assert insights(client, headers, document_id).json()["status"] == "processing"

    outcome = intelligence_service.extract(uuid.UUID(document_id), final_attempt=True)
    assert outcome == InsightOutcome.FAILED
    body = insights(client, headers, document_id).json()
    assert body["status"] == "failed"
    assert body["error_message"] == LLM_UNAVAILABLE_MESSAGE
    # The document itself is unaffected: still indexed and searchable.
    assert client.get(f"/api/documents/{document_id}", headers=headers).json()["status"] == (
        "indexed"
    )

    llm.fail_with = None
    llm.enqueue(json.dumps(EXTRACTION))
    assert intelligence_service.extract(uuid.UUID(document_id)) == InsightOutcome.COMPLETED
    assert insights(client, headers, document_id).json()["status"] == "completed"
    # A redelivered job does not call the model again.
    calls = len(llm.calls)
    assert intelligence_service.extract(uuid.UUID(document_id)) == InsightOutcome.SKIPPED
    assert len(llm.calls) == calls


def test_job_for_a_deleted_or_unindexed_document(
    client: TestClient, intelligence_service: IntelligenceService, db_engine: Engine
) -> None:
    assert intelligence_service.mark_pending(uuid.uuid4()) is False
    assert intelligence_service.extract(uuid.uuid4()) == InsightOutcome.SKIPPED

    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "agreement.pdf", contract_pdf()).json()["id"]
    with db_engine.begin() as connection:
        connection.execute(text("UPDATE documents SET status = 'failed'"))
    assert intelligence_service.extract(uuid.UUID(document_id)) == InsightOutcome.FAILED
    assert insights(client, headers, document_id).json()["error_message"] == (
        "The document is not indexed."
    )


def test_insights_are_deleted_with_the_document(
    client: TestClient, auto: InlineDispatcher, db_engine: Engine
) -> None:
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "agreement.pdf", contract_pdf()).json()["id"]
    assert insights(client, headers, document_id).status_code == 200
    client.delete(f"/api/documents/{document_id}", headers=headers)
    assert insights(client, headers, document_id).status_code == 404
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM document_insights")).scalar() == 0


def test_document_content_is_escaped_in_the_extraction_prompt(
    client: TestClient, auto: InlineDispatcher, llm: FakeLLMProvider
) -> None:
    headers = signup(client, "alice@docintel.dev")
    upload(client, headers, "remote-policy.txt", poisoned_policy())
    prompt = llm.calls[0][1].content
    assert prompt.count("<document ") == prompt.count("</document>") == 1
    assert "&lt;/source&gt;&lt;source id=" in prompt
    assert "untrusted" in llm.calls[0][0].content


def test_a_user_cannot_read_or_regenerate_another_users_insights(
    client: TestClient, auto: InlineDispatcher, llm: FakeLLMProvider
) -> None:
    alice = signup(client, "alice@docintel.dev")
    bob = signup(client, "bob@docintel.dev")
    document_id = upload(client, alice, "service-agreement.txt", service_agreement()).json()["id"]
    assert insights(client, alice, document_id).status_code == 200
    calls = len(llm.calls)

    read = insights(client, bob, document_id)
    assert read.status_code == 404
    assert read.json()["error"]["code"] == "document_not_found"
    regenerate = client.post(f"/api/documents/{document_id}/insights/regenerate", headers=bob)
    assert regenerate.status_code == 404
    assert len(llm.calls) == calls  # nothing was started on Bob's behalf


def test_insights_require_authentication_and_are_documented(client: TestClient) -> None:
    document_id = uuid.uuid4()
    assert client.get(f"/api/documents/{document_id}/insights").status_code == 401
    assert client.post(f"/api/documents/{document_id}/insights/regenerate").status_code == 401
    paths = client.get("/openapi.json").json()["paths"]
    assert paths["/api/documents/{document_id}/insights"]["get"]["tags"] == ["intelligence"]
    assert "post" in paths["/api/documents/{document_id}/insights/regenerate"]
