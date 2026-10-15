"""Document comparison through the API: job, sources, history, isolation."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.llm.base import LLMError
from app.llm.fake import FakeLLMProvider
from app.models.comparison import DocumentComparison
from app.models.document import Document
from app.rag.factory import RetrievalComponents
from app.repositories.corpus import PostgresChunkStore
from app.services.comparison_service import (
    DOCUMENT_UNAVAILABLE_MESSAGE,
    LLM_UNAVAILABLE_MESSAGE,
    ComparisonJobService,
    ComparisonOutcome,
    TransientComparisonError,
    build_document_comparator,
)
from tests.conftest import build_settings
from tests.fakes import RecordingDispatcher
from tests.fixtures.corpus import contract_2025, contract_2026, secret_merger_memo
from tests.integration.conftest import signup, upload

RESULT = {
    "summary": "The fee and the notice period increased; a penalty clause was added.",
    "differences": [
        {
            "aspect": "pricing",
            "title": "Monthly fee",
            "significance": "high",
            "document_a": {"value": "1,000 euros", "refs": ["A1"]},
            "document_b": {"value": "1,200 euros", "refs": ["B1"]},
        },
        {
            "aspect": "deadlines",
            "title": "Notice period",
            "significance": "high",
            "document_a": {"value": "Thirty days", "refs": ["A2"]},
            "document_b": {"value": "Ninety days", "refs": ["B2"]},
        },
        {
            "aspect": "obligations",
            "title": "Late payment penalty",
            "significance": "medium",
            "document_a": {"value": None, "refs": []},
            "document_b": {"value": "Two percent per month", "refs": ["B4"]},
        },
        {
            "aspect": "clauses",
            "title": "Not supported by any passage",
            "document_a": {"value": "x", "refs": ["A99"]},
            "document_b": {"value": "y", "refs": ["B99"]},
        },
    ],
    "similarities": ["The liability cap is identical."],
}


@pytest.fixture
def alice(client: TestClient) -> dict[str, Any]:
    headers = signup(client, "alice@docintel.dev")
    a = upload(client, headers, "contract_2025.txt", contract_2025()).json()["id"]
    b = upload(client, headers, "contract_2026.txt", contract_2026()).json()["id"]
    return {"headers": headers, "a": a, "b": b}


def _insert_comparison(
    session_factory: sessionmaker[Session], a: str, b: str, *, aspects: list[str]
) -> uuid.UUID:
    """Write a pending comparison owned by the owner of document ``a``, bypassing the API."""
    with session_factory() as session:
        owner_id = session.get_one(Document, uuid.UUID(a)).owner_id
        comparison = DocumentComparison(
            id=uuid.uuid4(),
            owner_id=owner_id,
            document_a_id=uuid.UUID(a),
            document_b_id=uuid.UUID(b),
            document_a_name="a",
            document_b_name="b",
            aspects=aspects,
        )
        session.add(comparison)
        session.commit()
        return comparison.id


def compare(client: TestClient, headers: dict[str, str], a: str, b: str, **body: Any) -> Any:
    return client.post(
        "/api/documents/compare",
        headers=headers,
        json={"document_a_id": a, "document_b_id": b, **body},
    )


def test_comparison_lists_differences_with_their_sources(
    client: TestClient, alice: dict[str, Any], llm: FakeLLMProvider
) -> None:
    llm.enqueue(json.dumps(RESULT))
    response = compare(client, alice["headers"], alice["a"], alice["b"])
    assert response.status_code == 202, response.text
    created = response.json()
    assert created["status"] == "completed"  # the test dispatcher runs the job inline
    assert created["document_a"] == {"id": alice["a"], "name": "contract_2025.txt"}
    assert created["document_b"] == {"id": alice["b"], "name": "contract_2026.txt"}
    assert created["aspects"] == [
        "pricing",
        "dates",
        "conditions",
        "clauses",
        "deadlines",
        "obligations",
    ]

    body = client.get(f"/api/comparisons/{created['id']}", headers=alice["headers"]).json()
    assert body == created
    result = body["result"]
    assert result["summary"].startswith("The fee and the notice period increased")
    assert result["similarities"] == ["The liability cap is identical."]
    assert result["model"] == "fake-llm"
    # The difference without any valid passage identifier is not shown.
    assert [d["title"] for d in result["differences"]] == [
        "Monthly fee",
        "Notice period",
        "Late payment penalty",
    ]

    fee = result["differences"][0]
    assert (fee["aspect"], fee["significance"]) == ("pricing", "high")
    source_a = fee["document_a"]["sources"][0]
    assert (source_a["ref"], source_a["document"]) == ("A1", "contract_2025.txt")
    assert source_a["document_id"] == alice["a"]
    assert "1,000 euros" in source_a["excerpt"]
    source_b = fee["document_b"]["sources"][0]
    assert (source_b["ref"], source_b["document_id"]) == ("B1", alice["b"])
    assert "1,200 euros" in source_b["excerpt"]

    penalty = result["differences"][2]
    assert penalty["document_a"] == {"value": None, "sources": []}
    assert "two percent" in penalty["document_b"]["sources"][0]["excerpt"]

    # Sources can be opened like chat citations.
    chunk = client.get(
        f"/api/documents/{source_b['document_id']}/chunks/{source_b['chunk_id']}",
        headers=alice["headers"],
    )
    assert chunk.status_code == 200

    # The model received the passages of both documents, labelled per side.
    prompt = llm.calls[0][1].content
    assert prompt.count('<source id="A') == 3
    assert prompt.count('<source id="B') == 4
    assert '<document_a name="contract_2025.txt">' in prompt


def test_requested_aspects(client: TestClient, alice: dict[str, Any], llm: FakeLLMProvider) -> None:
    body = compare(
        client, alice["headers"], alice["a"], alice["b"], aspects=["pricing", "pricing", "dates"]
    ).json()
    assert body["aspects"] == ["pricing", "dates"]
    assert "<aspects>\npricing, dates\n</aspects>" in llm.calls[0][1].content


def test_history(client: TestClient, alice: dict[str, Any]) -> None:
    first = compare(client, alice["headers"], alice["a"], alice["b"]).json()["id"]
    second = compare(client, alice["headers"], alice["b"], alice["a"]).json()["id"]
    listing = client.get("/api/comparisons", headers=alice["headers"]).json()
    assert listing["total"] == 2
    assert [item["id"] for item in listing["items"]] == [second, first]
    assert "result" not in listing["items"][0]
    assert listing["items"][0]["document_a"]["name"] == "contract_2026.txt"
    page = client.get("/api/comparisons?limit=1&offset=1", headers=alice["headers"]).json()
    assert [item["id"] for item in page["items"]] == [first]

    assert client.delete(f"/api/comparisons/{first}", headers=alice["headers"]).status_code == 204
    assert client.get(f"/api/comparisons/{first}", headers=alice["headers"]).status_code == 404
    assert client.delete(f"/api/comparisons/{first}", headers=alice["headers"]).status_code == 404
    assert client.get("/api/comparisons", headers=alice["headers"]).json()["total"] == 1


def test_result_survives_the_deletion_of_a_document(
    client: TestClient, alice: dict[str, Any], llm: FakeLLMProvider
) -> None:
    llm.enqueue(json.dumps(RESULT))
    comparison_id = compare(client, alice["headers"], alice["a"], alice["b"]).json()["id"]
    client.delete(f"/api/documents/{alice['a']}", headers=alice["headers"])
    body = client.get(f"/api/comparisons/{comparison_id}", headers=alice["headers"]).json()
    assert body["document_a"] == {"id": None, "name": "contract_2025.txt"}
    assert body["document_b"]["id"] == alice["b"]
    assert len(body["result"]["differences"]) == 3


def test_validation(client: TestClient, alice: dict[str, Any]) -> None:
    headers = alice["headers"]
    same = compare(client, headers, alice["a"], alice["a"])
    assert same.status_code == 422
    assert compare(client, headers, alice["a"], "not-a-uuid").status_code == 422
    assert (
        compare(client, headers, alice["a"], alice["b"], aspects=["astrology"]).status_code == 422
    )
    assert compare(client, headers, alice["a"], alice["b"], aspects=[]).status_code == 422

    unknown = compare(client, headers, alice["a"], str(uuid.uuid4()))
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "document_not_found"
    assert client.get("/api/comparisons", headers=headers).json()["total"] == 0

    assert client.get(f"/api/comparisons/{uuid.uuid4()}", headers=headers).status_code == 404
    assert client.post("/api/documents/compare", json={}).status_code == 401
    assert client.get("/api/comparisons").status_code == 401


def test_documents_must_be_indexed(make_client: Any) -> None:
    recorder = RecordingDispatcher()  # uploads are recorded, not processed
    client: TestClient = make_client(task_dispatcher=recorder)
    headers = signup(client, "alice@docintel.dev")
    a = upload(client, headers, "contract_2025.txt", contract_2025()).json()["id"]
    b = upload(client, headers, "contract_2026.txt", contract_2026()).json()["id"]
    response = compare(client, headers, a, b)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "document_not_indexed"
    assert recorder.comparisons == []


def test_comparison_is_queued_and_queue_outage_is_reported(
    make_client: Any, db_engine: Engine
) -> None:
    recorder = RecordingDispatcher()
    client: TestClient = make_client(task_dispatcher=recorder)
    headers = signup(client, "alice@docintel.dev")
    a = upload(client, headers, "contract_2025.txt", contract_2025()).json()["id"]
    b = upload(client, headers, "contract_2026.txt", contract_2026()).json()["id"]
    with db_engine.begin() as connection:
        connection.execute(text("UPDATE documents SET status = 'indexed'"))

    queued = compare(client, headers, a, b)
    assert queued.status_code == 202
    body = queued.json()
    assert (body["status"], body["result"], body["completed_at"]) == ("pending", None, None)
    assert recorder.comparisons == [uuid.UUID(body["id"])]

    recorder.fail = True
    assert compare(client, headers, a, b).status_code == 503
    listing = client.get("/api/comparisons", headers=headers).json()
    assert listing["items"][0]["status"] == "failed"
    assert "could not be scheduled" in listing["items"][0]["error_message"]


def test_model_outage_is_retried_then_reported(
    make_client: Any, llm: FakeLLMProvider, comparison_jobs: ComparisonJobService, db_engine: Engine
) -> None:
    recorder = RecordingDispatcher()
    setup: TestClient = make_client()  # inline jobs: documents get indexed
    headers = signup(setup, "alice@docintel.dev")
    a = upload(setup, headers, "contract_2025.txt", contract_2025()).json()["id"]
    b = upload(setup, headers, "contract_2026.txt", contract_2026()).json()["id"]
    client: TestClient = make_client(task_dispatcher=recorder)
    comparison_id = uuid.UUID(compare(client, headers, a, b).json()["id"])

    llm.fail_with = LLMError("llm: HTTP 503", status_code=503)
    with pytest.raises(TransientComparisonError):
        comparison_jobs.run(comparison_id, final_attempt=False)
    assert client.get(f"/api/comparisons/{comparison_id}", headers=headers).json()["status"] == (
        "processing"
    )
    assert comparison_jobs.run(comparison_id, final_attempt=True) == ComparisonOutcome.FAILED
    body = client.get(f"/api/comparisons/{comparison_id}", headers=headers).json()
    assert (body["status"], body["error_message"]) == ("failed", LLM_UNAVAILABLE_MESSAGE)

    llm.fail_with = None
    assert comparison_jobs.run(comparison_id) == ComparisonOutcome.COMPLETED
    calls = len(llm.calls)
    assert comparison_jobs.run(comparison_id) == ComparisonOutcome.SKIPPED  # redelivery
    assert len(llm.calls) == calls
    assert comparison_jobs.run(uuid.uuid4()) == ComparisonOutcome.SKIPPED

    # A document deleted before the job runs: clear failure, no model call.
    pending = uuid.UUID(compare(client, headers, a, b).json()["id"])
    with db_engine.begin() as connection:
        connection.execute(text("DELETE FROM documents WHERE id = :id"), {"id": a})
    assert comparison_jobs.run(pending) == ComparisonOutcome.FAILED
    assert len(llm.calls) == calls


def test_long_documents_use_a_search_restricted_to_each_document(
    client: TestClient,
    alice: dict[str, Any],
    llm: FakeLLMProvider,
    retrieval: RetrievalComponents,
    session_factory: sessionmaker[Session],
) -> None:
    bob = signup(client, "bob@docintel.dev")
    upload(client, bob, "merger-memo.txt", secret_merger_memo())  # also talks about termination

    settings = build_settings(
        comparison_max_passages_per_document=2, comparison_passages_per_aspect=1
    )
    comparator = build_document_comparator(
        llm, retrieval.pipeline, PostgresChunkStore(session_factory), settings
    )
    jobs = ComparisonJobService(session_factory, comparator)
    comparison_id = _insert_comparison(
        session_factory, alice["a"], alice["b"], aspects=["pricing", "deadlines"]
    )

    llm.calls.clear()
    assert jobs.run(comparison_id) == ComparisonOutcome.COMPLETED
    prompt = llm.calls[0][1].content
    block_a, block_b = prompt.split("<document_b ")
    # Each side only contains passages of its own document, at most two of them.
    assert 1 <= block_a.count("<source id=") <= 2
    assert 1 <= block_b.count("<source id=") <= 2
    assert "1,200 euros" not in block_a
    assert "ninety days" not in block_a
    assert "1,000 euros" not in block_b
    assert "Falcon" not in prompt  # Bob's document is never searched


# ------------------------------------------------------------------ isolation
def test_a_user_cannot_compare_or_read_another_users_documents(
    client: TestClient, alice: dict[str, Any], llm: FakeLLMProvider
) -> None:
    bob = signup(client, "bob@docintel.dev")
    bob_doc = upload(client, bob, "merger-memo.txt", secret_merger_memo()).json()["id"]
    alice_comparison = compare(client, alice["headers"], alice["a"], alice["b"]).json()["id"]
    calls = len(llm.calls)

    # Alice cannot use Bob's document, on either side.
    for a, b in ((alice["a"], bob_doc), (bob_doc, alice["a"])):
        response = compare(client, alice["headers"], a, b)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "document_not_found"
    # Bob cannot compare Alice's documents, nor read or delete her comparison.
    assert compare(client, bob, alice["a"], alice["b"]).status_code == 404
    assert client.get(f"/api/comparisons/{alice_comparison}", headers=bob).status_code == 404
    assert client.delete(f"/api/comparisons/{alice_comparison}", headers=bob).status_code == 404
    assert client.get("/api/comparisons", headers=bob).json()["total"] == 0

    assert len(llm.calls) == calls  # no model call was made for the refused requests
    assert client.get("/api/comparisons", headers=alice["headers"]).json()["total"] == 1
    assert (
        client.get(f"/api/comparisons/{alice_comparison}", headers=alice["headers"]).status_code
        == 200
    )


def test_job_refuses_documents_of_another_owner(
    client: TestClient,
    alice: dict[str, Any],
    comparison_jobs: ComparisonJobService,
    session_factory: sessionmaker[Session],
    llm: FakeLLMProvider,
) -> None:
    """Defence in depth: even a forged row cannot make the worker read foreign passages."""
    bob = signup(client, "bob@docintel.dev")
    bob_doc = upload(client, bob, "merger-memo.txt", secret_merger_memo()).json()["id"]
    forged_id = _insert_comparison(session_factory, alice["a"], bob_doc, aspects=["pricing"])

    llm.calls.clear()
    assert comparison_jobs.run(forged_id) == ComparisonOutcome.FAILED
    assert llm.calls == []
    with session_factory() as session:
        # Refused at the ownership check, before any passage is read.
        assert session.get_one(DocumentComparison, forged_id).error_message == (
            DOCUMENT_UNAVAILABLE_MESSAGE
        )


def test_compare_is_rate_limited_and_documented(make_client: Any) -> None:
    limited: TestClient = make_client(rate_limit_analysis="1/minute")
    headers = signup(limited, "alice@docintel.dev")
    a = upload(limited, headers, "contract_2025.txt", contract_2025()).json()["id"]
    b = upload(limited, headers, "contract_2026.txt", contract_2026()).json()["id"]
    assert compare(limited, headers, a, b).status_code == 202
    assert compare(limited, headers, a, b).status_code == 429

    paths = limited.get("/openapi.json").json()["paths"]
    assert {"202", "404", "409", "422", "429", "503"} <= set(
        paths["/api/documents/compare"]["post"]["responses"]
    )
    assert {"get", "delete"} <= set(paths["/api/comparisons/{comparison_id}"])
