"""Hybrid search through the API: upload -> ingestion (chunks + vectors) -> search."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi.testclient import TestClient

from app.rag.vector_store import InMemoryVectorStore
from tests.fixtures.corpus import service_agreement, travel_policy
from tests.integration.conftest import signup, upload

AGREEMENT = "service-agreement.txt"
POLICY = "travel-policy.txt"


def search(client: TestClient, headers: dict[str, str], query: str, **body: Any) -> Any:
    return client.post("/api/search", headers=headers, json={"query": query, **body})


def indexed_user(client: TestClient, email: str = "alice@docintel.dev") -> dict[str, Any]:
    headers = signup(client, email)
    agreement = upload(client, headers, AGREEMENT, service_agreement()).json()
    policy = upload(client, headers, POLICY, travel_policy()).json()
    return {"headers": headers, "agreement": agreement["id"], "policy": policy["id"]}


def test_search_returns_the_relevant_passage_with_its_source(client: TestClient) -> None:
    user = indexed_user(client)
    response = search(client, user["headers"], "What is the notice period to terminate?")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["query"] == "What is the notice period to terminate?"
    assert body["degraded"] == []
    assert body["candidate_count"] >= 1
    assert 1 <= len(body["results"]) <= 5

    best = body["results"][0]
    assert best["document_id"] == user["agreement"]
    assert best["filename"] == AGREEMENT
    assert "notice period of thirty days" in best["text"]
    assert best["bm25_rank"] == 1
    assert best["dense_rank"] is not None
    assert best["rerank_score"] is not None
    assert best["score"] == best["rerank_score"]
    scores = [result["score"] for result in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert {"bm25", "dense", "fusion", "hydration", "rerank", "total"} <= set(body["timings_ms"])

    # The passage can be opened through the documents API (used by the citations).
    chunk = client.get(
        f"/api/documents/{best['document_id']}/chunks/{best['chunk_id']}", headers=user["headers"]
    )
    assert chunk.status_code == 200
    assert chunk.json()["text"] == best["text"]


def test_search_finds_passages_across_documents(client: TestClient) -> None:
    user = indexed_user(client)
    hotel = search(client, user["headers"], "nightly hotel budget").json()["results"][0]
    assert hotel["document_id"] == user["policy"]
    assert "one hundred and eighty euros" in hotel["text"]

    penalties = search(client, user["headers"], "late payment penalties").json()["results"][0]
    assert penalties["document_id"] == user["agreement"]


def test_search_can_be_restricted_to_documents(client: TestClient) -> None:
    user = indexed_user(client)
    body = search(
        client, user["headers"], "notice period approval", document_ids=[user["policy"]]
    ).json()
    assert body["results"]
    assert {result["document_id"] for result in body["results"]} == {user["policy"]}

    unknown = search(
        client, user["headers"], "notice period", document_ids=[str(uuid.uuid4())]
    ).json()
    assert unknown["results"] == []


def test_top_k(client: TestClient) -> None:
    user = indexed_user(client)
    body = search(client, user["headers"], "agreement payment hotel expenses", top_k=2).json()
    assert len(body["results"]) == 2


def test_search_without_documents_returns_nothing(client: TestClient) -> None:
    headers = signup(client, "empty@docintel.dev")
    body = search(client, headers, "notice period").json()
    assert body["results"] == []
    assert body["candidate_count"] == 0


def test_new_documents_are_searchable_immediately(client: TestClient) -> None:
    """The per-user BM25 cache is invalidated by the corpus signature."""
    headers = signup(client, "alice@docintel.dev")
    upload(client, headers, AGREEMENT, service_agreement())
    before = search(client, headers, "nightly hotel budget").json()["results"]
    assert all(result["filename"] != POLICY for result in before)
    upload(client, headers, POLICY, travel_policy())
    best = search(client, headers, "nightly hotel budget").json()["results"][0]
    assert best["filename"] == POLICY


def test_deleted_documents_disappear_from_search_and_from_the_vector_store(
    client: TestClient, vector_store: InMemoryVectorStore
) -> None:
    user = indexed_user(client)
    assert len(vector_store) > 0
    before = search(client, user["headers"], "notice period thirty days").json()
    assert before["results"][0]["document_id"] == user["agreement"]

    assert (
        client.delete(f"/api/documents/{user['agreement']}", headers=user["headers"]).status_code
        == 204
    )
    after = search(client, user["headers"], "notice period thirty days").json()
    assert all(result["document_id"] != user["agreement"] for result in after["results"])

    client.delete(f"/api/documents/{user['policy']}", headers=user["headers"])
    assert len(vector_store) == 0
    assert search(client, user["headers"], "hotel").json()["results"] == []


def test_search_requires_authentication(client: TestClient) -> None:
    response = client.post("/api/search", json={"query": "notice period"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "not_authenticated"


def test_search_validation(client: TestClient, make_client: Any) -> None:
    headers = signup(client, "alice@docintel.dev")
    for payload in (
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 51},
        {"query": "ok", "document_ids": ["not-a-uuid"]},
    ):
        response = client.post("/api/search", headers=headers, json=payload)
        assert response.status_code == 422, payload
        assert response.json()["error"]["code"] == "validation_error"

    strict: TestClient = make_client(max_question_chars=50)
    response = search(strict, headers, "long question " * 10)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "query_too_long"


def test_search_is_rate_limited_per_user(make_client: Any) -> None:
    limited: TestClient = make_client(rate_limit_search="2/minute")
    alice = signup(limited, "alice@docintel.dev")
    bob = signup(limited, "bob@docintel.dev")
    assert search(limited, alice, "notice").status_code == 200
    assert search(limited, alice, "notice").status_code == 200
    blocked = search(limited, alice, "notice")
    assert blocked.status_code == 429
    assert "retry-after" in blocked.headers
    assert search(limited, bob, "notice").status_code == 200  # another user is not affected


def test_search_is_documented(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/search"]["post"]
    assert operation["tags"] == ["search"]
    assert {"200", "400", "401", "422", "429"} <= set(operation["responses"])


def test_readiness_reports_the_vector_store(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["checks"]["vector_store"]["status"] == "ok"
