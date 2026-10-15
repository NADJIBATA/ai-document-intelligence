"""User A can never retrieve the chunks of user B through the search.

Bob owns a memo whose vocabulary ("zebra", "falcon") exists nowhere else, and a clause
that looks a lot like one of Alice's ("termination ... notice period"): whatever Alice
asks, on whatever retrieval path, no passage of Bob may come back.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app.rag.embeddings import EmbeddingProvider
from app.rag.factory import RetrievalComponents
from app.rag.vector_store import InMemoryVectorStore, VectorPoint
from tests.fixtures.corpus import secret_merger_memo, service_agreement
from tests.integration.conftest import signup, upload

QUERIES = [
    "Project Zebra acquisition of Falcon Industries",  # only Bob's document matches
    "termination notice period",  # both users have a matching clause
    "ninety million euros board approved confidential",
]


def _chunk_ids(db_engine: Engine, document_id: str) -> set[str]:
    with db_engine.connect() as connection:
        rows = connection.execute(
            text("SELECT id FROM document_chunks WHERE document_id = :d"), {"d": document_id}
        ).all()
    return {str(row.id) for row in rows}


def _user_id(db_engine: Engine, email: str) -> uuid.UUID:
    with db_engine.connect() as connection:
        value: uuid.UUID = connection.execute(
            text("SELECT id FROM users WHERE email = :e"), {"e": email}
        ).scalar_one()
    return value


@pytest.fixture
def world(client: TestClient, db_engine: Engine) -> dict[str, Any]:
    alice = signup(client, "alice@docintel.dev")
    bob = signup(client, "bob@docintel.dev")
    alice_doc = upload(client, alice, "service-agreement.txt", service_agreement()).json()["id"]
    bob_doc = upload(client, bob, "merger-memo.txt", secret_merger_memo()).json()["id"]
    return {
        "alice": alice,
        "bob": bob,
        "alice_id": _user_id(db_engine, "alice@docintel.dev"),
        "bob_id": _user_id(db_engine, "bob@docintel.dev"),
        "alice_doc": alice_doc,
        "bob_doc": bob_doc,
        "alice_chunks": _chunk_ids(db_engine, alice_doc),
        "bob_chunks": _chunk_ids(db_engine, bob_doc),
    }


def _search(client: TestClient, headers: dict[str, str], query: str, **body: Any) -> Any:
    response = client.post("/api/search", headers=headers, json={"query": query, **body})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("query", QUERIES)
def test_api_search_never_returns_another_users_chunks(
    client: TestClient, world: dict[str, Any], query: str
) -> None:
    alice_results = _search(client, world["alice"], query, top_k=50)["results"]
    assert {result["chunk_id"] for result in alice_results} <= world["alice_chunks"]
    assert {result["document_id"] for result in alice_results} <= {world["alice_doc"]}
    assert all("Falcon" not in result["text"] for result in alice_results)

    bob_results = _search(client, world["bob"], query, top_k=50)["results"]
    assert bob_results, "the owner does find his own document"
    assert {result["chunk_id"] for result in bob_results} <= world["bob_chunks"]


def test_asking_for_a_foreign_document_id_returns_nothing(
    client: TestClient, world: dict[str, Any]
) -> None:
    body = _search(
        client, world["alice"], "Project Zebra Falcon Industries", document_ids=[world["bob_doc"]]
    )
    assert body["results"] == []
    assert body["candidate_count"] == 0

    mixed = _search(
        client,
        world["alice"],
        "termination notice period",
        document_ids=[world["bob_doc"], world["alice_doc"]],
    )
    assert mixed["results"]
    assert {result["document_id"] for result in mixed["results"]} == {world["alice_doc"]}


@pytest.mark.parametrize(
    "options",
    [
        {"use_dense": False, "use_reranker": False},  # BM25 only
        {"use_lexical": False, "use_reranker": False},  # dense (vector store) only
        {"use_reranker": False},  # hybrid
        {},  # hybrid + reranker
    ],
    ids=["bm25", "dense", "hybrid", "hybrid+rerank"],
)
def test_every_retrieval_path_is_scoped_to_the_user(
    world: dict[str, Any], retrieval: RetrievalComponents, options: dict[str, bool]
) -> None:
    for query in QUERIES:
        alice = retrieval.pipeline.retrieve(query, world["alice_id"], top_k=50, **options)
        assert {str(chunk.chunk_id) for chunk in alice.chunks} <= world["alice_chunks"], query
        bob = retrieval.pipeline.retrieve(query, world["bob_id"], top_k=50, **options)
        assert bob.chunks, query
        assert {str(chunk.chunk_id) for chunk in bob.chunks} <= world["bob_chunks"], query


def test_the_filter_is_applied_by_the_retrievers_themselves(
    world: dict[str, Any], retrieval: RetrievalComponents, embedder: EmbeddingProvider
) -> None:
    """Not only by the final hydration: each stage is safe on its own."""
    query = "Project Zebra acquisition of Falcon Industries"
    lexical = retrieval.lexical.search(query, world["alice_id"], limit=50)
    assert {str(hit.chunk_id) for hit in lexical} <= world["alice_chunks"]

    dense = retrieval.vector_store.search(embedder.embed_query(query), world["alice_id"], limit=50)
    assert {str(hit.chunk_id) for hit in dense} <= world["alice_chunks"]
    assert len(dense) == len(world["alice_chunks"])  # everything Alice owns, nothing else

    bob_dense = retrieval.vector_store.search(
        embedder.embed_query(query), world["bob_id"], limit=50
    )
    assert {str(hit.chunk_id) for hit in bob_dense} == world["bob_chunks"]


def test_a_corrupted_vector_index_cannot_leak_a_passage(
    client: TestClient,
    world: dict[str, Any],
    vector_store: InMemoryVectorStore,
    embedder: EmbeddingProvider,
) -> None:
    """Defence in depth: vectors of Bob's chunks wrongly tagged with Alice's user id
    (bug, manual operation on Qdrant...) are still filtered out by PostgreSQL."""
    query = "Project Zebra acquisition of Falcon Industries"
    vector_store.upsert(
        [
            VectorPoint(
                chunk_id=uuid.UUID(chunk_id),
                vector=embedder.embed_query(query),
                user_id=world["alice_id"],
                document_id=uuid.UUID(world["bob_doc"]),
                payload={},
            )
            for chunk_id in world["bob_chunks"]
        ]
    )
    results = _search(client, world["alice"], query, top_k=50)["results"]
    assert {result["chunk_id"] for result in results} <= world["alice_chunks"]


def test_deleting_a_user_document_does_not_affect_the_other_user(
    client: TestClient, world: dict[str, Any]
) -> None:
    client.delete(f"/api/documents/{world['bob_doc']}", headers=world["bob"])
    assert _search(client, world["bob"], "Project Zebra")["results"] == []
    results = _search(client, world["alice"], "termination notice period")["results"]
    assert results[0]["document_id"] == world["alice_doc"]
