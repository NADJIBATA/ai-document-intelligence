"""QdrantVectorStore against a real Qdrant (qdrant-test in docker-compose.test.yml).

Skipped when Qdrant is not reachable; set ``REQUIRE_QDRANT=1`` (CI) to fail instead.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import httpx
import pytest

from app.rag.vector_store import QdrantVectorStore, VectorPoint, VectorStoreError

QDRANT_URL = os.environ.get("TEST_QDRANT_URL", "http://127.0.0.1:6335")

ALICE = uuid.uuid4()
BOB = uuid.uuid4()
DOC_ALICE_1 = uuid.uuid4()
DOC_ALICE_2 = uuid.uuid4()
DOC_BOB = uuid.uuid4()


@pytest.fixture(scope="module")
def qdrant_available() -> None:
    try:
        httpx.get(f"{QDRANT_URL}/readyz", timeout=3).raise_for_status()
    except httpx.HTTPError as exc:
        reason = f"Qdrant is not reachable at {QDRANT_URL}: {type(exc).__name__}"
        if os.environ.get("REQUIRE_QDRANT") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.fixture
def collection(qdrant_available: None) -> Iterator[str]:
    name = f"test__{uuid.uuid4().hex}"
    yield name
    httpx.delete(f"{QDRANT_URL}/collections/{name}", timeout=10)


@pytest.fixture
def store(collection: str) -> QdrantVectorStore:
    return QdrantVectorStore(url=QDRANT_URL, collection=collection)


def point(vector: list[float], user: uuid.UUID, document: uuid.UUID) -> VectorPoint:
    return VectorPoint(
        chunk_id=uuid.uuid4(),
        vector=vector,
        user_id=user,
        document_id=document,
        payload={"filename": "a.pdf", "page": 1, "section": None, "chunk_index": 0},
    )


def test_ping(store: QdrantVectorStore) -> None:
    store.ping()


def test_collection_is_created_with_a_tenant_index(
    store: QdrantVectorStore, collection: str
) -> None:
    store.ensure_collection(4)
    store.ensure_collection(4)  # idempotent
    QdrantVectorStore(url=QDRANT_URL, collection=collection).ensure_collection(4)

    info = httpx.get(f"{QDRANT_URL}/collections/{collection}", timeout=10).json()["result"]
    assert info["config"]["params"]["vectors"] == {"size": 4, "distance": "Cosine"}
    schema = info["payload_schema"]
    assert schema["user_id"]["data_type"] == "keyword"
    assert schema["user_id"].get("params", {}).get("is_tenant") is True
    assert schema["document_id"]["data_type"] == "keyword"

    with pytest.raises(VectorStoreError, match="has dimension 4"):
        QdrantVectorStore(url=QDRANT_URL, collection=collection).ensure_collection(8)


def test_search_is_filtered_by_user_inside_qdrant(store: QdrantVectorStore) -> None:
    store.ensure_collection(4)
    alice_close = point([1.0, 0.0, 0.0, 0.0], ALICE, DOC_ALICE_1)
    alice_far = point([0.0, 1.0, 0.0, 0.0], ALICE, DOC_ALICE_2)
    # Bob's vector is the closest to the query: it must still never be returned to Alice.
    bob_exact = point([0.9, 0.1, 0.0, 0.0], BOB, DOC_BOB)
    store.upsert([alice_close, alice_far, bob_exact])
    query = [0.9, 0.1, 0.0, 0.0]

    alice_hits = store.search(query, ALICE, limit=10)
    assert [hit.chunk_id for hit in alice_hits] == [alice_close.chunk_id, alice_far.chunk_id]
    assert alice_hits[0].score > alice_hits[1].score

    assert [hit.chunk_id for hit in store.search(query, BOB, limit=10)] == [bob_exact.chunk_id]
    assert store.search(query, uuid.uuid4(), limit=10) == []

    # Document filter, including a document that belongs to someone else.
    only_second = store.search(query, ALICE, limit=10, document_ids=[DOC_ALICE_2])
    assert [hit.chunk_id for hit in only_second] == [alice_far.chunk_id]
    assert store.search(query, ALICE, limit=10, document_ids=[DOC_BOB]) == []
    assert len(store.search(query, ALICE, limit=1)) == 1


def test_upsert_is_idempotent_and_delete_document_removes_only_that_document(
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(4)
    first = point([1.0, 0.0, 0.0, 0.0], ALICE, DOC_ALICE_1)
    second = point([0.0, 1.0, 0.0, 0.0], ALICE, DOC_ALICE_2)
    store.upsert([first, second])
    store.upsert([first, second])  # same ids: overwritten, not duplicated
    assert len(store.search([1.0, 0.0, 0.0, 0.0], ALICE, limit=10)) == 2

    store.delete_document(DOC_ALICE_1)
    remaining = store.search([1.0, 0.0, 0.0, 0.0], ALICE, limit=10)
    assert [hit.chunk_id for hit in remaining] == [second.chunk_id]
    store.delete_document(DOC_ALICE_1)  # idempotent


def test_missing_collection(store: QdrantVectorStore) -> None:
    assert store.search([1.0, 0.0, 0.0, 0.0], ALICE, limit=5) == []
    store.delete_document(DOC_ALICE_1)  # nothing to delete, no error


def test_unreachable_server() -> None:
    store = QdrantVectorStore(url="http://127.0.0.1:1", collection="x", timeout_seconds=1)
    with pytest.raises(VectorStoreError):
        store.ping()
