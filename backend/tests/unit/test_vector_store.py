"""Vector stores. The Qdrant client is tested against a recorded REST conversation:
what matters is *what is sent* — above all the mandatory ``user_id`` filter."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest

from app.rag.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorPoint,
    VectorStoreError,
    build_filter,
    collection_name,
)

ALICE = uuid.UUID(int=1)
BOB = uuid.UUID(int=2)
DOC_1 = uuid.UUID(int=101)
DOC_2 = uuid.UUID(int=102)
COLLECTION = "chunks__qwen-qwen3-embedding-0-6b__4"


class FakeQdrant:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.collection: dict[str, Any] | None = None
        self.search_points: list[dict[str, Any]] = []
        self.fail_with: int | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.calls.append((request.method, path, body))
        if self.fail_with is not None:
            return httpx.Response(self.fail_with, text="qdrant error")
        if path == "/readyz":
            return httpx.Response(200, text="all shards are ready")
        if path == f"/collections/{COLLECTION}":
            if request.method == "GET":
                if self.collection is None:
                    return httpx.Response(404, json={"status": {"error": "Not found"}})
                return httpx.Response(
                    200, json={"result": {"config": {"params": {"vectors": self.collection}}}}
                )
            self.collection = body["vectors"]
            return httpx.Response(200, json={"result": True})
        if path.endswith("/points/query"):
            if self.collection is None:
                return httpx.Response(404, json={"status": {"error": "Not found"}})
            return httpx.Response(200, json={"result": {"points": self.search_points}})
        return httpx.Response(200, json={"result": {"status": "completed"}})

    def bodies(self, method: str, suffix: str) -> list[Any]:
        return [body for m, path, body in self.calls if m == method and path.endswith(suffix)]


@pytest.fixture
def qdrant() -> FakeQdrant:
    return FakeQdrant()


@pytest.fixture
def store(qdrant: FakeQdrant) -> QdrantVectorStore:
    client = httpx.Client(transport=httpx.MockTransport(qdrant), base_url="http://qdrant:6333")
    return QdrantVectorStore(url="http://qdrant:6333", collection=COLLECTION, client=client)


def point(chunk: int, user: uuid.UUID = ALICE, document: uuid.UUID = DOC_1) -> VectorPoint:
    return VectorPoint(
        chunk_id=uuid.UUID(int=1000 + chunk),
        vector=[1.0, 0.0, 0.0, 0.0],
        user_id=user,
        document_id=document,
        payload={"filename": "contract.pdf", "page": 1},
    )


def test_collection_name_isolates_models_and_dimensions() -> None:
    assert collection_name("chunks", "Qwen/Qwen3-Embedding-0.6B", 4) == COLLECTION
    assert collection_name("chunks", "text-embedding-3-small", 1536) == (
        "chunks__text-embedding-3-small__1536"
    )


def test_filter_always_contains_the_user() -> None:
    assert build_filter(ALICE) == {"must": [{"key": "user_id", "match": {"value": str(ALICE)}}]}
    assert build_filter(ALICE, [DOC_1, DOC_2])["must"] == [
        {"key": "user_id", "match": {"value": str(ALICE)}},
        {"key": "document_id", "match": {"any": [str(DOC_1), str(DOC_2)]}},
    ]


def test_ensure_collection_creates_it_with_a_tenant_index(
    store: QdrantVectorStore, qdrant: FakeQdrant
) -> None:
    store.ensure_collection(4)
    assert qdrant.collection == {"size": 4, "distance": "Cosine"}
    indexes = qdrant.bodies("PUT", "/index")
    assert {"field_name": "user_id", "field_schema": {"type": "keyword", "is_tenant": True}} in (
        indexes
    )
    assert {"field_name": "document_id", "field_schema": "keyword"} in indexes

    calls_before = len(qdrant.calls)
    store.ensure_collection(4)  # idempotent and cached
    assert len(qdrant.calls) == calls_before


def test_concurrent_collection_creation_is_tolerated() -> None:
    """Two workers start together: the second PUT fails, but the collection exists."""
    state = {"created": False}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/collections/{COLLECTION}" and request.method == "GET":
            if not state["created"]:
                state["created"] = True  # ...by the other worker, right after this check
                return httpx.Response(404, json={})
            return httpx.Response(200, json={"result": {}})
        if path == f"/collections/{COLLECTION}":
            return httpx.Response(409, json={"status": {"error": "already exists"}})
        return httpx.Response(200, json={"result": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://qdrant:6333")
    QdrantVectorStore(
        url="http://qdrant:6333", collection=COLLECTION, client=client
    ).ensure_collection(4)


def test_ensure_collection_rejects_another_dimension(
    store: QdrantVectorStore, qdrant: FakeQdrant
) -> None:
    qdrant.collection = {"size": 1024, "distance": "Cosine"}
    with pytest.raises(VectorStoreError, match="has dimension 1024"):
        store.ensure_collection(4)


def test_upsert_writes_the_tenant_keys_last(store: QdrantVectorStore, qdrant: FakeQdrant) -> None:
    forged = VectorPoint(
        chunk_id=uuid.UUID(int=1001),
        vector=[1.0, 0.0, 0.0, 0.0],
        user_id=ALICE,
        document_id=DOC_1,
        payload={"user_id": str(BOB), "document_id": "other", "filename": "contract.pdf"},
    )
    store.upsert([forged])
    sent = qdrant.bodies("PUT", "/points")[0]["points"][0]
    assert sent["id"] == str(uuid.UUID(int=1001))
    assert sent["payload"]["user_id"] == str(ALICE)  # the payload cannot override the owner
    assert sent["payload"]["document_id"] == str(DOC_1)
    assert sent["payload"]["chunk_id"] == sent["id"]
    assert sent["payload"]["filename"] == "contract.pdf"


def test_upsert_is_batched(store: QdrantVectorStore, qdrant: FakeQdrant) -> None:
    store.upsert([point(i) for i in range(300)])
    assert [len(body["points"]) for body in qdrant.bodies("PUT", "/points")] == [128, 128, 44]
    store.upsert([])
    assert len(qdrant.bodies("PUT", "/points")) == 3


def test_search_always_sends_the_user_filter(store: QdrantVectorStore, qdrant: FakeQdrant) -> None:
    qdrant.collection = {"size": 4, "distance": "Cosine"}
    qdrant.search_points = [
        {"id": str(uuid.UUID(int=1002)), "score": 0.91},
        {"id": str(uuid.UUID(int=1001)), "score": 0.42},
    ]
    hits = store.search([0.1, 0.2, 0.3, 0.4], ALICE, limit=30)
    assert [(hit.chunk_id.int, hit.score) for hit in hits] == [(1002, 0.91), (1001, 0.42)]

    body = qdrant.bodies("POST", "/points/query")[0]
    assert body["filter"] == {"must": [{"key": "user_id", "match": {"value": str(ALICE)}}]}
    assert body["limit"] == 30
    assert body["query"] == [0.1, 0.2, 0.3, 0.4]

    store.search([0.1, 0.2, 0.3, 0.4], BOB, limit=5, document_ids=[DOC_2])
    assert qdrant.bodies("POST", "/points/query")[1]["filter"]["must"] == [
        {"key": "user_id", "match": {"value": str(BOB)}},
        {"key": "document_id", "match": {"any": [str(DOC_2)]}},
    ]


def test_search_with_an_empty_document_list_returns_nothing(
    store: QdrantVectorStore, qdrant: FakeQdrant
) -> None:
    assert store.search([1.0, 0.0, 0.0, 0.0], ALICE, limit=5, document_ids=[]) == []
    assert qdrant.calls == []


def test_search_before_any_indexing_returns_nothing(store: QdrantVectorStore) -> None:
    assert store.search([1.0, 0.0, 0.0, 0.0], ALICE, limit=5) == []  # 404: no collection yet


def test_delete_document(store: QdrantVectorStore, qdrant: FakeQdrant) -> None:
    store.delete_document(DOC_1)
    assert qdrant.bodies("POST", "/points/delete")[0] == {
        "filter": {"must": [{"key": "document_id", "match": {"value": str(DOC_1)}}]}
    }


def test_errors_are_wrapped(store: QdrantVectorStore, qdrant: FakeQdrant) -> None:
    qdrant.fail_with = 500
    with pytest.raises(VectorStoreError, match="HTTP 500"):
        store.search([1.0, 0.0, 0.0, 0.0], ALICE, limit=5)
    with pytest.raises(VectorStoreError):
        store.upsert([point(1)])
    with pytest.raises(VectorStoreError):
        store.delete_document(DOC_1)
    with pytest.raises(VectorStoreError):
        store.ping()
    with pytest.raises(VectorStoreError):
        store.ensure_collection(4)


def test_network_errors_are_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://qdrant:6333")
    store = QdrantVectorStore(url="http://qdrant:6333", collection=COLLECTION, client=client)
    with pytest.raises(VectorStoreError, match="ConnectError"):
        store.ping()


def test_ping_and_api_key(monkeypatch: pytest.MonkeyPatch, qdrant: FakeQdrant) -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return qdrant(request)

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    store = QdrantVectorStore(url="http://qdrant:6333/", collection=COLLECTION, api_key="k-test")
    store.ping()
    assert seen[0]["api-key"] == "k-test"
    assert qdrant.calls[0][:2] == ("GET", "/readyz")


# ----------------------------------------------------------- InMemoryVectorStore
def test_in_memory_search_is_scoped_to_the_user() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(4)
    store.upsert([point(1, ALICE, DOC_1), point(2, BOB, DOC_2)])

    assert [hit.chunk_id.int for hit in store.search([1.0, 0, 0, 0], ALICE, limit=10)] == [1001]
    assert [hit.chunk_id.int for hit in store.search([1.0, 0, 0, 0], BOB, limit=10)] == [1002]
    # Asking for someone else's document id does not help.
    assert store.search([1.0, 0, 0, 0], ALICE, limit=10, document_ids=[DOC_2]) == []
    assert store.search([1.0, 0, 0, 0], uuid.UUID(int=3), limit=10) == []


def test_in_memory_ranking_limit_and_delete() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorPoint(uuid.UUID(int=1), [1.0, 0.0], ALICE, DOC_1, {}),
            VectorPoint(uuid.UUID(int=2), [0.6, 0.8], ALICE, DOC_1, {}),
            VectorPoint(uuid.UUID(int=3), [0.0, 1.0], ALICE, DOC_2, {}),
        ]
    )
    hits = store.search([1.0, 0.0], ALICE, limit=2)
    assert [hit.chunk_id.int for hit in hits] == [1, 2]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.6)

    store.delete_document(DOC_1)
    assert len(store) == 1
    assert [hit.chunk_id.int for hit in store.search([1.0, 0.0], ALICE, limit=5)] == [3]


def test_in_memory_dimension_is_checked() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(4)
    store.ensure_collection(4)
    with pytest.raises(VectorStoreError, match="dimension"):
        store.ensure_collection(8)
