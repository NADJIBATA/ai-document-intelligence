"""The search keeps answering when a model server or the vector database is down."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.rag.embeddings import EmbeddingError, EmbeddingProvider, HashingEmbedder
from app.rag.factory import build_retrieval
from app.rag.reranker import RerankerError
from app.rag.types import RetrievedChunk
from app.rag.vector_store import InMemoryVectorStore
from tests.fixtures.corpus import service_agreement
from tests.integration.conftest import signup, upload


class DownEmbedder(HashingEmbedder):
    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingError("embeddings: ConnectError")


class DownReranker:
    model_name = "down"

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        raise RerankerError("reranker: ConnectError")


def test_search_degrades_to_bm25_when_the_model_servers_are_down(
    make_client: Any,
    integration_settings: Settings,
    session_factory: sessionmaker[Session],
    embedder: EmbeddingProvider,
    vector_store: InMemoryVectorStore,
) -> None:
    healthy: TestClient = make_client()
    headers = signup(healthy, "alice@docintel.dev")
    upload(healthy, headers, "service-agreement.txt", service_agreement())

    broken = build_retrieval(
        integration_settings,
        session_factory,
        embedder=DownEmbedder(dimension=embedder.dimension),
        vector_store=vector_store,
        reranker=DownReranker(),
    )
    try:
        degraded: TestClient = make_client(retrieval=broken)
        response = degraded.post(
            "/api/search", headers=headers, json={"query": "notice period thirty days"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["degraded"] == ["dense", "rerank"]
        best = body["results"][0]
        assert "notice period of thirty days" in best["text"]
        assert best["bm25_rank"] == 1
        assert best["dense_rank"] is None
        assert best["rerank_score"] is None
        assert best["score"] == best["rrf_score"]
    finally:
        broken.close()
