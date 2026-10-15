from __future__ import annotations

import uuid

from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.rag.embeddings import HashingEmbedder
from app.rag.vector_store import InMemoryVectorStore
from app.services.indexing_service import ChunkIndexer
from scripts.reindex import reindex_documents
from tests.fixtures.corpus import service_agreement, travel_policy
from tests.integration.conftest import signup, upload


def test_reindex_rebuilds_the_vectors_from_postgresql(
    client: TestClient, session_factory: sessionmaker[Session], db_engine: Engine
) -> None:
    headers = signup(client, "alice@docintel.dev")
    upload(client, headers, "service-agreement.txt", service_agreement())
    policy = upload(client, headers, "travel-policy.txt", travel_policy()).json()["id"]

    # A new embedding model means a new (empty) collection.
    new_store = InMemoryVectorStore()
    indexer = ChunkIndexer(HashingEmbedder(dimension=64), new_store)
    report = reindex_documents(session_factory, indexer)

    assert report.documents == 2
    assert report.chunks == 6 == len(new_store)
    assert report.failed == []
    with db_engine.connect() as connection:
        models = connection.execute(text("SELECT DISTINCT embedding_model FROM documents")).all()
    assert [row.embedding_model for row in models] == ["hashing-64"]

    only_one = InMemoryVectorStore()
    report = reindex_documents(
        session_factory,
        ChunkIndexer(HashingEmbedder(dimension=64), only_one),
        document_id=uuid.UUID(policy),
    )
    assert report.documents == 1
    assert len(only_one) == 3
