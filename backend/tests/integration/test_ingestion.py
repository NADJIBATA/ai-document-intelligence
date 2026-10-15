"""IngestionService against a real PostgreSQL (the job executed by the Celery worker)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.user import User
from app.rag.embeddings import EmbeddingError, EmbeddingProvider, HashingEmbedder
from app.rag.vector_store import InMemoryVectorStore
from app.services.indexing_service import ChunkIndexer
from app.services.ingestion_service import (
    INDEXING_ERROR_MESSAGE,
    IngestionOutcome,
    IngestionService,
    TransientIngestionError,
    build_document_processor,
    chunk_id_for,
)
from tests.conftest import build_settings
from tests.fakes import InMemoryStorage
from tests.fixtures.builders import contract_pdf


@pytest.fixture
def stored_document(session_factory: sessionmaker[Session], storage: InMemoryStorage) -> Document:
    with session_factory() as session:
        user = User(email="worker@docintel.dev", hashed_password="x")
        session.add(user)
        session.flush()
        document_id = uuid.uuid4()
        key = f"users/{user.id}/documents/{document_id}/original.pdf"
        content = contract_pdf()
        storage.objects[key] = (content, "application/pdf")
        document = Document(
            id=document_id,
            owner_id=user.id,
            filename="contract.pdf",
            content_type="application/pdf",
            extension="pdf",
            size_bytes=len(content),
            sha256="0" * 64,
            storage_key=key,
        )
        session.add(document)
        session.commit()
        return document


def _document_row(db_engine: Engine, document_id: uuid.UUID) -> Any:
    with db_engine.connect() as connection:
        return connection.execute(
            text(
                "SELECT status, error_message, chunk_count, page_count FROM documents "
                "WHERE id = :id"
            ),
            {"id": document_id},
        ).one()


def test_successful_ingestion(
    ingestion_service: IngestionService, stored_document: Document, db_engine: Engine
) -> None:
    result = ingestion_service.process(stored_document.id)
    assert result.outcome == IngestionOutcome.INDEXED
    row = _document_row(db_engine, stored_document.id)
    assert row.status == "indexed"
    assert row.page_count == 3
    assert row.chunk_count == result.chunk_count > 0

    with db_engine.connect() as connection:
        ids = connection.execute(
            text("SELECT id, chunk_index FROM document_chunks ORDER BY chunk_index")
        ).all()
    assert all(row.id == chunk_id_for(stored_document.id, row.chunk_index) for row in ids)


def test_ingestion_is_idempotent(
    ingestion_service: IngestionService,
    stored_document: Document,
    db_engine: Engine,
    session_factory: sessionmaker[Session],
) -> None:
    first = ingestion_service.process(stored_document.id)
    # Already indexed: a redelivered job is a no-op.
    assert ingestion_service.process(stored_document.id).outcome == IngestionOutcome.SKIPPED
    # Forced re-run (e.g. after a failure) replaces the chunks instead of duplicating them.
    with session_factory() as session:
        session.get_one(Document, stored_document.id).status = DocumentStatus.FAILED
        session.commit()
    second = ingestion_service.process(stored_document.id)
    assert second.chunk_count == first.chunk_count
    with db_engine.connect() as connection:
        count = connection.execute(text("SELECT count(*) FROM document_chunks")).scalar()
    assert count == first.chunk_count


def test_deleted_document_is_skipped(
    ingestion_service: IngestionService, stored_document: Document, db_engine: Engine
) -> None:
    with db_engine.begin() as connection:
        connection.execute(text("DELETE FROM documents"))
    assert ingestion_service.process(stored_document.id).outcome == IngestionOutcome.SKIPPED


def test_missing_file_fails_permanently(
    ingestion_service: IngestionService,
    stored_document: Document,
    storage: InMemoryStorage,
    db_engine: Engine,
) -> None:
    storage.objects.clear()
    result = ingestion_service.process(stored_document.id)
    assert result.outcome == IngestionOutcome.FAILED
    assert _document_row(db_engine, stored_document.id).status == "failed"


def test_storage_outage_is_retried_then_fails(
    ingestion_service: IngestionService,
    stored_document: Document,
    storage: InMemoryStorage,
    db_engine: Engine,
) -> None:
    storage.available = False
    with pytest.raises(TransientIngestionError):
        ingestion_service.process(stored_document.id, final_attempt=False)
    assert _document_row(db_engine, stored_document.id).status == "processing"

    result = ingestion_service.process(stored_document.id, final_attempt=True)
    assert result.outcome == IngestionOutcome.FAILED
    row = _document_row(db_engine, stored_document.id)
    assert row.status == "failed"
    assert row.error_message == "The file could not be read from storage."

    storage.available = True
    assert ingestion_service.process(stored_document.id).outcome == IngestionOutcome.INDEXED


# ------------------------------------------------------------------ vector index
class FlakyEmbedder(HashingEmbedder):
    """Embedding server that can be switched off."""

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension=dimension)
        self.available = True

    def embed_documents(self, texts: Any) -> list[list[float]]:
        if not self.available:
            raise EmbeddingError("embeddings: ConnectError")
        return super().embed_documents(texts)


def test_ingestion_writes_one_vector_per_chunk(
    ingestion_service: IngestionService,
    stored_document: Document,
    db_engine: Engine,
    vector_store: InMemoryVectorStore,
    embedder: EmbeddingProvider,
) -> None:
    result = ingestion_service.process(stored_document.id)
    assert len(vector_store) == result.chunk_count

    # Every vector belongs to the owner: the store finds them for him, and only for him.
    query = embedder.embed_query("contract")
    hits = vector_store.search(query, stored_document.owner_id, limit=1000)
    assert {hit.chunk_id for hit in hits} == {
        chunk_id_for(stored_document.id, index) for index in range(result.chunk_count)
    }
    assert vector_store.search(query, uuid.uuid4(), limit=1000) == []

    with db_engine.connect() as connection:
        model = connection.execute(
            text("SELECT embedding_model FROM documents WHERE id = :id"),
            {"id": stored_document.id},
        ).scalar_one()
    assert model == embedder.model_name


def test_reprocessing_does_not_duplicate_vectors(
    ingestion_service: IngestionService,
    stored_document: Document,
    session_factory: sessionmaker[Session],
    vector_store: InMemoryVectorStore,
) -> None:
    first = ingestion_service.process(stored_document.id)
    with session_factory() as session:
        session.get_one(Document, stored_document.id).status = DocumentStatus.FAILED
        session.commit()
    ingestion_service.process(stored_document.id)
    assert len(vector_store) == first.chunk_count


def test_embedding_outage_is_retried_then_fails(
    session_factory: sessionmaker[Session],
    storage: InMemoryStorage,
    stored_document: Document,
    db_engine: Engine,
    vector_store: InMemoryVectorStore,
) -> None:
    settings = build_settings()
    embedder = FlakyEmbedder(settings.embedding_dimension)
    service = IngestionService(
        session_factory,
        storage,
        build_document_processor(settings),
        ChunkIndexer(embedder, vector_store),
    )

    embedder.available = False
    with pytest.raises(TransientIngestionError):
        service.process(stored_document.id, final_attempt=False)
    assert _document_row(db_engine, stored_document.id).status == "processing"

    result = service.process(stored_document.id, final_attempt=True)
    assert result.outcome == IngestionOutcome.FAILED
    row = _document_row(db_engine, stored_document.id)
    assert row.status == "failed"
    assert row.error_message == INDEXING_ERROR_MESSAGE
    # A document is never visible to the search without its vectors.
    assert row.chunk_count == 0
    assert len(vector_store) == 0
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM document_chunks")).scalar() == 0

    embedder.available = True
    recovered = service.process(stored_document.id)
    assert recovered.outcome == IngestionOutcome.INDEXED
    assert len(vector_store) == recovered.chunk_count


def test_vectors_of_a_document_deleted_meanwhile_are_removed(
    session_factory: sessionmaker[Session],
    storage: InMemoryStorage,
    stored_document: Document,
    db_engine: Engine,
    vector_store: InMemoryVectorStore,
) -> None:
    settings = build_settings()

    class DeletingEmbedder(HashingEmbedder):
        """Simulates a user deleting the document while it is being embedded."""

        def embed_documents(self, texts: Any) -> list[list[float]]:
            with db_engine.begin() as connection:
                connection.execute(text("DELETE FROM documents"))
            return super().embed_documents(texts)

    service = IngestionService(
        session_factory,
        storage,
        build_document_processor(settings),
        ChunkIndexer(DeletingEmbedder(settings.embedding_dimension), vector_store),
    )
    result = service.process(stored_document.id)
    assert result.outcome == IngestionOutcome.SKIPPED
    assert len(vector_store) == 0
