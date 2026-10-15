from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.vector_store import InMemoryVectorStore, VectorPoint, VectorStoreError
from app.services.indexing_service import (
    ChunkIndexer,
    IndexableChunk,
    IndexingError,
    embedding_text,
)

OWNER = uuid.UUID(int=1)
DOCUMENT = uuid.UUID(int=10)


def chunks(count: int) -> list[IndexableChunk]:
    return [
        IndexableChunk(
            chunk_id=uuid.UUID(int=100 + i),
            chunk_index=i,
            text=f"passage number {i} about termination",
            page=i + 1,
            section="Termination" if i == 0 else None,
        )
        for i in range(count)
    ]


class RecordingStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.dimension: int | None = None
        self.upserted: list[VectorPoint] = []

    def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension
        super().ensure_collection(dimension)

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        self.upserted.extend(points)
        super().upsert(points)


def test_embedding_text_adds_document_and_section_context() -> None:
    first, second = chunks(2)
    assert embedding_text("contract.pdf", first) == (
        "Document: contract.pdf\nSection: Termination\n\npassage number 0 about termination"
    )
    assert embedding_text("contract.pdf", second) == (
        "Document: contract.pdf\n\npassage number 1 about termination"
    )


def test_index_document_writes_one_point_per_chunk() -> None:
    store = RecordingStore()
    embedder = HashingEmbedder(dimension=32)
    indexer = ChunkIndexer(embedder, store)

    count = indexer.index_document(
        document_id=DOCUMENT, owner_id=OWNER, filename="contract.pdf", chunks=chunks(3)
    )

    assert count == 3
    assert store.dimension == 32
    assert indexer.model_name == "hashing-32"
    first = store.upserted[0]
    assert first.chunk_id == uuid.UUID(int=100)  # the point id is the chunk id
    assert (first.user_id, first.document_id) == (OWNER, DOCUMENT)
    assert first.payload == {
        "filename": "contract.pdf",
        "page": 1,
        "section": "Termination",
        "chunk_index": 0,
    }
    assert len(first.vector) == 32
    assert (
        list(first.vector)
        == embedder.embed_documents([embedding_text("contract.pdf", chunks(1)[0])])[0]
    )


def test_reindexing_replaces_the_previous_vectors() -> None:
    store = InMemoryVectorStore()
    indexer = ChunkIndexer(HashingEmbedder(dimension=32), store)
    indexer.index_document(document_id=DOCUMENT, owner_id=OWNER, filename="a.pdf", chunks=chunks(5))
    indexer.index_document(document_id=DOCUMENT, owner_id=OWNER, filename="a.pdf", chunks=chunks(2))
    assert len(store) == 2


def test_delete_document() -> None:
    store = InMemoryVectorStore()
    indexer = ChunkIndexer(HashingEmbedder(dimension=32), store)
    indexer.index_document(document_id=DOCUMENT, owner_id=OWNER, filename="a.pdf", chunks=chunks(2))
    indexer.delete_document(DOCUMENT)
    assert len(store) == 0


def test_embedding_failure_leaves_the_existing_vectors_untouched() -> None:
    store = InMemoryVectorStore()
    ChunkIndexer(HashingEmbedder(dimension=32), store).index_document(
        document_id=DOCUMENT, owner_id=OWNER, filename="a.pdf", chunks=chunks(2)
    )

    class BrokenEmbedder(HashingEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            raise EmbeddingError("embeddings: ConnectError")

    with pytest.raises(IndexingError, match="ConnectError"):
        ChunkIndexer(BrokenEmbedder(dimension=32), store).index_document(
            document_id=DOCUMENT, owner_id=OWNER, filename="a.pdf", chunks=chunks(2)
        )
    assert len(store) == 2


def test_vector_store_failures_are_wrapped() -> None:
    class BrokenStore(InMemoryVectorStore):
        def upsert(self, points: Sequence[VectorPoint]) -> None:
            raise VectorStoreError("qdrant: HTTP 500")

        def delete_document(self, document_id: uuid.UUID) -> None:
            raise VectorStoreError("qdrant: HTTP 500")

    indexer = ChunkIndexer(HashingEmbedder(dimension=32), BrokenStore())
    with pytest.raises(IndexingError, match="HTTP 500"):
        indexer.index_document(
            document_id=DOCUMENT, owner_id=OWNER, filename="a.pdf", chunks=chunks(1)
        )
    with pytest.raises(IndexingError, match="HTTP 500"):
        indexer.delete_document(DOCUMENT)
