"""Vector indexing of a document's chunks (embeddings + vector store).

Used by the ingestion job and by ``scripts/reindex.py``. Indexing a document is
idempotent: the vectors of the document are replaced as a whole, and point ids are the
(deterministic) chunk ids.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from app.rag.embeddings import EmbeddingError, EmbeddingProvider
from app.rag.vector_store import VectorPoint, VectorStore, VectorStoreError

logger = structlog.get_logger(__name__)


class IndexingError(Exception):
    """The embedding server or the vector database is unavailable."""


@dataclass(frozen=True, slots=True)
class IndexableChunk:
    chunk_id: uuid.UUID
    chunk_index: int
    text: str
    page: int | None
    section: str | None


def embedding_text(filename: str, chunk: IndexableChunk) -> str:
    """Text sent to the embedding model.

    A short contextual header (document name, section title) is prepended: a passage
    such as "It ends after 12 months." becomes retrievable for "contract duration".
    The stored chunk text itself is unchanged, so citations stay exact.
    """
    header = f"Document: {filename}"
    if chunk.section:
        header += f"\nSection: {chunk.section}"
    return f"{header}\n\n{chunk.text}"


class ChunkIndexer:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    def index_document(
        self,
        *,
        document_id: uuid.UUID,
        owner_id: uuid.UUID,
        filename: str,
        chunks: Sequence[IndexableChunk],
    ) -> int:
        """Embed ``chunks`` and replace the vectors of the document. Returns the count."""
        try:
            vectors = self._embedder.embed_documents(
                [embedding_text(filename, chunk) for chunk in chunks]
            )
            self._vector_store.ensure_collection(self._embedder.dimension)
            # Removes leftovers of a previous run that produced more chunks.
            self._vector_store.delete_document(document_id)
            self._vector_store.upsert(
                [
                    VectorPoint(
                        chunk_id=chunk.chunk_id,
                        vector=vector,
                        user_id=owner_id,
                        document_id=document_id,
                        payload={
                            "filename": filename,
                            "page": chunk.page,
                            "section": chunk.section,
                            "chunk_index": chunk.chunk_index,
                        },
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
            )
        except (EmbeddingError, VectorStoreError) as exc:
            raise IndexingError(str(exc)) from exc
        return len(chunks)

    def delete_document(self, document_id: uuid.UUID) -> None:
        try:
            self._vector_store.delete_document(document_id)
        except VectorStoreError as exc:
            raise IndexingError(str(exc)) from exc
