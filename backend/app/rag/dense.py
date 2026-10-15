"""Dense (semantic) retrieval: embed the query, search the vector store."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from app.rag.embeddings import EmbeddingProvider
from app.rag.types import SearchHit
from app.rag.vector_store import VectorStore


class DenseRetriever:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        user_id: uuid.UUID,
        *,
        limit: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[SearchHit]:
        """``user_id`` is mandatory: there is no way to run an unscoped search."""
        vector = self._embedder.embed_query(query)
        return self._vector_store.search(vector, user_id, limit=limit, document_ids=document_ids)
