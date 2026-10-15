"""Data structures and protocols shared by the retrieval components.

Nothing here depends on FastAPI, SQLAlchemy or a specific vendor: each retrieval
stage can be unit-tested and replaced independently.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A ranked reference to a chunk, as returned by a single retriever."""

    chunk_id: uuid.UUID
    score: float


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """Minimal view of a chunk needed to build a lexical index."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str


@dataclass(slots=True)
class RetrievedChunk:
    """A passage returned to the caller, with the score of every stage."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    text: str
    page: int | None
    section: str | None
    chunk_index: int
    flagged_injection: bool = False
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


@dataclass(slots=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Stages skipped because a dependency was unavailable (e.g. "dense", "rerank").
    degraded: list[str] = field(default_factory=list)
    candidate_count: int = 0


class LexicalRetriever(Protocol):
    """Keyword search restricted to one user (BM25 today, OpenSearch tomorrow)."""

    def search(
        self,
        query: str,
        user_id: uuid.UUID,
        *,
        limit: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[SearchHit]: ...


class VectorRetriever(Protocol):
    """Semantic search restricted to one user."""

    def search(
        self,
        query: str,
        user_id: uuid.UUID,
        *,
        limit: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[SearchHit]: ...


class CorpusSource(Protocol):
    """Where a lexical index reads a user's chunks from."""

    def signature(self, user_id: uuid.UUID) -> str:
        """Cheap value that changes whenever the user's indexed corpus changes."""
        ...

    def load(self, user_id: uuid.UUID) -> list[CorpusEntry]: ...


class ChunkStore(Protocol):
    """Loads full chunks. Implementations MUST filter on ``user_id``."""

    def get_many(
        self, user_id: uuid.UUID, chunk_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, RetrievedChunk]: ...
