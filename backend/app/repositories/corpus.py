"""Synchronous, read-only access to a user's indexed chunks for the retrieval engine.

Both classes implement protocols of ``app.rag.types`` and are the **only** place where
the retrieval engine reads chunks. Every query is filtered on the owner twice (chunk
and parent document), and only documents in the ``indexed`` status are visible.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.rag.types import CorpusEntry, RetrievedChunk


class PostgresCorpusSource:
    """Feeds the per-user BM25 indexes."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def signature(self, user_id: uuid.UUID) -> str:
        """Changes whenever a document of the user is indexed, re-indexed or deleted.

        One aggregate over an indexed column: cheap enough to run before every search,
        and it works across processes (the worker indexes, the API searches).
        """
        statement = select(
            func.count(Document.id),
            func.coalesce(func.sum(Document.chunk_count), 0),
            func.max(Document.indexed_at),
        ).where(Document.owner_id == user_id, Document.status == DocumentStatus.INDEXED)
        with self._session_factory() as session:
            count, chunk_total, last_indexed = session.execute(statement).one()
        stamp = last_indexed.isoformat() if last_indexed is not None else "-"
        return f"{count}:{chunk_total}:{stamp}"

    def load(self, user_id: uuid.UUID) -> list[CorpusEntry]:
        statement = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.text,
                DocumentChunk.section,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.owner_id == user_id,
                Document.owner_id == user_id,
                Document.status == DocumentStatus.INDEXED,
            )
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )
        with self._session_factory() as session:
            rows = session.execute(statement).all()
        return [
            CorpusEntry(
                chunk_id=row.id,
                document_id=row.document_id,
                # The section title is searchable too ("termination clause"...).
                text=f"{row.section}\n{row.text}" if row.section else row.text,
            )
            for row in rows
        ]


class PostgresChunkStore:
    """Hydrates search candidates. PostgreSQL is the source of truth: a chunk id that
    is unknown, belongs to another user or to a document being deleted is not returned.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def _select(self, user_id: uuid.UUID) -> Select[Any]:
        """Chunks visible to ``user_id``: the owner filter lives in one place."""
        return (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.text,
                DocumentChunk.page,
                DocumentChunk.section,
                DocumentChunk.chunk_index,
                DocumentChunk.flagged_injection,
                Document.filename,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.owner_id == user_id,
                Document.owner_id == user_id,
                Document.status == DocumentStatus.INDEXED,
            )
        )

    @staticmethod
    def _chunk(row: Any) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            filename=row.filename,
            text=row.text,
            page=row.page,
            section=row.section,
            chunk_index=row.chunk_index,
            flagged_injection=row.flagged_injection,
        )

    def get_many(
        self, user_id: uuid.UUID, chunk_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, RetrievedChunk]:
        if not chunk_ids:
            return {}
        statement = self._select(user_id).where(DocumentChunk.id.in_(list(chunk_ids)))
        with self._session_factory() as session:
            rows = session.execute(statement).all()
        return {row.id: self._chunk(row) for row in rows}

    def list_document(
        self, user_id: uuid.UUID, document_id: uuid.UUID, *, limit: int | None = None
    ) -> list[RetrievedChunk]:
        """Passages of one document of the user, in reading order."""
        statement = (
            self._select(user_id)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        if limit is not None:
            statement = statement.limit(limit)
        with self._session_factory() as session:
            rows = session.execute(statement).all()
        return [self._chunk(row) for row in rows]
