from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import ForeignKey, Index, String, Text, UniqueConstraint, false
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import CreatedAtMixin

if TYPE_CHECKING:
    from app.models.document import Document


class DocumentChunk(CreatedAtMixin, Base):
    """A passage of a document. ``id`` is also the Qdrant point id.

    The id is deterministic (uuid5 of ``document_id`` + ``chunk_index``) and is set by
    the ingestion pipeline, which makes re-indexing idempotent.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        Index("ix_document_chunks_owner_id_document_id", "owner_id", "document_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    # Denormalised owner: every lexical/dense lookup is filtered on it without a join.
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column()
    # NB: attribute named "text" -> use sa.text() below, not a bare text() import.
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column()
    section: Mapped[str | None] = mapped_column(String(512))
    char_start: Mapped[int] = mapped_column()
    char_end: Mapped[int] = mapped_column()
    flagged_injection: Mapped[bool] = mapped_column(default=False, server_default=false())
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", default=dict, server_default=sa.text("'{}'::jsonb")
    )

    document: Mapped[Document] = relationship(back_populates="chunks", lazy="raise")
