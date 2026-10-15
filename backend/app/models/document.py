from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import DocumentStatus, enum_type
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.chunk import DocumentChunk


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("owner_id", "sha256"),
        Index("ix_documents_owner_id_created_at", "owner_id", "created_at"),
        Index("ix_documents_owner_id_status", "owner_id", "status"),
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(127))
    extension: Mapped[str] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    status: Mapped[DocumentStatus] = mapped_column(
        enum_type(DocumentStatus, "document_status"),
        default=DocumentStatus.UPLOADED,
        server_default=DocumentStatus.UPLOADED.value,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column()
    chunk_count: Mapped[int] = mapped_column(default=0, server_default="0")
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    celery_task_id: Mapped[str | None] = mapped_column(String(255))
    processing_started_at: Mapped[datetime | None] = mapped_column()
    processing_completed_at: Mapped[datetime | None] = mapped_column()
    indexed_at: Mapped[datetime | None] = mapped_column()

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        passive_deletes=True,
        lazy="raise",
        order_by="DocumentChunk.chunk_index",
    )
