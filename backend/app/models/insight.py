from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import JobStatus, enum_type
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class DocumentInsight(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Structured information extracted from a document by the LLM (one per document)."""

    __tablename__ = "document_insights"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), unique=True
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[JobStatus] = mapped_column(
        enum_type(JobStatus, "insight_status"),
        default=JobStatus.PENDING,
        server_default=JobStatus.PENDING.value,
    )
    document_type: Mapped[str | None] = mapped_column(String(50))
    title: Mapped[str | None] = mapped_column(String(1024))
    language: Mapped[str | None] = mapped_column(String(16))
    summary: Mapped[str | None] = mapped_column(Text)
    data: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))
    model: Mapped[str | None] = mapped_column(String(255))
    error_message: Mapped[str | None] = mapped_column(Text)
