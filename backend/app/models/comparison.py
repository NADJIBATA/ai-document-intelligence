from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import JobStatus, enum_type
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class DocumentComparison(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "document_comparisons"
    __table_args__ = (
        Index("ix_document_comparisons_owner_id_created_at", "owner_id", "created_at"),
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # SET NULL: the comparison result stays in the history after a document is deleted.
    document_a_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    document_b_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    document_a_name: Mapped[str] = mapped_column(String(255))
    document_b_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[JobStatus] = mapped_column(
        enum_type(JobStatus, "comparison_status"),
        default=JobStatus.PENDING,
        server_default=JobStatus.PENDING.value,
    )
    aspects: Mapped[list[str]] = mapped_column(default=list, server_default=text("'[]'::jsonb"))
    result: Mapped[dict[str, Any] | None] = mapped_column()
    error_message: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column()
