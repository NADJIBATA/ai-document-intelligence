from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(UTC)


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)


class CreatedAtMixin:
    created_at: Mapped[datetime] = mapped_column(default=utc_now, server_default=func.now())


class TimestampMixin(CreatedAtMixin):
    updated_at: Mapped[datetime] = mapped_column(
        default=utc_now, onupdate=utc_now, server_default=func.now()
    )
