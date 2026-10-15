from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin


class RefreshToken(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Server-side record of an issued refresh token.

    Only the SHA-256 of the token is stored. Tokens issued from the same login share a
    ``family_id``: presenting an already-rotated token revokes the whole family.
    """

    __tablename__ = "refresh_tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    family_id: Mapped[uuid.UUID] = mapped_column(index=True)
    expires_at: Mapped[datetime] = mapped_column()
    revoked_at: Mapped[datetime | None] = mapped_column()
    replaced_by_id: Mapped[uuid.UUID | None] = mapped_column()
    user_agent: Mapped[str | None] = mapped_column(String(512))
    ip_address: Mapped[str | None] = mapped_column(String(64))
