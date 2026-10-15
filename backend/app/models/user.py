from __future__ import annotations

from datetime import datetime

from sqlalchemy import String, true
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import TimestampMixin, UUIDPrimaryKeyMixin


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    # Always stored lower-cased (normalised by the API schema) -> plain unique constraint.
    email: Mapped[str] = mapped_column(String(320), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    last_login_at: Mapped[datetime | None] = mapped_column()

    def __repr__(self) -> str:  # never expose the password hash
        return f"User(id={self.id!s}, email={self.email!r})"
