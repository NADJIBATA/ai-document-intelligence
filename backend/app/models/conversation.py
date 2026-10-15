from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import MessageRole, enum_type
from app.models.mixins import CreatedAtMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Conversation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_owner_id_updated_at", "owner_id", "updated_at"),)

    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(255))

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        passive_deletes=True,
        lazy="raise",
        order_by="Message.created_at",
    )


class Message(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[MessageRole] = mapped_column(enum_type(MessageRole, "message_role"))
    content: Mapped[str] = mapped_column(Text)
    # Snapshot of the sources shown to the user (filename, page, excerpt, ...):
    # the history stays readable even if the document is deleted later.
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        default=list, server_default=text("'[]'::jsonb")
    )
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", default=dict, server_default=text("'{}'::jsonb")
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages", lazy="raise")
