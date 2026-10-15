"""Conversation persistence. Every query is scoped by ``owner_id``."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation, Message


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, conversation: Conversation) -> Conversation:
        self._session.add(conversation)
        return conversation

    def add_message(self, message: Message) -> Message:
        self._session.add(message)
        return message

    async def get_owned(
        self, owner_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation | None:
        result = await self._session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.owner_id == owner_id
            )
        )
        return result.scalar_one_or_none()

    async def list_owned(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[tuple[Conversation, int]], int]:
        """Most recently active first, with the number of messages of each conversation."""
        total = await self._session.scalar(
            select(func.count()).select_from(Conversation).where(Conversation.owner_id == owner_id)
        )
        message_count = (
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == Conversation.id)
            .correlate(Conversation)
            .scalar_subquery()
        )
        result = await self._session.execute(
            select(Conversation, message_count)
            .where(Conversation.owner_id == owner_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = [(conversation, int(count or 0)) for conversation, count in result.all()]
        return rows, int(total or 0)

    async def list_messages(
        self, owner_id: uuid.UUID, conversation_id: uuid.UUID, *, last: int | None = None
    ) -> list[Message]:
        """Messages in chronological order; ``last`` keeps only the most recent ones.

        The join on the conversation owner makes the method safe on its own.
        """
        statement = (
            select(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(Message.conversation_id == conversation_id, Conversation.owner_id == owner_id)
        )
        if last is None:
            result = await self._session.execute(statement.order_by(Message.created_at, Message.id))
            return list(result.scalars().all())
        if last <= 0:
            return []
        result = await self._session.execute(
            statement.order_by(Message.created_at.desc(), Message.id.desc()).limit(last)
        )
        return list(reversed(result.scalars().all()))

    async def delete_owned(self, owner_id: uuid.UUID, conversation_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            delete(Conversation).where(
                Conversation.id == conversation_id, Conversation.owner_id == owner_id
            )
        )
        return isinstance(result, CursorResult) and result.rowcount > 0  # messages cascade
