"""Conversation history. Every route is scoped to the authenticated user."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response, status

from app.api.deps import ChatServiceDep, CurrentUser
from app.models.conversation import Message
from app.schemas.chat import SourceRead
from app.schemas.common import ERROR_RESPONSES, ErrorResponse, Page
from app.schemas.conversation import ConversationDetail, ConversationRead, MessageRead

router = APIRouter(prefix="/conversations", tags=["conversations"])

NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Conversation not found"}
}


def _message(message: Message) -> MessageRead:
    answered = message.meta.get("answered")
    return MessageRead(
        id=message.id,
        role=message.role,
        content=message.content,
        sources=[SourceRead.model_validate(source) for source in message.citations],
        answered=answered if isinstance(answered, bool) else None,
        created_at=message.created_at,
    )


@router.get(
    "",
    response_model=Page[ConversationRead],
    summary="List my conversations",
    description="Most recently active first.",
    responses=ERROR_RESPONSES,
)
async def list_conversations(
    current_user: CurrentUser,
    service: ChatServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[ConversationRead]:
    rows, total = await service.list_conversations(current_user, limit=limit, offset=offset)
    return Page[ConversationRead](
        items=[
            ConversationRead(
                id=conversation.id,
                title=conversation.title,
                message_count=message_count,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
            for conversation, message_count in rows
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get a conversation with its messages",
    responses={**NOT_FOUND, **ERROR_RESPONSES},
)
async def get_conversation(
    conversation_id: uuid.UUID, current_user: CurrentUser, service: ChatServiceDep
) -> ConversationDetail:
    conversation, messages = await service.get_conversation(current_user, conversation_id)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[_message(message) for message in messages],
    )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
    responses={**NOT_FOUND, **ERROR_RESPONSES},
)
async def delete_conversation(
    conversation_id: uuid.UUID, current_user: CurrentUser, service: ChatServiceDep
) -> Response:
    await service.delete_conversation(current_user, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
