from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import MessageRole
from app.schemas.chat import SourceRead


class ConversationRead(BaseModel):
    id: uuid.UUID
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class MessageRead(BaseModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    sources: list[SourceRead] = Field(
        description="Snapshot taken when the answer was generated: it stays readable even "
        "if the document is deleted later."
    )
    answered: bool | None = Field(description="Assistant messages only.")
    created_at: datetime


class ConversationDetail(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageRead]
