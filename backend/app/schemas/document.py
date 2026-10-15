from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import DocumentStatus


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    extension: str
    size_bytes: int
    status: DocumentStatus
    error_message: str | None
    page_count: int | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None


class DocumentStatusRead(BaseModel):
    """Lightweight payload for status polling."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: DocumentStatus
    error_message: str | None
    page_count: int | None
    chunk_count: int
    updated_at: datetime
    indexed_at: datetime | None


class ChunkRead(BaseModel):
    """A passage of a document, e.g. the target of a citation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    text: str
    page: int | None
    section: str | None
    char_start: int
    char_end: int
