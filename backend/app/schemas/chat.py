from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class ChatRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)
    ]
    conversation_id: uuid.UUID | None = Field(
        default=None, description="Continue this conversation (a new one is created otherwise)."
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        max_length=100,
        description="Restrict the answer to these documents (all my documents by default).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What is the notice period to terminate the contract?"}]
        }
    }


class SourceRead(BaseModel):
    """A passage backing the answer. Built from the indexed passage, never from the
    model output: the page number cannot be invented."""

    ref: str = Field(description="Marker used in the answer, e.g. `S1` for `[S1]`.")
    document_id: uuid.UUID
    document: str = Field(description="File name of the document.")
    page: int | None
    section: str | None
    chunk_id: uuid.UUID = Field(
        description="Open it with GET /api/documents/{document_id}/chunks/{chunk_id}."
    )
    excerpt: str
    score: float


class ChatMetadata(BaseModel):
    model: str | None = Field(description="None when no model call was needed.")
    retrieval_ms: float
    llm_ms: float
    total_ms: float
    degraded: list[str] = Field(
        description="Retrieval stages skipped because a dependency was unavailable."
    )


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    answered: bool = Field(
        description="False when the documents do not contain the requested information."
    )
    sources: list[SourceRead]
    metadata: ChatMetadata
