"""Enumerations persisted as VARCHAR + CHECK constraint (no native PostgreSQL ENUM)."""

from __future__ import annotations

from enum import StrEnum

import sqlalchemy as sa


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class JobStatus(StrEnum):
    """Status of asynchronous LLM jobs (document insights, comparisons)."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


def enum_type(enum_cls: type[StrEnum], name: str) -> sa.Enum:
    """VARCHAR(32) column restricted by a named CHECK constraint to the enum values."""
    return sa.Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )
