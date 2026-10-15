"""Request-scoped logging context (request_id, user_id) based on contextvars."""

from __future__ import annotations

import re
import uuid

import structlog

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def new_request_id() -> str:
    return uuid.uuid4().hex


def sanitize_request_id(value: str | None) -> str:
    """Keep a client-provided request id only if it is short and safe to log."""
    if value and _REQUEST_ID_PATTERN.match(value):
        return value
    return new_request_id()


def bind_request_context(request_id: str) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)


def bind_user_id(user_id: str) -> None:
    structlog.contextvars.bind_contextvars(user_id=user_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()


def get_request_id() -> str | None:
    value = structlog.contextvars.get_contextvars().get("request_id")
    return str(value) if value is not None else None
