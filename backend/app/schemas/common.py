from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ErrorItem(BaseModel):
    loc: list[str]
    msg: str
    type: str


class ErrorContent(BaseModel):
    code: str = Field(examples=["not_found"])
    message: str = Field(examples=["The requested resource was not found."])
    request_id: str | None = Field(default=None, examples=["5f0c9d3a0b2e4a8c9d1e2f3a4b5c6d7e"])
    details: list[ErrorItem] | None = None


class ErrorResponse(BaseModel):
    """Shape of every error returned by the API."""

    error: ErrorContent


class Page(BaseModel, Generic[T]):  # noqa: UP046 - explicit TypeVar keeps 3.11 tooling happy
    model_config = ConfigDict(from_attributes=True)

    items: list[T]
    total: int
    limit: int
    offset: int


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing, invalid or expired credentials"},
    422: {"model": ErrorResponse, "description": "Invalid request payload"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
}
