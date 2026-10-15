"""Types shared by every LLM provider."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class JsonSchema:
    """Asks the provider to constrain its output to a JSON document of this shape."""

    name: str
    schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class LLMError(Exception):
    """The language model is unavailable or returned an unusable response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMProvider(Protocol):
    """Synchronous on purpose: the same provider is used by the API (through the
    threadpool) and by the Celery worker."""

    @property
    def name(self) -> str:
        """Provider identifier (``openai``, ``vllm``, ``ollama``, ``fake``)."""
        ...

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: JsonSchema | None = None,
    ) -> LLMResponse:
        """Return the completion for ``messages`` or raise :class:`LLMError`."""
        ...
