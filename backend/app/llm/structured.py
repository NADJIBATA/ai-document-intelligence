"""Structured outputs: ask for JSON matching a Pydantic model and validate it.

The JSON schema is passed to the provider (constrained decoding when the server
supports it), and the answer is always validated locally. On a validation failure the
errors are sent back to the model for **one** corrective attempt (configurable).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TypeVar

import structlog
from pydantic import BaseModel, ValidationError

from app.llm.base import ChatMessage, JsonSchema, LLMError, LLMProvider

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class StructuredOutputError(LLMError):
    """The model did not return a document matching the expected schema."""


def extract_json(text: str) -> str:
    """Return the JSON object contained in ``text`` (tolerates code fences and prose)."""
    stripped = text.strip()
    fenced = _CODE_FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the model output")
    return stripped[start : end + 1]


def _describe(error: Exception) -> str:
    if isinstance(error, ValidationError):
        problems = [
            f"- {'.'.join(str(part) for part in item['loc']) or '(root)'}: {item['msg']}"
            for item in error.errors()[:10]
        ]
        return "\n".join(problems)
    return f"- {error}"


def generate_structured(  # noqa: UP047 - explicit TypeVar keeps 3.11 tooling happy
    provider: LLMProvider,
    messages: Sequence[ChatMessage],
    schema: type[T],
    *,
    max_corrections: int = 1,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
) -> T:
    json_schema = JsonSchema(name=schema.__name__, schema=schema.model_json_schema())
    conversation = list(messages)
    last_error: Exception | None = None
    for attempt in range(max_corrections + 1):
        response = provider.generate(
            conversation, temperature=temperature, max_tokens=max_tokens, json_schema=json_schema
        )
        try:
            return schema.model_validate(json.loads(extract_json(response.content)))
        except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
            last_error = exc
            logger.warning(
                "structured_output_invalid",
                schema=schema.__name__,
                attempt=attempt + 1,
                error=type(exc).__name__,
            )
            conversation = [
                *conversation,
                ChatMessage("assistant", response.content),
                ChatMessage(
                    "user",
                    "Your previous answer is not valid for the expected JSON schema:\n"
                    f"{_describe(exc)}\n"
                    "Reply again with the corrected JSON object only, without any commentary.",
                ),
            ]
    raise StructuredOutputError(
        f"llm: output does not match {schema.__name__} ({type(last_error).__name__})"
    )
