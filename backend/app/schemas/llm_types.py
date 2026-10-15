"""Forgiving field types for structured LLM outputs.

A model sometimes returns a string where a list is expected, an unknown category, a
value that is slightly too long... These types normalise such values instead of
rejecting the whole document: one malformed field must not discard an extraction.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator


def clean_text(value: object) -> object:
    """Collapse whitespace."""
    return " ".join(value.split()) if isinstance(value, str) else value


def optional_text(value: object) -> object:
    """Collapse whitespace; an empty string becomes ``None``."""
    return clean_text(value) or None


def text_list(value: object) -> object:
    """Accept ``None`` or a single string where a list is expected; drop empty items."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        return [" ".join(item.split()) for item in value if isinstance(item, str) and item.strip()]
    return value


def bounded(limit: int) -> Any:
    """Collapse whitespace and truncate: an over-long value is cut, not rejected."""

    def normalise(value: object) -> object:
        return " ".join(value.split())[:limit].rstrip() if isinstance(value, str) else value

    return BeforeValidator(normalise)


def choice(allowed: tuple[str, ...], default: str = "other") -> Any:
    """Normalise a category ("Scientific Paper" -> "scientific_paper"); anything that is
    not in ``allowed`` becomes ``default``."""

    def normalise(value: object) -> object:
        if isinstance(value, str):
            candidate = value.strip().lower().replace(" ", "_").replace("-", "_")
            return candidate if candidate in allowed else default
        return default

    return BeforeValidator(normalise)


Text = Annotated[str, BeforeValidator(clean_text)]
ShortText = Annotated[str, bounded(200)]
Description = Annotated[str, bounded(500)]
OptionalText = Annotated[str | None, BeforeValidator(optional_text)]
TextList = Annotated[list[str], BeforeValidator(text_list)]
