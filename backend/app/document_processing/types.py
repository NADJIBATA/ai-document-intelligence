"""Data structures shared by extractors, cleaner and chunker (framework-free)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class DocumentProcessingError(Exception):
    """A permanent, user-explainable processing failure (the job must not be retried)."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class UnsupportedDocumentError(DocumentProcessingError):
    """No extractor is registered for this content type."""


class ExtractionError(DocumentProcessingError):
    """The file could not be read (corrupted, encrypted, no text layer, ...)."""


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A paragraph-like unit of text with its location in the source document."""

    text: str
    page: int | None = None  # 1-based page number (PDF only)
    section: str | None = None  # nearest preceding heading


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    blocks: list[TextBlock]
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_length(self) -> int:
        return sum(len(block.text) for block in self.blocks)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A passage ready to be indexed.

    ``char_start`` / ``char_end`` are offsets in the document's cleaned text, i.e. the
    concatenation of every block separated by ``BLOCK_SEPARATOR``.
    """

    index: int
    text: str
    page: int | None
    section: str | None
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    extracted: ExtractedDocument
    chunks: list[Chunk]


BLOCK_SEPARATOR = "\n\n"
