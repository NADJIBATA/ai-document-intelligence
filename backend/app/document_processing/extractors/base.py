from __future__ import annotations

from typing import Protocol

from app.document_processing.types import ExtractedDocument


class DocumentExtractor(Protocol):
    """Turns raw file bytes into ordered text blocks.

    Adding a format (OCR for images, CSV, XLSX, ...) means writing one class that
    implements this protocol and registering it in ``registry.py``.
    """

    content_types: tuple[str, ...]

    def extract(self, data: bytes, filename: str) -> ExtractedDocument: ...
