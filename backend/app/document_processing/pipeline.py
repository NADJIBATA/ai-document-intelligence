"""extract -> clean -> chunk, without any I/O besides the bytes given as input."""

from __future__ import annotations

from app.document_processing.chunking import Chunker
from app.document_processing.cleaning import TextCleaner
from app.document_processing.registry import ExtractorRegistry
from app.document_processing.types import ExtractionError, ProcessedDocument


class DocumentProcessor:
    def __init__(self, registry: ExtractorRegistry, cleaner: TextCleaner, chunker: Chunker) -> None:
        self._registry = registry
        self._cleaner = cleaner
        self._chunker = chunker

    def process(self, data: bytes, content_type: str, filename: str) -> ProcessedDocument:
        extracted = self._registry.get(content_type).extract(data, filename)
        cleaned = self._cleaner.clean(extracted)
        chunks = self._chunker.chunk(cleaned)
        if not chunks:
            raise ExtractionError("The document does not contain any indexable text.")
        return ProcessedDocument(extracted=cleaned, chunks=chunks)
