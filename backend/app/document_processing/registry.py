"""Maps a content type to its extractor. The single place to plug a new format in."""

from __future__ import annotations

from app.document_processing.extractors.base import DocumentExtractor
from app.document_processing.extractors.docx import DocxExtractor
from app.document_processing.extractors.pdf import PdfExtractor
from app.document_processing.extractors.txt import TxtExtractor
from app.document_processing.types import UnsupportedDocumentError


class ExtractorRegistry:
    def __init__(self, extractors: list[DocumentExtractor] | None = None) -> None:
        self._by_type: dict[str, DocumentExtractor] = {}
        for extractor in extractors or []:
            self.register(extractor)

    def register(self, extractor: DocumentExtractor) -> None:
        for content_type in extractor.content_types:
            self._by_type[content_type] = extractor

    def get(self, content_type: str) -> DocumentExtractor:
        try:
            return self._by_type[content_type]
        except KeyError:
            raise UnsupportedDocumentError(
                f"No extractor available for content type '{content_type}'."
            ) from None

    @property
    def content_types(self) -> list[str]:
        return sorted(self._by_type)


def default_registry() -> ExtractorRegistry:
    # Future formats: ImageOcrExtractor (images), CsvExtractor, XlsxExtractor.
    return ExtractorRegistry([PdfExtractor(), DocxExtractor(), TxtExtractor()])
