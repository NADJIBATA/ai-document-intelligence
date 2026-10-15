"""PDF extraction with pdfplumber (MIT, built on pdfminer.six).

For each page, words are grouped into lines; lines are then grouped into paragraphs.
A line is considered a heading when its font is noticeably larger than the body text
of the document (or bold and short). The page number is kept on every block.
"""

from __future__ import annotations

import io
import statistics
from dataclasses import dataclass
from typing import Any

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.utils.exceptions import MalformedPDFException, PdfminerException

from app.document_processing.cleaning import is_edge_line, line_signature, repeated_edge_signatures
from app.document_processing.types import ExtractedDocument, ExtractionError, TextBlock

HEADING_SIZE_RATIO = 1.15
MAX_HEADING_CHARS = 120
LINE_TOLERANCE = 3.0  # points: words whose "top" differ less than this share a line
PARAGRAPH_GAP_RATIO = 1.6  # vertical gap (in line heights) that starts a new paragraph


@dataclass(slots=True)
class _Line:
    text: str
    size: float
    bold: bool
    top: float
    bottom: float


def _group_lines(words: list[dict[str, Any]]) -> list[_Line]:
    lines: list[_Line] = []
    current: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"]))):
        if current and abs(float(word["top"]) - float(current[0]["top"])) > LINE_TOLERANCE:
            lines.append(_make_line(current))
            current = []
        current.append(word)
    if current:
        lines.append(_make_line(current))
    return lines


def _make_line(words: list[dict[str, Any]]) -> _Line:
    words = sorted(words, key=lambda w: float(w["x0"]))
    sizes = [float(w.get("size", 0.0)) for w in words]
    fonts = [str(w.get("fontname", "")) for w in words]
    return _Line(
        text=" ".join(str(w["text"]) for w in words),
        size=max(sizes) if sizes else 0.0,
        bold=bool(fonts) and all("bold" in font.lower() for font in fonts),
        top=min(float(w["top"]) for w in words),
        bottom=max(float(w["bottom"]) for w in words),
    )


class PdfExtractor:
    content_types: tuple[str, ...] = ("application/pdf",)

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages_lines = [
                    _group_lines(
                        page.extract_words(keep_blank_chars=False, extra_attrs=["size", "fontname"])
                    )
                    for page in pdf.pages
                ]
                page_count = len(pdf.pages)
                metadata = {
                    key.lower(): str(value)
                    for key, value in (pdf.metadata or {}).items()
                    if key in ("Title", "Author", "Subject") and value
                }
        except (
            PDFPasswordIncorrect,
            PdfminerException,
            MalformedPDFException,
            PDFSyntaxError,
        ) as exc:
            if "password" in repr(exc).lower() or isinstance(exc, PDFPasswordIncorrect):
                raise ExtractionError("This PDF is password-protected.") from exc
            raise ExtractionError("This PDF is corrupted or cannot be read.") from exc

        # Remove running headers / footers before paragraph detection.
        repeated = repeated_edge_signatures(
            [[line.text for line in lines] for lines in pages_lines]
        )
        pages_lines = [
            [
                line
                for i, line in enumerate(lines)
                if not (is_edge_line(i, len(lines)) and line_signature(line.text) in repeated)
            ]
            for lines in pages_lines
        ]

        body_sizes = [line.size for lines in pages_lines for line in lines if line.size > 0]
        body_size = statistics.median(body_sizes) if body_sizes else 0.0

        blocks: list[TextBlock] = []
        section: str | None = None
        for page_number, lines in enumerate(pages_lines, start=1):
            paragraph: list[str] = []
            previous: _Line | None = None
            for line in lines:
                if self._is_heading(line, body_size):
                    self._flush(blocks, paragraph, page_number, section)
                    section = line.text.strip()
                    previous = None
                    continue
                if previous is not None:
                    height = max(previous.bottom - previous.top, 1.0)
                    if line.top - previous.bottom > height * (PARAGRAPH_GAP_RATIO - 1):
                        self._flush(blocks, paragraph, page_number, section)
                paragraph.append(line.text)
                previous = line
            self._flush(blocks, paragraph, page_number, section)

        if not any(block.text.strip() for block in blocks):
            raise ExtractionError(
                "No text could be extracted from this PDF (scanned document? OCR is not "
                "supported yet)."
            )
        return ExtractedDocument(blocks=blocks, page_count=page_count, metadata=metadata)

    @staticmethod
    def _flush(
        blocks: list[TextBlock], paragraph: list[str], page: int, section: str | None
    ) -> None:
        if paragraph:
            blocks.append(TextBlock("\n".join(paragraph), page=page, section=section))
            paragraph.clear()

    @staticmethod
    def _is_heading(line: _Line, body_size: float) -> bool:
        text = line.text.strip()
        if not text or len(text) > MAX_HEADING_CHARS or not any(c.isalpha() for c in text):
            return False
        if body_size and line.size >= body_size * HEADING_SIZE_RATIO:
            return True
        return line.bold and len(text) <= 80 and not text.endswith((".", ",", ";"))
