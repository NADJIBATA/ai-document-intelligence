"""DOCX extraction with python-docx (MIT).

Paragraphs and tables are read in document order. Paragraph styles "Heading N",
"Titre N" (French Word) and "Title" define the current section. DOCX files have no
reliable pagination, so ``page`` is always ``None``.
"""

from __future__ import annotations

import io
import re
import zipfile

import docx
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.document_processing.types import ExtractedDocument, ExtractionError, TextBlock

_HEADING_STYLE = re.compile(r"^(heading|titre|title|titel|título)\b", re.IGNORECASE)


def _iter_body(document: DocxDocument) -> list[Paragraph | Table]:
    items: list[Paragraph | Table] = []
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            items.append(Paragraph(child, document))
        elif tag == "tbl":
            items.append(Table(child, document))
    return items


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = " ".join(cell.text.split())
            if not cells or cells[-1] != text:  # merged cells are repeated by python-docx
                cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


class DocxExtractor:
    content_types: tuple[str, ...] = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        try:
            document = docx.Document(io.BytesIO(data))
        except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise ExtractionError("This DOCX file is corrupted or cannot be read.") from exc

        blocks: list[TextBlock] = []
        section: str | None = None
        for item in _iter_body(document):
            if isinstance(item, Table):
                text = _table_text(item)
                if text:
                    blocks.append(TextBlock(text=text, section=section))
                continue
            text = item.text.strip()
            if not text:
                continue
            style_name = item.style.name if item.style is not None and item.style.name else ""
            if _HEADING_STYLE.match(style_name):
                section = text
                continue
            blocks.append(TextBlock(text=text, section=section))

        if not blocks:
            raise ExtractionError("This DOCX file does not contain any text.")

        properties = document.core_properties
        metadata = {
            key: value
            for key, value in (("title", properties.title), ("author", properties.author))
            if value
        }
        return ExtractedDocument(blocks=blocks, page_count=None, metadata=metadata)
