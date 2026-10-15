"""Plain-text extraction with encoding detection (charset-normalizer, MIT).

Headings are detected with simple heuristics: Markdown titles (``# Title``) and short
lines written in capital letters followed by a blank line.
"""

from __future__ import annotations

import re

from charset_normalizer import from_bytes

from app.document_processing.types import ExtractedDocument, ExtractionError, TextBlock

_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
_MAX_HEADING_CHARS = 100


def decode_text(data: bytes) -> str:
    for bom, encoding in (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if data.startswith(bom):
            return data.decode(encoding, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        best = from_bytes(data).best()
        if best is None:
            raise ExtractionError("The text encoding of this file could not be detected.") from None
        return str(best)


def _is_caps_heading(paragraph: str) -> bool:
    letters = [c for c in paragraph if c.isalpha()]
    return (
        "\n" not in paragraph
        and len(paragraph) <= _MAX_HEADING_CHARS
        and len(letters) >= 3
        and all(c.isupper() for c in letters)
    )


class TxtExtractor:
    content_types: tuple[str, ...] = ("text/plain",)

    def extract(self, data: bytes, filename: str) -> ExtractedDocument:
        text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n")
        blocks: list[TextBlock] = []
        section: str | None = None
        for raw in re.split(r"\n\s*\n", text):
            paragraph = raw.strip()
            if not paragraph:
                continue
            first_line, _, rest = paragraph.partition("\n")
            markdown = _MARKDOWN_HEADING.match(first_line)
            if markdown:
                section = markdown.group("title").strip()
                paragraph = rest.strip()
                if not paragraph:
                    continue
            elif _is_caps_heading(paragraph):
                section = paragraph
                continue
            blocks.append(TextBlock(text=paragraph, section=section))

        if not blocks:
            raise ExtractionError("This text file is empty.")
        return ExtractedDocument(blocks=blocks, page_count=None)
