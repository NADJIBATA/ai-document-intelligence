"""Text normalisation applied to every extracted block."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from app.document_processing.types import ExtractedDocument, TextBlock

# End-of-line hyphenation: "infor-\nmation" -> "information" (only before a lowercase letter).
_HYPHENATION = re.compile(r"(\w)-\n(?=[a-zà-ÿ])")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‍﻿]")
_SPACES = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# "12", "- 12 -", "Page 3", "Page 3 / 12", "p. 3 of 12", "3 sur 12"...
_PAGE_NUMBER_LINE = re.compile(
    r"^\W*(page|p\.?|seite|pagina)?\s*\d{1,4}(\s*(/|of|sur|de|von)\s*\d{1,4})?\W*$",
    re.IGNORECASE,
)

# Lines repeated on at least this share of pages are treated as headers/footers.
REPEATED_LINE_RATIO = 0.5
REPEATED_LINE_MIN_PAGES = 3
EDGE_LINES = 2  # only the first/last lines of a page can be headers/footers


def normalize_text(text: str) -> str:
    """Unicode NFKC, control characters, hyphenation, wrapped lines and spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHENATION.sub(r"\1", text)
    paragraphs = [
        " ".join(line.strip() for line in part.split("\n") if line.strip())
        for part in re.split(r"\n\s*\n", text)
    ]
    text = "\n\n".join(p for p in paragraphs if p)
    text = _SPACES.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def line_signature(line: str) -> str:
    """Comparable form of a line; all page-number lines share the same signature."""
    text = " ".join(line.split()).lower()
    return "<page-number>" if _PAGE_NUMBER_LINE.match(text) else text


def is_edge_line(index: int, total: int) -> bool:
    return index < EDGE_LINES or index >= total - EDGE_LINES


def repeated_edge_signatures(pages: list[list[str]]) -> set[str]:
    """Signatures of lines found at the top/bottom of many pages (headers, footers)."""
    if len(pages) < REPEATED_LINE_MIN_PAGES:
        return set()
    counter: Counter[str] = Counter()
    for lines in pages:
        edges = {
            line_signature(line) for i, line in enumerate(lines) if is_edge_line(i, len(lines))
        }
        counter.update(sig for sig in edges if sig)
    threshold = max(REPEATED_LINE_MIN_PAGES, int(len(pages) * REPEATED_LINE_RATIO))
    return {sig for sig, count in counter.items() if count >= threshold}


def strip_repeated_lines(pages: list[list[str]]) -> list[list[str]]:
    """Remove running headers/footers (e.g. "ACME Corp - Confidential", "Page 3 / 12")."""
    repeated = repeated_edge_signatures(pages)
    if not repeated:
        return pages
    return [
        [
            line
            for i, line in enumerate(lines)
            if not (is_edge_line(i, len(lines)) and line_signature(line) in repeated)
        ]
        for lines in pages
    ]


class TextCleaner:
    def clean(self, document: ExtractedDocument) -> ExtractedDocument:
        blocks = []
        for block in document.blocks:
            text = normalize_text(block.text)
            if text:
                section = normalize_text(block.section) if block.section else None
                blocks.append(TextBlock(text=text, page=block.page, section=section or None))
        return ExtractedDocument(
            blocks=blocks, page_count=document.page_count, metadata=dict(document.metadata)
        )
