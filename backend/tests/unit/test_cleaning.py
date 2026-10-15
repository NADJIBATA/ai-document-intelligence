from __future__ import annotations

from app.document_processing.cleaning import TextCleaner, normalize_text, strip_repeated_lines
from app.document_processing.types import ExtractedDocument, TextBlock


def test_hyphenation_and_wrapped_lines() -> None:
    assert (
        normalize_text("The infor-\nmation is\nconfidential.") == "The information is confidential."
    )


def test_hyphen_before_uppercase_is_kept() -> None:
    assert normalize_text("Franco-\nGerman") == "Franco- German"


def test_paragraph_breaks_are_preserved() -> None:
    assert (
        normalize_text("First line\n\n\n\nSecond   paragraph") == "First line\n\nSecond paragraph"
    )


def test_unicode_and_control_characters() -> None:
    assert normalize_text("\ufb01nal\u00a0price\x00\u200b: 10\tEUR") == "final price: 10 EUR"


def test_windows_line_endings() -> None:
    assert normalize_text("a\r\nb\r\rc") == "a b\n\nc"


def test_repeated_headers_and_page_numbers_are_removed() -> None:
    pages = [
        ["ACME - Confidential", f"Clause {n}.1", "Shared body line", f"Clause {n}.2", f"Page {n}"]
        for n in range(1, 5)
    ]
    cleaned = strip_repeated_lines(pages)
    # Lines in the middle of the page are kept even when they repeat.
    assert cleaned[0] == ["Clause 1.1", "Shared body line", "Clause 1.2"]
    assert all("Confidential" not in " ".join(page) for page in cleaned)


def test_short_documents_are_left_untouched() -> None:
    pages = [["Header", "Body"], ["Header", "Body 2"]]
    assert strip_repeated_lines(pages) == pages


def test_cleaner_drops_empty_blocks_and_keeps_locations() -> None:
    document = ExtractedDocument(
        blocks=[
            TextBlock("  ", page=1),
            TextBlock("Real   text", page=2, section="  Fees\n"),
        ],
        page_count=2,
    )
    cleaned = TextCleaner().clean(document)
    assert cleaned.blocks == [TextBlock("Real text", page=2, section="Fees")]
    assert cleaned.page_count == 2
