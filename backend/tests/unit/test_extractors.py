from __future__ import annotations

import pytest

from app.document_processing.extractors.docx import DocxExtractor
from app.document_processing.extractors.pdf import PdfExtractor
from app.document_processing.extractors.txt import TxtExtractor
from app.document_processing.registry import default_registry
from app.document_processing.types import ExtractionError, UnsupportedDocumentError
from tests.fixtures.builders import PdfLine, build_docx, build_pdf, contract_pdf, simple_pdf


class TestPdfExtractor:
    def test_pages_sections_and_metadata(self) -> None:
        document = PdfExtractor().extract(contract_pdf(), "contract.pdf")
        assert document.page_count == 3
        assert document.metadata == {"title": "Service Agreement"}
        by_page = {block.page: block for block in document.blocks}
        assert by_page[2].section == "Payment Terms"
        assert "4,500 EUR" in by_page[2].text
        assert by_page[3].section == "Termination"
        assert "sixty (60) days" in by_page[3].text

    def test_running_headers_and_footers_are_removed(self) -> None:
        document = PdfExtractor().extract(contract_pdf(), "contract.pdf")
        full_text = " ".join(block.text for block in document.blocks)
        assert "Confidential" not in full_text
        assert "Page 2 / 3" not in full_text

    def test_section_carries_over_to_the_next_page(self) -> None:
        pdf = build_pdf(
            [
                [PdfLine("Obligations", size=18), PdfLine(""), PdfLine("First obligation text.")],
                [PdfLine("Second obligation text continues on page two.")],
            ]
        )
        blocks = PdfExtractor().extract(pdf, "x.pdf").blocks
        assert [(b.page, b.section) for b in blocks] == [(1, "Obligations"), (2, "Obligations")]

    def test_non_ascii_text(self) -> None:
        blocks = (
            PdfExtractor().extract(simple_pdf("Résiliation du contrat à échéance."), "x.pdf").blocks
        )
        assert blocks[0].text == "Résiliation du contrat à échéance."

    def test_pdf_without_text_layer(self) -> None:
        with pytest.raises(ExtractionError, match="OCR"):
            PdfExtractor().extract(build_pdf([[]]), "scan.pdf")

    def test_corrupted_pdf(self) -> None:
        with pytest.raises(ExtractionError):
            PdfExtractor().extract(b"%PDF-1.4\nthis is not really a pdf", "broken.pdf")


class TestDocxExtractor:
    def test_headings_define_sections_and_tables_are_kept(self) -> None:
        data = build_docx(
            [
                ("Title", "Master Agreement"),
                ("Heading 1", "Scope"),
                ("Normal", "The supplier provides hosting."),
                ("Heading 2", "Fees"),
                ("Normal", "The fee is 100 EUR per month."),
            ],
            tables=[[["Item", "Price"], ["Hosting", "100 EUR"]]],
        )
        document = DocxExtractor().extract(data, "contract.docx")
        assert document.page_count is None
        assert document.metadata == {"title": "Test document", "author": "QA"}
        assert [(b.section, b.text) for b in document.blocks] == [
            ("Scope", "The supplier provides hosting."),
            ("Fees", "The fee is 100 EUR per month."),
            ("Fees", "Item | Price\nHosting | 100 EUR"),
        ]

    def test_empty_document(self) -> None:
        with pytest.raises(ExtractionError):
            DocxExtractor().extract(build_docx([("Normal", "   ")]), "empty.docx")

    def test_corrupted_document(self) -> None:
        with pytest.raises(ExtractionError):
            DocxExtractor().extract(b"PK\x03\x04garbage", "broken.docx")


class TestTxtExtractor:
    def test_markdown_and_capital_headings(self) -> None:
        text = (
            "# Introduction\nThis note explains the project.\n\n"
            "DELIVERABLES\n\nA prototype and a report.\n\n"
            "## Budget\n\nThe budget is 12 000 EUR."
        )
        blocks = TxtExtractor().extract(text.encode(), "notes.txt").blocks
        assert [(b.section, b.text) for b in blocks] == [
            ("Introduction", "This note explains the project."),
            ("DELIVERABLES", "A prototype and a report."),
            ("Budget", "The budget is 12 000 EUR."),
        ]

    @pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "latin-1", "utf-16"])
    def test_encodings(self, encoding: str) -> None:
        content = "Le délai de préavis est de trois mois pour les deux parties."
        blocks = TxtExtractor().extract(content.encode(encoding), "notes.txt").blocks
        assert blocks[0].text == content

    def test_empty_file(self) -> None:
        with pytest.raises(ExtractionError):
            TxtExtractor().extract(b"\n\n   \n", "empty.txt")


def test_registry_rejects_unknown_types() -> None:
    registry = default_registry()
    assert "application/pdf" in registry.content_types
    with pytest.raises(UnsupportedDocumentError):
        registry.get("image/png")
