from __future__ import annotations

import pytest

from app.document_processing.validation import (
    InvalidFileError,
    detect_file_type,
    sanitize_filename,
)
from tests.fixtures.builders import build_docx, simple_pdf

ALLOWED = ["pdf", "txt", "docx"]


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("contract.pdf", "contract.pdf"),
            ("../../etc/passwd", "passwd"),
            ("C:\\Users\\bob\\report.docx", "report.docx"),
            ("my<b>alert(1)</b>.txt", "b_.txt"),  # "/" is a path separator
            ("inv<oice>|?.pdf", "inv_oice_.pdf"),
            ("contrat été 2026.pdf", "contrat été 2026.pdf"),
            ("line\nbreak\x00.pdf", "linebreak.pdf"),
            ("", "document"),
            (None, "document"),
            ("...", "document"),
        ],
    )
    def test_sanitize(self, raw: str | None, expected: str) -> None:
        assert sanitize_filename(raw) == expected

    def test_long_names_keep_their_extension(self) -> None:
        name = sanitize_filename("a" * 500 + ".pdf")
        assert len(name) <= 200
        assert name.endswith(".pdf")


class TestDetectFileType:
    def test_pdf(self) -> None:
        detected = detect_file_type(simple_pdf("hello"), "a.pdf", ALLOWED)
        assert detected.extension == "pdf"
        assert detected.content_type == "application/pdf"

    def test_docx(self) -> None:
        detected = detect_file_type(build_docx([("Normal", "hello")]), "a.docx", ALLOWED)
        assert detected.extension == "docx"

    @pytest.mark.parametrize(
        "content",
        ["Bonjour, voilà un texte accentué.".encode(), "Hello\r\nWorld".encode("latin-1")],
    )
    def test_txt(self, content: bytes) -> None:
        assert detect_file_type(content, "notes.txt", ALLOWED).extension == "txt"

    def test_extension_is_case_insensitive(self) -> None:
        assert detect_file_type(simple_pdf("x"), "A.PDF", ALLOWED).extension == "pdf"

    def test_disallowed_extension(self) -> None:
        with pytest.raises(InvalidFileError) as exc_info:
            detect_file_type(b"MZ\x90\x00", "virus.exe", ALLOWED)
        assert exc_info.value.code == "unsupported_extension"

    def test_executable_renamed_to_pdf_is_rejected(self) -> None:
        with pytest.raises(InvalidFileError) as exc_info:
            detect_file_type(b"MZ\x90\x00\x03\x00\x00\x00" * 100, "invoice.pdf", ALLOWED)
        assert exc_info.value.code == "content_mismatch"

    def test_pdf_renamed_to_txt_is_rejected(self) -> None:
        with pytest.raises(InvalidFileError):
            detect_file_type(simple_pdf("x"), "notes.txt", ALLOWED)

    def test_plain_zip_renamed_to_docx_is_rejected(self) -> None:
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.md", "not a word document")
        with pytest.raises(InvalidFileError):
            detect_file_type(buffer.getvalue(), "fake.docx", ALLOWED)

    def test_binary_content_is_not_text(self) -> None:
        with pytest.raises(InvalidFileError):
            detect_file_type(bytes(range(256)) * 10, "data.txt", ALLOWED)
