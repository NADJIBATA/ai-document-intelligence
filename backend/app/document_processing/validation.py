"""Upload validation: file name sanitisation and content sniffing (magic bytes).

The extension claimed by the client is never trusted on its own: it must match the
actual content of the file.
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
DOCX_MAIN_PART = "word/document.xml"
SNIFF_SIZE = 8192
MAX_FILENAME_LENGTH = 200

CONTENT_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
}

_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\- ()\[\]]+", re.UNICODE)


class InvalidFileError(Exception):
    """The uploaded file is not acceptable. ``code`` is a stable machine-readable reason."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True, slots=True)
class DetectedFileType:
    extension: str
    content_type: str


def sanitize_filename(raw: str | None) -> str:
    """Return a display-safe file name (no path, no control chars, bounded length)."""
    name = raw or ""
    # Browsers may send a full path (old IE) or a POSIX path: keep the last component.
    name = PureWindowsPath(PurePosixPath(name).name).name
    name = unicodedata.normalize("NFKC", name)
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" ._")
    if not name:
        return "document"
    if len(name) > MAX_FILENAME_LENGTH:
        stem, dot, ext = name.rpartition(".")
        if dot and 0 < len(ext) <= 10:
            name = f"{stem[: MAX_FILENAME_LENGTH - len(ext) - 1]}.{ext}"
        else:
            name = name[:MAX_FILENAME_LENGTH]
    return name


def extension_of(filename: str) -> str:
    _, dot, ext = filename.rpartition(".")
    return ext.lower() if dot else ""


def _looks_like_text(head: bytes) -> bool:
    if not head:
        return False
    if b"\x00" in head:
        # UTF-16 files contain NUL bytes but start with a BOM.
        return head.startswith((b"\xff\xfe", b"\xfe\xff"))
    printable = sum(1 for byte in head if byte >= 0x20 or byte in (0x09, 0x0A, 0x0D, 0x0C))
    return printable / len(head) > 0.95


def _is_docx(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return DOCX_MAIN_PART in archive.namelist()
    except zipfile.BadZipFile:
        return False


def detect_file_type(
    data: bytes, claimed_filename: str, allowed_extensions: list[str]
) -> DetectedFileType:
    """Check that the content matches an allowed type and the claimed extension.

    Raises :class:`InvalidFileError` otherwise.
    """
    claimed = extension_of(claimed_filename)
    if claimed not in allowed_extensions:
        allowed = ", ".join(f".{ext}" for ext in allowed_extensions)
        raise InvalidFileError(
            f"Unsupported file extension. Allowed: {allowed}.", code="unsupported_extension"
        )

    head = data[:SNIFF_SIZE]
    if head.startswith(PDF_MAGIC):
        detected = "pdf"
    elif head.startswith(ZIP_MAGIC) and _is_docx(data):
        detected = "docx"
    elif _looks_like_text(head):
        detected = "txt"
    else:
        detected = ""

    if detected != claimed:
        raise InvalidFileError(
            "The file content does not match its extension.", code="content_mismatch"
        )
    return DetectedFileType(extension=detected, content_type=CONTENT_TYPES[detected])
