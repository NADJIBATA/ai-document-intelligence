from __future__ import annotations

import pytest

from app.document_processing.types import ExtractionError, UnsupportedDocumentError
from app.services.ingestion_service import build_document_processor, chunk_id_for
from tests.conftest import build_settings
from tests.fixtures.builders import contract_pdf, long_text


def test_pdf_end_to_end() -> None:
    processor = build_document_processor(build_settings())
    processed = processor.process(contract_pdf(), "application/pdf", "contract.pdf")
    assert processed.extracted.page_count == 3
    termination = [c for c in processed.chunks if c.section == "Termination"]
    assert termination and termination[0].page == 3


def test_long_text_uses_configured_chunk_size() -> None:
    processor = build_document_processor(build_settings(chunk_size=400, chunk_overlap=60))
    processed = processor.process(long_text(10).encode(), "text/plain", "notes.txt")
    assert len(processed.chunks) > 5
    assert max(len(c.text) for c in processed.chunks[:-1]) <= 400


def test_unsupported_content_type() -> None:
    processor = build_document_processor(build_settings())
    with pytest.raises(UnsupportedDocumentError):
        processor.process(b"\x89PNG", "image/png", "image.png")


def test_document_without_indexable_text() -> None:
    processor = build_document_processor(build_settings())
    with pytest.raises(ExtractionError):
        processor.process(b"---- ....", "text/plain", "noise.txt")


def test_chunk_ids_are_deterministic_and_unique() -> None:
    import uuid

    document_id = uuid.uuid4()
    assert chunk_id_for(document_id, 3) == chunk_id_for(document_id, 3)
    assert chunk_id_for(document_id, 3) != chunk_id_for(document_id, 4)
    assert chunk_id_for(document_id, 3) != chunk_id_for(uuid.uuid4(), 3)
