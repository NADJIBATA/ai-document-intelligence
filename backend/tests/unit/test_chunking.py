from __future__ import annotations

from itertools import pairwise

import pytest

from app.document_processing.chunking import RecursiveTextChunker
from app.document_processing.types import BLOCK_SEPARATOR, ExtractedDocument, TextBlock
from tests.fixtures.builders import long_text


def _document_text(document: ExtractedDocument) -> str:
    return BLOCK_SEPARATOR.join(block.text for block in document.blocks)


def _single_block(text: str, page: int | None = 1, section: str | None = "S") -> ExtractedDocument:
    return ExtractedDocument(blocks=[TextBlock(text, page=page, section=section)])


@pytest.mark.parametrize(("size", "overlap"), [(1000, 150), (300, 50), (120, 0)])
def test_offsets_always_point_to_the_chunk_text(size: int, overlap: int) -> None:
    document = ExtractedDocument(
        blocks=[
            TextBlock(long_text(3), page=1, section="A"),
            TextBlock(long_text(2), page=2, section="A"),
            TextBlock(long_text(4), page=2, section="B"),
        ]
    )
    full = _document_text(document)
    chunks = RecursiveTextChunker(size, overlap, min_size=20).chunk(document)
    assert chunks
    for chunk in chunks:
        assert full[chunk.char_start : chunk.char_end] == chunk.text


def test_chunk_size_is_respected() -> None:
    chunker = RecursiveTextChunker(chunk_size=300, overlap=50, min_size=50)
    chunks = chunker.chunk(_single_block(long_text(10)))
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 + 50 for c in chunks)  # last chunk may absorb a tail
    assert all(len(c.text) <= 300 for c in chunks[:-1])


def test_consecutive_chunks_overlap_on_word_boundaries() -> None:
    chunks = RecursiveTextChunker(chunk_size=300, overlap=80, min_size=20).chunk(
        _single_block(long_text(6))
    )
    for previous, current in pairwise(chunks):
        assert current.char_start < previous.char_end  # overlap exists
        assert previous.char_end - current.char_start <= 80
        assert not current.text[0].isspace()


def test_no_overlap_when_disabled() -> None:
    chunks = RecursiveTextChunker(chunk_size=200, overlap=0, min_size=0).chunk(
        _single_block(long_text(4))
    )
    for previous, current in pairwise(chunks):
        assert current.char_start >= previous.char_end


def test_chunks_never_cross_pages_or_sections() -> None:
    document = ExtractedDocument(
        blocks=[
            TextBlock("Page one text about payment.", page=1, section="Payment"),
            TextBlock("Page two text about payment.", page=2, section="Payment"),
            TextBlock("Page two text about termination.", page=2, section="Termination"),
        ]
    )
    chunks = RecursiveTextChunker(chunk_size=1000, overlap=100, min_size=10).chunk(document)
    assert [(c.page, c.section, c.text) for c in chunks] == [
        (1, "Payment", "Page one text about payment."),
        (2, "Payment", "Page two text about payment."),
        (2, "Termination", "Page two text about termination."),
    ]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_small_trailing_piece_is_merged() -> None:
    text = "A" * 90 + ". " + "B" * 90 + ". " + "tail."
    chunks = RecursiveTextChunker(chunk_size=100, overlap=0, min_size=20).chunk(_single_block(text))
    assert chunks[-1].text.endswith("tail.")
    assert all(len(c.text) >= 20 for c in chunks)


def test_word_longer_than_chunk_size_is_hard_cut() -> None:
    chunks = RecursiveTextChunker(chunk_size=100, overlap=0, min_size=0).chunk(
        _single_block("x" * 350)
    )
    assert [len(c.text) for c in chunks] == [100, 100, 100, 50]


def test_chunking_is_deterministic() -> None:
    document = _single_block(long_text(8))
    chunker = RecursiveTextChunker(chunk_size=250, overlap=40, min_size=30)
    assert chunker.chunk(document) == chunker.chunk(document)


def test_punctuation_only_chunks_are_skipped() -> None:
    document = ExtractedDocument(
        blocks=[TextBlock("----", page=1), TextBlock("Real content here.", page=2)]
    )
    chunks = RecursiveTextChunker(chunk_size=100, overlap=0, min_size=0).chunk(document)
    assert [c.text for c in chunks] == ["Real content here."]


@pytest.mark.parametrize(
    ("size", "overlap", "min_size"), [(0, 0, 0), (100, 100, 0), (100, 10, 100)]
)
def test_invalid_parameters(size: int, overlap: int, min_size: int) -> None:
    with pytest.raises(ValueError):
        RecursiveTextChunker(size, overlap, min_size)
