"""Chunking: split cleaned blocks into overlapping passages.

Design rules:

* a chunk never crosses a page boundary nor a section boundary, so the page and
  section attached to a chunk are always exact (citations depend on it);
* splitting is recursive on natural boundaries (paragraph, line, sentence, word);
* consecutive chunks of the same group overlap by about ``overlap`` characters,
  and the overlap always starts on a boundary (never in the middle of a word);
* ``text == document_text[char_start:char_end]`` always holds.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import Protocol

from app.document_processing.types import BLOCK_SEPARATOR, Chunk, ExtractedDocument

# Ordered from the strongest to the weakest boundary. A match ends *after* the separator.
_SEPARATORS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n\n"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?;:])\s+"),
    re.compile(r"\s+"),
)
_MEANINGFUL = re.compile(r"\w")


class Chunker(Protocol):
    def chunk(self, document: ExtractedDocument) -> list[Chunk]: ...


@dataclass(frozen=True, slots=True)
class _Group:
    page: int | None
    section: str | None
    start: int  # offset of the group in the document text
    text: str


class RecursiveTextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 150, min_size: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        if not 0 <= min_size < chunk_size:
            raise ValueError("min_size must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size

    # ------------------------------------------------------------------ public
    def chunk(self, document: ExtractedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for group in self._groups(document):
            for start, end in self._windows(group.text):
                raw = group.text[start:end]
                stripped = raw.strip()
                if not _MEANINGFUL.search(stripped):
                    continue
                offset = group.start + start + (len(raw) - len(raw.lstrip()))
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        text=stripped,
                        page=group.page,
                        section=group.section,
                        char_start=offset,
                        char_end=offset + len(stripped),
                    )
                )
        return chunks

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _groups(document: ExtractedDocument) -> Iterator[_Group]:
        """Consecutive blocks sharing the same (page, section), with document offsets."""
        offset = 0
        positioned: list[tuple[int, int | None, str | None, str]] = []
        for block in document.blocks:
            positioned.append((offset, block.page, block.section, block.text))
            offset += len(block.text) + len(BLOCK_SEPARATOR)

        for (page, section), items in groupby(positioned, key=lambda item: (item[1], item[2])):
            members = list(items)
            yield _Group(
                page=page,
                section=section,
                start=members[0][0],
                text=BLOCK_SEPARATOR.join(member[3] for member in members),
            )

    def _boundaries(self, text: str, start: int, end: int, level: int = 0) -> list[int]:
        """Split points in ``text[start:end]`` so that every piece is <= chunk_size."""
        if end - start <= self.chunk_size:
            return []
        if level >= len(_SEPARATORS):
            # No natural boundary left: hard cut.
            return list(range(start + self.chunk_size, end, self.chunk_size))

        cuts = [m.end() + start for m in _SEPARATORS[level].finditer(text[start:end])]
        cuts = [cut for cut in cuts if start < cut < end]
        if not cuts:
            return self._boundaries(text, start, end, level + 1)

        points: list[int] = []
        previous = start
        for cut in [*cuts, end]:
            if cut - previous > self.chunk_size:
                points.extend(self._boundaries(text, previous, cut, level + 1))
            if cut != end:
                points.append(cut)
            previous = cut
        return points

    def _windows(self, text: str) -> list[tuple[int, int]]:
        """Greedy packing of pieces into windows of at most ``chunk_size`` characters."""
        if len(text) <= self.chunk_size:
            return [(0, len(text))]

        boundaries = [0, *self._boundaries(text, 0, len(text)), len(text)]
        windows: list[tuple[int, int]] = []
        start_idx = 0
        while start_idx < len(boundaries) - 1:
            start = boundaries[start_idx]
            end_idx = start_idx + 1
            while (
                end_idx + 1 < len(boundaries) and boundaries[end_idx + 1] - start <= self.chunk_size
            ):
                end_idx += 1
            end = boundaries[end_idx]
            windows.append((start, end))
            if end_idx == len(boundaries) - 1:
                break
            # Next window starts at the first boundary inside the overlap zone.
            next_idx = end_idx
            while next_idx - 1 > start_idx and end - boundaries[next_idx - 1] <= self.overlap:
                next_idx -= 1
            # If the next piece cannot fit after the overlap, drop the overlap rather
            # than producing a window that ends at the same place.
            if boundaries[end_idx + 1] - boundaries[next_idx] > self.chunk_size:
                next_idx = end_idx
            start_idx = next_idx

        # Merge a too-small trailing window into the previous one (the result may exceed
        # chunk_size by at most min_size characters).
        if len(windows) > 1:
            last_start, last_end = windows[-1]
            if len(text[last_start:last_end].strip()) < self.min_size:
                previous_start, _ = windows[-2]
                windows[-2:] = [(previous_start, last_end)]
        return windows
