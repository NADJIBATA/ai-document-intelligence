"""Local corpus for the offline evaluation.

The files of a directory (PDF, DOCX, TXT) go through the same processing as uploaded
documents (extraction, cleaning, chunking). Ids are deterministic: the document id
derives from the file name and the chunk id from (document id, chunk index), exactly
like in the backend, so a dataset keeps pointing at the same passages across runs.

List the passages to annotate a dataset::

    python -m evaluation.corpus evaluation/datasets/sample/corpus
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings
from app.document_processing.validation import InvalidFileError, detect_file_type
from app.services.ingestion_service import build_document_processor, chunk_id_for
from evaluation.dataset import DatasetError

# Namespace of the document ids of evaluation corpora (any fixed UUID works).
EVALUATION_NAMESPACE = uuid.UUID("0b8f9a52-7e57-4c1e-9a7a-3f5d2c9e6a10")


@dataclass(frozen=True, slots=True)
class CorpusChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    text: str
    page: int | None
    section: str | None

    @property
    def ref(self) -> str:
        return f"{self.filename}#{self.chunk_index}"


@dataclass
class Corpus:
    chunks: list[CorpusChunk]
    _by_ref: dict[str, CorpusChunk] = field(init=False, repr=False)
    _by_id: dict[str, CorpusChunk] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_ref = {chunk.ref: chunk for chunk in self.chunks}
        self._by_id = {str(chunk.chunk_id): chunk for chunk in self.chunks}

    @property
    def filenames(self) -> list[str]:
        return sorted({chunk.filename for chunk in self.chunks})

    def resolve(self, reference: str) -> str:
        """Chunk id (string) for a dataset reference: a UUID or "file#index"."""
        if reference in self._by_id:
            return reference
        chunk = self._by_ref.get(reference)
        if chunk is None:
            raise DatasetError(
                f"unknown chunk '{reference}': use a chunk id or '<file name>#<index>' "
                "(python -m evaluation.corpus <dir> lists them)"
            )
        return str(chunk.chunk_id)

    def get(self, chunk_id: str) -> CorpusChunk | None:
        return self._by_id.get(chunk_id)


def document_id_for(filename: str) -> uuid.UUID:
    return uuid.uuid5(EVALUATION_NAMESPACE, filename)


def load_corpus(directory: Path, settings: Settings) -> Corpus:
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    processor = build_document_processor(settings)
    chunks: list[CorpusChunk] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        data = path.read_bytes()
        try:
            detected = detect_file_type(data, path.name, settings.allowed_extensions)
        except InvalidFileError as exc:
            raise ValueError(f"{path.name}: {exc.message}") from exc
        processed = processor.process(data, detected.content_type, path.name)
        document_id = document_id_for(path.name)
        chunks.extend(
            CorpusChunk(
                chunk_id=chunk_id_for(document_id, chunk.index),
                document_id=document_id,
                filename=path.name,
                chunk_index=chunk.index,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
            )
            for chunk in processed.chunks
        )
    if not chunks:
        raise ValueError(f"no indexable document in {directory}")
    return Corpus(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List the passages of a local corpus.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--width", type=int, default=90, help="preview length")
    args = parser.parse_args(argv)

    corpus = load_corpus(args.directory, Settings())
    for chunk in corpus.chunks:
        preview = " ".join(chunk.text.split())[: args.width]
        location = f"p.{chunk.page}" if chunk.page else ""
        sys.stdout.write(f"{chunk.ref:<40} {chunk.chunk_id}  {location:<5} {preview}\n")
    sys.stdout.write(f"\n{len(corpus.chunks)} passages in {len(corpus.filenames)} documents\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
