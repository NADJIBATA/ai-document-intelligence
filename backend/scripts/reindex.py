"""Rebuild the vector index from the chunks stored in PostgreSQL.

Run it after changing ``EMBEDDING_MODEL`` / ``EMBEDDING_DIMENSION`` (vectors of another
model live in another collection) or after losing the Qdrant volume. Documents are not
re-extracted: the chunks already stored in PostgreSQL are embedded again.

    python -m scripts.reindex                         # every indexed document
    python -m scripts.reindex --user-id <uuid>
    python -m scripts.reindex --document-id <uuid>

In Docker: ``docker compose exec backend python -m scripts.reindex``
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.session import create_sync_db_engine, create_sync_session_factory
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.rag.factory import build_embedder, build_vector_store
from app.services.indexing_service import ChunkIndexer, IndexableChunk, IndexingError


@dataclass(frozen=True, slots=True)
class ReindexReport:
    documents: int
    chunks: int
    failed: list[uuid.UUID]


def reindex_documents(
    session_factory: sessionmaker[Session],
    indexer: ChunkIndexer,
    *,
    user_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
) -> ReindexReport:
    statement = (
        select(Document.id, Document.owner_id, Document.filename)
        .where(Document.status == DocumentStatus.INDEXED)
        .order_by(Document.created_at)
    )
    if user_id is not None:
        statement = statement.where(Document.owner_id == user_id)
    if document_id is not None:
        statement = statement.where(Document.id == document_id)
    with session_factory() as session:
        documents = session.execute(statement).all()

    done = 0
    chunk_total = 0
    failed: list[uuid.UUID] = []
    for document in documents:
        with session_factory() as session:
            rows = session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.chunk_index,
                    DocumentChunk.text,
                    DocumentChunk.page,
                    DocumentChunk.section,
                )
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index)
            ).all()
        chunks = [
            IndexableChunk(
                chunk_id=row.id,
                chunk_index=row.chunk_index,
                text=row.text,
                page=row.page,
                section=row.section,
            )
            for row in rows
        ]
        try:
            indexer.index_document(
                document_id=document.id,
                owner_id=document.owner_id,
                filename=document.filename,
                chunks=chunks,
            )
        except IndexingError as exc:
            failed.append(document.id)
            sys.stderr.write(f"FAILED  {document.id}  {exc}\n")
            continue
        with session_factory() as session:
            session.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(embedding_model=indexer.model_name[:255])
            )
            session.commit()
        done += 1
        chunk_total += len(chunks)
        sys.stdout.write(f"indexed {document.id}  {len(chunks)} chunks\n")
    return ReindexReport(documents=done, chunks=chunk_total, failed=failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--user-id", type=uuid.UUID, default=None)
    parser.add_argument("--document-id", type=uuid.UUID, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    engine = create_sync_db_engine(settings, pool_size=2)
    embedder = build_embedder(settings)
    indexer = ChunkIndexer(embedder, build_vector_store(settings, embedder))
    try:
        report = reindex_documents(
            create_sync_session_factory(engine),
            indexer,
            user_id=args.user_id,
            document_id=args.document_id,
        )
    finally:
        engine.dispose()
    sys.stdout.write(
        f"{report.documents} document(s), {report.chunks} chunk(s) indexed with "
        f"{indexer.model_name}; {len(report.failed)} failure(s)\n"
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
