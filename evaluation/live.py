"""Live mode: evaluate the data of an existing user of a deployment.

Uses the configured PostgreSQL, Qdrant and model servers (``.env``). References of the
form "<file name>#<chunk index>" are resolved against that user's documents, so the
sample dataset can also be evaluated after uploading the sample corpus.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.session import create_sync_db_engine, create_sync_session_factory
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.user import User
from app.rag.embeddings import EmbeddingProvider
from app.rag.factory import RetrievalComponents, build_retrieval
from app.rag.vector_store import VectorStore
from evaluation.dataset import DatasetError
from evaluation.systems import retrieval_config


class LiveContext:
    def __init__(
        self,
        settings: Settings,
        user_email: str,
        *,
        top_k: int,
        embedder: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """``embedder`` and ``vector_store`` default to the configured ones (tests inject)."""
        self._engine: Engine = create_sync_db_engine(settings, pool_size=2)
        self._sessions: sessionmaker[Session] = create_sync_session_factory(self._engine)
        with self._sessions() as session:
            user_id = session.scalar(select(User.id).where(User.email == user_email.lower()))
        if user_id is None:
            self.close()
            raise ValueError(f"no user with the email {user_email}")
        self.user_id: uuid.UUID = user_id
        self.retrieval: RetrievalComponents = build_retrieval(
            settings, self._sessions, embedder=embedder, vector_store=vector_store
        )
        # The CLI asks for top_k results whatever RERANK_TOP_K is.
        self.retrieval.pipeline._config = retrieval_config(settings, top_k)

    def resolve(self, reference: str) -> str:
        try:
            return str(uuid.UUID(reference))
        except ValueError:
            pass
        filename, separator, index = reference.rpartition("#")
        if not separator or not index.isdigit():
            raise DatasetError(f"invalid chunk reference '{reference}'")
        with self._sessions() as session:
            chunk_id = session.scalar(
                select(DocumentChunk.id)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(
                    Document.owner_id == self.user_id,
                    Document.status == DocumentStatus.INDEXED,
                    Document.filename == filename,
                    DocumentChunk.chunk_index == int(index),
                )
            )
        if chunk_id is None:
            raise DatasetError(f"'{reference}' does not match an indexed passage of this user")
        return str(chunk_id)

    def close(self) -> None:
        if hasattr(self, "retrieval"):
            self.retrieval.close()
        self._engine.dispose()
