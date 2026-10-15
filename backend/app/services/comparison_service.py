"""Document comparison.

* ``DocumentComparator`` - selects the passages of each document, asks the model for a
  structured list of differences, and attaches verified sources to every side.
* ``ComparisonJobService`` - the background job run by the Celery worker (synchronous).
* ``ComparisonService`` - the API side (create, read, list, delete).

Passage selection: a short document is sent entirely (best coverage); for a long one, a
hybrid search **restricted to that document** is run for each requested aspect, and the
results are merged. As everywhere else, sources (document, page, excerpt) are built
from the indexed passages; the model only returns passage identifiers (A1, B2...).
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import ConflictError, NotFoundError, ServiceUnavailableError
from app.llm.base import LLMError, LLMProvider
from app.llm.structured import generate_structured
from app.models.comparison import DocumentComparison
from app.models.document import Document
from app.models.enums import DocumentStatus, JobStatus
from app.models.user import User
from app.rag.generation import make_excerpt
from app.rag.pipeline import RAGPipeline
from app.rag.prompts import build_comparison_messages, comparison_ref
from app.rag.types import RetrievedChunk
from app.repositories.comparison import ComparisonRepository
from app.repositories.document import DocumentRepository
from app.schemas.comparison import DEFAULT_ASPECTS, ComparisonDraft, SideDraft
from app.services.task_dispatcher import TaskDispatcher, TaskDispatchError

logger = structlog.get_logger(__name__)

# Search query used to find the passages of a long document that matter for an aspect
# (French and English vocabulary: the lexical and the dense retrievers both benefit).
ASPECT_QUERIES: dict[str, str] = {
    "pricing": "price amount fee cost rate payment invoice prix montant tarif paiement facture",
    "dates": "effective date start date end date term duration date d'effet durée échéance",
    "conditions": "conditions requirements warranty eligibility conditions générales garantie",
    "clauses": (
        "termination liability confidentiality governing law clause résiliation "
        "responsabilité confidentialité"
    ),
    "deadlines": "deadline notice period within days delivery time délai préavis livraison",
    "obligations": "obligations shall must responsibilities duties engagements doit s'engage",
}

LLM_UNAVAILABLE_MESSAGE = "The language model is unavailable. Please retry later."
UNEXPECTED_ERROR_MESSAGE = "An unexpected error occurred while comparing the documents."
DOCUMENT_UNAVAILABLE_MESSAGE = "One of the documents is no longer available."
NO_TEXT_MESSAGE = "One of the documents contains no text to compare."
SCHEDULING_FAILED_MESSAGE = "The comparison could not be scheduled. Please retry later."

_REF = re.compile(r"[AB]\d+", re.IGNORECASE)


# ====================================================================== comparator
@dataclass(frozen=True, slots=True)
class DocumentRef:
    id: uuid.UUID
    filename: str
    chunk_count: int


class PassageSource(Protocol):
    """Reads the passages of one document. Implementations MUST filter on the user."""

    def list_document(
        self, user_id: uuid.UUID, document_id: uuid.UUID, *, limit: int | None = None
    ) -> list[RetrievedChunk]: ...


class ComparisonError(Exception):
    """The documents cannot be compared (permanent)."""


class DocumentComparator:
    def __init__(
        self,
        llm: LLMProvider,
        pipeline: RAGPipeline,
        passages: PassageSource,
        *,
        passages_per_aspect: int = 3,
        max_passages_per_document: int = 16,
        max_tokens: int = 3000,
    ) -> None:
        self._llm = llm
        self._pipeline = pipeline
        self._passages = passages
        self._passages_per_aspect = passages_per_aspect
        self._max_passages = max_passages_per_document
        self._max_tokens = max_tokens

    def compare(
        self,
        owner_id: uuid.UUID,
        document_a: DocumentRef,
        document_b: DocumentRef,
        aspects: Sequence[str],
    ) -> dict[str, Any]:
        """Return the JSON-serialisable comparison result.

        Raises ``ComparisonError`` (nothing to compare) or ``LLMError``.
        """
        aspects = list(aspects) or list(DEFAULT_ASPECTS)
        chunks_a = self._select_passages(owner_id, document_a, aspects)
        chunks_b = self._select_passages(owner_id, document_b, aspects)
        if not chunks_a or not chunks_b:
            raise ComparisonError(NO_TEXT_MESSAGE)

        draft = generate_structured(
            self._llm,
            build_comparison_messages(
                filename_a=document_a.filename,
                chunks_a=chunks_a,
                filename_b=document_b.filename,
                chunks_b=chunks_b,
                aspects=aspects,
            ),
            ComparisonDraft,
            max_tokens=self._max_tokens,
        )

        by_ref_a = {comparison_ref("A", i): chunk for i, chunk in enumerate(chunks_a, start=1)}
        by_ref_b = {comparison_ref("B", i): chunk for i, chunk in enumerate(chunks_b, start=1)}
        differences: list[dict[str, Any]] = []
        dropped = 0
        for difference in draft.differences:
            side_a = self._side(difference.document_a, by_ref_a)
            side_b = self._side(difference.document_b, by_ref_b)
            if not side_a["sources"] and not side_b["sources"]:
                dropped += 1  # not supported by any provided passage: not shown
                continue
            differences.append(
                {
                    "aspect": difference.aspect,
                    "title": difference.title,
                    "significance": difference.significance,
                    "document_a": side_a,
                    "document_b": side_b,
                }
            )
        if dropped:
            logger.warning("comparison_unsourced_differences_dropped", count=dropped)
        return {
            "summary": draft.summary,
            "differences": differences,
            "similarities": draft.similarities[:20],
            "model": self._llm.model_name,
            "passages": {"document_a": len(chunks_a), "document_b": len(chunks_b)},
        }

    def _select_passages(
        self, owner_id: uuid.UUID, document: DocumentRef, aspects: Sequence[str]
    ) -> list[RetrievedChunk]:
        if document.chunk_count <= self._max_passages:
            return self._passages.list_document(owner_id, document.id)

        selected: dict[uuid.UUID, RetrievedChunk] = {}
        for aspect in aspects:
            result = self._pipeline.retrieve(
                ASPECT_QUERIES.get(aspect, aspect),
                owner_id,
                document_ids=[document.id],
                top_k=self._passages_per_aspect,
            )
            for chunk in result.chunks:
                selected.setdefault(chunk.chunk_id, chunk)
        chunks = list(selected.values())[: self._max_passages]
        if not chunks:  # nothing matched the aspect queries: fall back to the beginning
            return self._passages.list_document(owner_id, document.id, limit=self._max_passages)
        # Reading order makes the context easier to follow for the model.
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    @staticmethod
    def _side(side: SideDraft, by_ref: dict[str, RetrievedChunk]) -> dict[str, Any]:
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in side.refs:
            for ref in _REF.findall(raw):
                ref = ref.upper()
                chunk = by_ref.get(ref)  # an A identifier is unknown on side B, and vice versa
                if chunk is None or ref in seen:
                    continue
                seen.add(ref)
                sources.append(
                    {
                        "ref": ref,
                        "document_id": str(chunk.document_id),
                        "document": chunk.filename,
                        "page": chunk.page,
                        "section": chunk.section,
                        "chunk_id": str(chunk.chunk_id),
                        "excerpt": make_excerpt(chunk.text),
                    }
                )
        return {"value": side.value, "sources": sources}


def build_document_comparator(
    llm: LLMProvider, pipeline: RAGPipeline, passages: PassageSource, settings: Any
) -> DocumentComparator:
    return DocumentComparator(
        llm,
        pipeline,
        passages,
        passages_per_aspect=settings.comparison_passages_per_aspect,
        max_passages_per_document=settings.comparison_max_passages_per_document,
        max_tokens=settings.llm_structured_max_tokens,
    )


# ================================================================== background job
class TransientComparisonError(Exception):
    """Temporary failure (model unavailable): the job should be retried."""


class ComparisonOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ComparisonJobService:
    """Synchronous job executed by the Celery worker. Idempotent."""

    def __init__(
        self, session_factory: sessionmaker[Session], comparator: DocumentComparator
    ) -> None:
        self._session_factory = session_factory
        self._comparator = comparator

    def run(self, comparison_id: uuid.UUID, *, final_attempt: bool = True) -> ComparisonOutcome:
        started = time.perf_counter()
        log = logger.bind(comparison_id=str(comparison_id))

        # 1. Claim the job and check that both documents are still usable.
        with self._session_factory() as session:
            comparison = session.get(DocumentComparison, comparison_id, with_for_update=True)
            if comparison is None:
                log.info("comparison_skipped", reason="deleted")
                return ComparisonOutcome.SKIPPED
            if comparison.status == JobStatus.COMPLETED:
                log.info("comparison_skipped", reason="already_completed")
                return ComparisonOutcome.SKIPPED
            owner_id = comparison.owner_id
            aspects = list(comparison.aspects)
            documents = [
                self._document_ref(session, owner_id, document_id)
                for document_id in (comparison.document_a_id, comparison.document_b_id)
            ]
            if documents[0] is None or documents[1] is None:
                comparison.status = JobStatus.FAILED
                comparison.error_message = DOCUMENT_UNAVAILABLE_MESSAGE
                session.commit()
                return ComparisonOutcome.FAILED
            comparison.status = JobStatus.PROCESSING
            comparison.error_message = None
            session.commit()
        document_a, document_b = documents[0], documents[1]

        # 2. Passage selection + model call.
        try:
            result = self._comparator.compare(owner_id, document_a, document_b, aspects)
        except ComparisonError as exc:
            return self._fail(comparison_id, str(exc))
        except LLMError as exc:
            if not final_attempt:
                log.warning("comparison_transient_error", error=str(exc))
                raise TransientComparisonError(str(exc)) from exc
            log.error("comparison_llm_failed", error=str(exc))
            return self._fail(comparison_id, LLM_UNAVAILABLE_MESSAGE)
        except Exception:
            log.exception("comparison_unexpected_error")
            return self._fail(comparison_id, UNEXPECTED_ERROR_MESSAGE)

        # 3. Store the result.
        with self._session_factory() as session:
            comparison = session.get(DocumentComparison, comparison_id, with_for_update=True)
            if comparison is None:
                return ComparisonOutcome.SKIPPED
            comparison.status = JobStatus.COMPLETED
            comparison.result = result
            comparison.error_message = None
            comparison.completed_at = datetime.now(UTC)
            session.commit()
        log.info(
            "comparison_completed",
            differences=len(result["differences"]),
            duration_s=round(time.perf_counter() - started, 3),
        )
        return ComparisonOutcome.COMPLETED

    @staticmethod
    def _document_ref(
        session: Session, owner_id: uuid.UUID, document_id: uuid.UUID | None
    ) -> DocumentRef | None:
        if document_id is None:  # deleted since the comparison was requested
            return None
        document = session.get(Document, document_id)
        if (
            document is None
            or document.owner_id != owner_id
            or document.status != DocumentStatus.INDEXED
        ):
            return None
        return DocumentRef(document.id, document.filename, document.chunk_count)

    def _fail(self, comparison_id: uuid.UUID, message: str) -> ComparisonOutcome:
        with self._session_factory() as session:
            comparison = session.get(DocumentComparison, comparison_id)
            if comparison is None:
                return ComparisonOutcome.SKIPPED
            comparison.status = JobStatus.FAILED
            comparison.error_message = message
            comparison.completed_at = datetime.now(UTC)
            session.commit()
        logger.warning("comparison_failed", comparison_id=str(comparison_id), reason=message)
        return ComparisonOutcome.FAILED


# ========================================================================== API side
class ComparisonService:
    def __init__(self, session: AsyncSession, dispatcher: TaskDispatcher) -> None:
        self._session = session
        self._dispatcher = dispatcher
        self._documents = DocumentRepository(session)
        self._comparisons = ComparisonRepository(session)

    async def create(
        self,
        user: User,
        document_a_id: uuid.UUID,
        document_b_id: uuid.UUID,
        aspects: Sequence[str] | None,
    ) -> DocumentComparison:
        document_a = await self._indexed_document(user, document_a_id, "A")
        document_b = await self._indexed_document(user, document_b_id, "B")
        comparison = self._comparisons.add(
            DocumentComparison(
                id=uuid.uuid4(),
                owner_id=user.id,
                document_a_id=document_a.id,
                document_b_id=document_b.id,
                document_a_name=document_a.filename,
                document_b_name=document_b.filename,
                status=JobStatus.PENDING,
                aspects=list(aspects or DEFAULT_ASPECTS),
            )
        )
        await self._session.commit()
        logger.info("comparison_requested", comparison_id=str(comparison.id))

        try:
            await run_in_threadpool(self._dispatcher.compare_documents, comparison.id)
        except TaskDispatchError as exc:
            logger.error("comparison_dispatch_failed", comparison_id=str(comparison.id))
            comparison.status = JobStatus.FAILED
            comparison.error_message = SCHEDULING_FAILED_MESSAGE
            await self._session.commit()
            raise ServiceUnavailableError(SCHEDULING_FAILED_MESSAGE) from exc

        # In tests the job ran inline and changed the row from another session.
        await self._session.refresh(comparison)
        return comparison

    async def get(self, user: User, comparison_id: uuid.UUID) -> DocumentComparison:
        comparison = await self._comparisons.get_owned(user.id, comparison_id)
        if comparison is None:
            raise NotFoundError("Comparison not found.", code="comparison_not_found")
        return comparison

    async def list(
        self, user: User, *, limit: int, offset: int
    ) -> tuple[list[DocumentComparison], int]:
        return await self._comparisons.list_owned(user.id, limit=limit, offset=offset)

    async def delete(self, user: User, comparison_id: uuid.UUID) -> None:
        if not await self._comparisons.delete_owned(user.id, comparison_id):
            raise NotFoundError("Comparison not found.", code="comparison_not_found")
        await self._session.commit()

    async def _indexed_document(
        self, user: User, document_id: uuid.UUID, label: Literal["A", "B"]
    ) -> Document:
        document = await self._documents.get_owned(user.id, document_id)
        if document is None:
            # Same answer whether it does not exist or belongs to someone else.
            raise NotFoundError(f"Document {label} not found.", code="document_not_found")
        if document.status != DocumentStatus.INDEXED:
            raise ConflictError(
                f"Document {label} is not indexed yet (status: {document.status.value}).",
                code="document_not_indexed",
            )
        return document
