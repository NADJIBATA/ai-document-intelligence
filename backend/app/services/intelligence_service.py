"""Document Intelligence: structured information extracted from a document by the LLM.

* ``InsightExtractor`` - pure logic (passages in, structured result out), map-reduce
  for long documents. No database access: easy to test and to reuse.
* ``IntelligenceService`` - the background job run by the Celery worker (synchronous).
* ``InsightService`` - the API side (read, regenerate).

Map-reduce: the passages are grouped into windows of at most
``INTELLIGENCE_MAX_CHARS_PER_CALL`` characters; each window yields a partial extraction
(structured output validated with Pydantic); lists are merged and de-duplicated in
Python, and only the final summary needs one more model call.

Pages are never taken from the model: for dates and amounts the model must quote the
document, and the quote is then located in the indexed passages.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import ConflictError, NotFoundError, ServiceUnavailableError
from app.llm.base import LLMError, LLMProvider
from app.llm.structured import generate_structured
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.enums import DocumentStatus, JobStatus
from app.models.insight import DocumentInsight
from app.models.user import User
from app.rag.prompts import build_insights_messages, build_insights_reduce_messages
from app.repositories.document import DocumentRepository
from app.repositories.insight import InsightRepository
from app.schemas.insight import InsightsDraft, SummaryDraft
from app.services.task_dispatcher import TaskDispatcher, TaskDispatchError

logger = structlog.get_logger(__name__)

MAX_AUTHORS = 20
MAX_KEYWORDS = 15
MAX_ENTITIES = 60
MAX_DATES = 50
MAX_AMOUNTS = 50
MAX_KEY_POINTS = 12
# A job still "pending"/"processing" after this delay is considered lost: it can be
# started again.
IN_PROGRESS_GRACE = timedelta(minutes=15)

LLM_UNAVAILABLE_MESSAGE = "The language model is unavailable. Please retry later."
UNEXPECTED_ERROR_MESSAGE = "An unexpected error occurred while analysing the document."
NOT_INDEXED_MESSAGE = "The document is not indexed."
NO_TEXT_MESSAGE = "The document contains no text to analyse."
SCHEDULING_FAILED_MESSAGE = "The analysis could not be scheduled. Please retry later."

_SPACES = re.compile(r"\s+")


# ====================================================================== extraction
@dataclass(frozen=True, slots=True)
class Passage:
    chunk_id: uuid.UUID
    page: int | None
    text: str


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    document_type: str
    title: str | None
    language: str | None
    summary: str
    # JSON-serialisable: authors, keywords, entities, important_dates,
    # monetary_amounts, key_points, coverage.
    data: dict[str, Any]
    model: str
    llm_calls: int


class InsightError(Exception):
    """The document cannot be analysed (permanent)."""


def build_windows(passages: Sequence[Passage], max_chars: int) -> list[list[Passage]]:
    """Group consecutive passages into windows of at most ``max_chars`` characters."""
    windows: list[list[Passage]] = []
    current: list[Passage] = []
    size = 0
    for passage in passages:
        length = len(passage.text) + 2
        if current and size + length > max_chars:
            windows.append(current)
            current, size = [], 0
        current.append(passage)
        size += length
    if current:
        windows.append(current)
    return windows


def select_windows(windows: list[list[Passage]], limit: int) -> list[list[Passage]]:
    """Keep at most ``limit`` windows spread over the whole document.

    The first window (title, parties, authors) and the last one (signatures, totals,
    conclusions) are always kept.
    """
    if len(windows) <= limit:
        return windows
    if limit == 1:
        return [windows[0]]
    step = (len(windows) - 1) / (limit - 1)
    indices = sorted({round(index * step) for index in range(limit)})
    return [windows[index] for index in indices]


def _normalise(text: str) -> str:
    return _SPACES.sub(" ", text).strip().casefold()


def locate_quote(quote: str, passages: Sequence[Passage]) -> Passage | None:
    """First passage containing ``quote`` (case and whitespace insensitive)."""
    needle = _normalise(quote)
    if len(needle) < 2:
        return None
    for passage in passages:
        if needle in _normalise(passage.text):
            return passage
    return None


def _unique(values: Sequence[str], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result[:limit]


def _most_common(values: Sequence[str]) -> str | None:
    """Most frequent value; the earliest one wins a tie."""
    if not values:
        return None
    counts = Counter(values)
    best = max(counts.values())
    return next(value for value in values if counts[value] == best)


class InsightExtractor:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_chars_per_call: int = 12000,
        max_windows: int = 8,
        max_tokens: int = 3000,
    ) -> None:
        self._llm = llm
        self._max_chars_per_call = max_chars_per_call
        self._max_windows = max_windows
        self._max_tokens = max_tokens

    def extract(self, filename: str, passages: Sequence[Passage]) -> ExtractionResult:
        """Raises ``InsightError`` (nothing to analyse) or ``LLMError`` (model down or
        output never matching the schema)."""
        if not passages:
            raise InsightError(NO_TEXT_MESSAGE)
        all_windows = build_windows(passages, self._max_chars_per_call)
        windows = select_windows(all_windows, self._max_windows)

        # Map: one structured extraction per window.
        drafts: list[InsightsDraft] = []
        for index, window in enumerate(windows, start=1):
            text = "\n\n".join(passage.text for passage in window)
            drafts.append(
                generate_structured(
                    self._llm,
                    build_insights_messages(filename, text, part=index, parts=len(windows)),
                    InsightsDraft,
                    max_tokens=self._max_tokens,
                )
            )
        llm_calls = len(drafts)

        # Reduce: deterministic merge of the lists...
        key_points = _unique([point for draft in drafts for point in draft.key_points], 40)
        summaries = [draft.summary for draft in drafts if draft.summary]
        summary = summaries[0] if summaries else ""
        # ...and one model call to consolidate the summaries of a long document.
        if len(summaries) > 1:
            reduced = generate_structured(
                self._llm,
                build_insights_reduce_messages(summaries, key_points),
                SummaryDraft,
                max_tokens=self._max_tokens,
            )
            llm_calls += 1
            summary = reduced.summary or " ".join(summaries[:3])
            key_points = _unique(reduced.key_points, MAX_KEY_POINTS) or key_points

        types = [draft.document_type for draft in drafts if draft.document_type != "other"]
        languages = [draft.language.lower()[:16] for draft in drafts if draft.language]
        keyword_counts = Counter(
            keyword.casefold() for draft in drafts for keyword in set(draft.keywords)
        )
        keywords = _unique([keyword for draft in drafts for keyword in draft.keywords], 200)
        # Keywords found in several parts of the document first (stable sort).
        keywords.sort(key=lambda keyword: -keyword_counts[keyword.casefold()])

        data: dict[str, Any] = {
            "authors": _unique([a for draft in drafts for a in draft.authors], MAX_AUTHORS),
            "keywords": keywords[:MAX_KEYWORDS],
            "entities": self._merge_entities(drafts),
            "important_dates": self._merge_dates(drafts, passages),
            "monetary_amounts": self._merge_amounts(drafts, passages),
            "key_points": key_points[:MAX_KEY_POINTS],
            "coverage": {
                "windows_total": len(all_windows),
                "windows_analysed": len(windows),
                "truncated": len(windows) < len(all_windows),
            },
        }
        return ExtractionResult(
            document_type=_most_common(types) or "other",
            title=next((draft.title for draft in drafts if draft.title), None),
            language=_most_common(languages),
            summary=summary,
            data=data,
            model=self._llm.model_name,
            llm_calls=llm_calls,
        )

    @staticmethod
    def _merge_entities(drafts: Sequence[InsightsDraft]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for draft in drafts:
            for entity in draft.entities:
                key = entity.name.casefold()
                known = merged.get(key)
                if known is None:
                    merged[key] = {"name": entity.name, "type": entity.type}
                elif known["type"] == "other" and entity.type != "other":
                    known["type"] = entity.type  # a later part was more specific
        return list(merged.values())[:MAX_ENTITIES]

    @staticmethod
    def _merge_dates(
        drafts: Sequence[InsightsDraft], passages: Sequence[Passage]
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for draft in drafts:
            for item in draft.important_dates:
                key = (item.date.isoformat() if item.date else "", item.text.casefold())
                if key in merged:
                    continue
                source = locate_quote(item.text, passages)
                merged[key] = {
                    "date": item.date.isoformat() if item.date else None,
                    "text": item.text,
                    "description": item.description,
                    "page": source.page if source else None,
                    "chunk_id": str(source.chunk_id) if source else None,
                }
        return list(merged.values())[:MAX_DATES]

    @staticmethod
    def _merge_amounts(
        drafts: Sequence[InsightsDraft], passages: Sequence[Passage]
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for draft in drafts:
            for item in draft.monetary_amounts:
                key = item.text.casefold()
                if key in merged:
                    continue
                source = locate_quote(item.text, passages)
                merged[key] = {
                    "value": item.value,
                    "currency": item.currency.upper()[:16] if item.currency else None,
                    "text": item.text,
                    "description": item.description,
                    "page": source.page if source else None,
                    "chunk_id": str(source.chunk_id) if source else None,
                }
        return list(merged.values())[:MAX_AMOUNTS]


def build_insight_extractor(llm: LLMProvider, settings: Any) -> InsightExtractor:
    return InsightExtractor(
        llm,
        max_chars_per_call=settings.intelligence_max_chars_per_call,
        max_windows=settings.intelligence_max_windows,
        max_tokens=settings.llm_structured_max_tokens,
    )


# ================================================================== background job
class TransientInsightError(Exception):
    """Temporary failure (model unavailable): the job should be retried."""


class InsightOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class IntelligenceService:
    """Synchronous job executed by the Celery worker. Idempotent."""

    def __init__(self, session_factory: sessionmaker[Session], extractor: InsightExtractor) -> None:
        self._session_factory = session_factory
        self._extractor = extractor

    def mark_pending(self, document_id: uuid.UUID) -> bool:
        """Create or reset the insight row. Returns False if the document is gone."""
        with self._session_factory() as session:
            owner_id = session.scalar(select(Document.owner_id).where(Document.id == document_id))
            if owner_id is None:
                return False
            session.execute(pending_upsert(document_id, owner_id))
            session.commit()
        return True

    def mark_failed(self, document_id: uuid.UUID, message: str) -> None:
        with self._session_factory() as session:
            row = session.scalar(
                select(DocumentInsight).where(DocumentInsight.document_id == document_id)
            )
            if row is not None:
                row.status = JobStatus.FAILED
                row.error_message = message
                session.commit()

    def extract(self, document_id: uuid.UUID, *, final_attempt: bool = True) -> InsightOutcome:
        started = time.perf_counter()
        log = logger.bind(document_id=str(document_id))

        # 1. Claim the job and load the passages.
        with self._session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                log.info("insights_skipped", reason="document_deleted")
                return InsightOutcome.SKIPPED
            row = session.scalar(
                select(DocumentInsight)
                .where(DocumentInsight.document_id == document_id)
                .with_for_update()
            )
            if row is None:
                row = DocumentInsight(document_id=document_id, owner_id=document.owner_id)
                session.add(row)
            elif row.status == JobStatus.COMPLETED:
                log.info("insights_skipped", reason="already_completed")
                return InsightOutcome.SKIPPED
            if document.status != DocumentStatus.INDEXED:
                row.status = JobStatus.FAILED
                row.error_message = NOT_INDEXED_MESSAGE
                session.commit()
                return InsightOutcome.FAILED
            row.status = JobStatus.PROCESSING
            row.error_message = None
            filename = document.filename
            passages = [
                Passage(chunk_id=chunk_id, page=page, text=text)
                for chunk_id, page, text in session.execute(
                    select(DocumentChunk.id, DocumentChunk.page, DocumentChunk.text)
                    .where(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index)
                ).all()
            ]
            session.commit()

        # 2. Extraction (several model calls for a long document).
        try:
            result = self._extractor.extract(filename, passages)
        except InsightError as exc:
            return self._fail(document_id, str(exc))
        except LLMError as exc:
            if not final_attempt:
                log.warning("insights_transient_error", error=str(exc))
                raise TransientInsightError(str(exc)) from exc
            log.error("insights_llm_failed", error=str(exc))
            return self._fail(document_id, LLM_UNAVAILABLE_MESSAGE)
        except Exception:
            log.exception("insights_unexpected_error")
            return self._fail(document_id, UNEXPECTED_ERROR_MESSAGE)

        # 3. Store the result.
        with self._session_factory() as session:
            row = session.scalar(
                select(DocumentInsight)
                .where(DocumentInsight.document_id == document_id)
                .with_for_update()
            )
            if row is None:  # the document was deleted meanwhile (cascade)
                log.info("insights_skipped", reason="document_deleted_during_extraction")
                return InsightOutcome.SKIPPED
            row.status = JobStatus.COMPLETED
            row.document_type = result.document_type
            row.title = result.title[:1024] if result.title else None
            row.language = result.language
            row.summary = result.summary
            row.data = result.data
            row.model = result.model[:255]
            row.error_message = None
            session.commit()
        log.info(
            "insights_completed",
            llm_calls=result.llm_calls,
            truncated=result.data["coverage"]["truncated"],
            duration_s=round(time.perf_counter() - started, 3),
        )
        return InsightOutcome.COMPLETED

    def _fail(self, document_id: uuid.UUID, message: str) -> InsightOutcome:
        self.mark_failed(document_id, message)
        logger.warning("insights_failed", document_id=str(document_id), reason=message)
        return InsightOutcome.FAILED


def pending_upsert(document_id: uuid.UUID, owner_id: uuid.UUID) -> Any:
    """INSERT ... ON CONFLICT: (re)set the insight of a document to ``pending``.

    The previous content is kept until the new extraction replaces it.
    """
    now = datetime.now(UTC)
    statement = pg_insert(DocumentInsight).values(
        id=uuid.uuid4(),
        document_id=document_id,
        owner_id=owner_id,
        status=JobStatus.PENDING,
        data={},
        created_at=now,
        updated_at=now,
    )
    return statement.on_conflict_do_update(
        index_elements=[DocumentInsight.document_id],
        set_={"status": JobStatus.PENDING, "error_message": None, "updated_at": now},
    )


def schedule_insights(
    document_id: uuid.UUID, service: IntelligenceService, dispatcher: TaskDispatcher
) -> None:
    """Called by the worker once a document is indexed (``INTELLIGENCE_ENABLED``)."""
    if not service.mark_pending(document_id):
        return
    try:
        dispatcher.extract_insights(document_id)
    except TaskDispatchError:
        logger.error("insights_dispatch_failed", document_id=str(document_id))
        service.mark_failed(document_id, SCHEDULING_FAILED_MESSAGE)


# ========================================================================== API side
class InsightService:
    def __init__(self, session: AsyncSession, dispatcher: TaskDispatcher) -> None:
        self._session = session
        self._dispatcher = dispatcher
        self._documents = DocumentRepository(session)
        self._insights = InsightRepository(session)

    async def _get_document(self, user: User, document_id: uuid.UUID) -> Document:
        document = await self._documents.get_owned(user.id, document_id)
        if document is None:
            raise NotFoundError("Document not found.", code="document_not_found")
        return document

    async def get(self, user: User, document_id: uuid.UUID) -> DocumentInsight:
        await self._get_document(user, document_id)
        insight = await self._insights.get_owned(user.id, document_id)
        if insight is None:
            raise NotFoundError(
                "No insights have been extracted for this document yet.",
                code="insights_not_found",
            )
        return insight

    async def regenerate(self, user: User, document_id: uuid.UUID) -> DocumentInsight:
        document = await self._get_document(user, document_id)
        if document.status != DocumentStatus.INDEXED:
            raise ConflictError(
                "Insights can only be extracted from an indexed document.",
                code="document_not_indexed",
            )
        current = await self._insights.get_owned(user.id, document_id)
        if (
            current is not None
            and current.status in (JobStatus.PENDING, JobStatus.PROCESSING)
            and datetime.now(UTC) - current.updated_at < IN_PROGRESS_GRACE
        ):
            raise ConflictError(
                "The analysis of this document is already in progress.",
                code="insights_in_progress",
            )
        await self._session.execute(pending_upsert(document_id, user.id))
        await self._session.commit()

        try:
            await run_in_threadpool(self._dispatcher.extract_insights, document_id)
        except TaskDispatchError as exc:
            logger.error("insights_dispatch_failed", document_id=str(document_id))
            insight = await self._insights.get_owned(user.id, document_id, refresh=True)
            if insight is not None:
                insight.status = JobStatus.FAILED
                insight.error_message = SCHEDULING_FAILED_MESSAGE
                await self._session.commit()
            raise ServiceUnavailableError(SCHEDULING_FAILED_MESSAGE) from exc

        # The row was written with a Core statement (and, in tests, by the inline job):
        # re-read it rather than trusting the identity map.
        insight = await self._insights.get_owned(user.id, document_id, refresh=True)
        if insight is None:  # document deleted in the meantime
            raise NotFoundError("Document not found.", code="document_not_found")
        return insight
