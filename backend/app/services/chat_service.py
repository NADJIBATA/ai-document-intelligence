"""Chat use cases: answer a question from the user's documents, manage conversations."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import BadRequestError, NotFoundError, ServiceUnavailableError
from app.llm.base import ChatMessage, LLMError
from app.models.conversation import Conversation, Message
from app.models.enums import MessageRole
from app.models.user import User
from app.rag.generation import AnswerGenerator, AnswerSource, GeneratedAnswer
from app.rag.pipeline import RAGPipeline
from app.rag.types import RetrievalResult
from app.repositories.conversation import ConversationRepository

logger = structlog.get_logger(__name__)

TITLE_MAX_CHARS = 80


@dataclass(frozen=True, slots=True)
class ChatResult:
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: GeneratedAnswer
    retrieval: RetrievalResult
    total_ms: float


def make_title(question: str) -> str:
    """Conversation title derived from its first question."""
    first_line = " ".join(question.split())
    if len(first_line) <= TITLE_MAX_CHARS:
        return first_line
    return first_line[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def source_snapshot(source: AnswerSource) -> dict[str, Any]:
    """JSON stored with the assistant message (same shape as the API ``SourceRead``)."""
    return {
        "ref": source.ref,
        "document_id": str(source.document_id),
        "document": source.filename,
        "page": source.page,
        "section": source.section,
        "chunk_id": str(source.chunk_id),
        "excerpt": source.excerpt,
        "score": source.score,
    }


class ChatService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        pipeline: RAGPipeline,
        generator: AnswerGenerator,
    ) -> None:
        self._session = session
        self._settings = settings
        self._pipeline = pipeline
        self._generator = generator
        self._conversations = ConversationRepository(session)

    # --------------------------------------------------------------------- ask
    async def ask(
        self,
        user: User,
        question: str,
        *,
        conversation_id: uuid.UUID | None = None,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        asked_at = datetime.now(UTC)
        if len(question) > self._settings.max_question_chars:
            raise BadRequestError(
                f"The question exceeds {self._settings.max_question_chars} characters.",
                code="question_too_long",
            )

        # 1. Conversation and history (only conversations of the user can be continued).
        history: list[ChatMessage] = []
        conversation: Conversation | None = None
        if conversation_id is not None:
            conversation = await self._get_owned(user, conversation_id)
            previous = await self._conversations.list_messages(
                user.id, conversation_id, last=2 * self._settings.chat_history_turns
            )
            history = [ChatMessage(message.role.value, message.content) for message in previous]
        # The retrieval and the model call can take seconds: the database connection goes
        # back to the pool meanwhile.
        await self._session.commit()

        # 2. Retrieval, restricted to the user's documents by every stage of the pipeline.
        retrieval = await run_in_threadpool(
            self._pipeline.retrieve, question, user.id, document_ids=document_ids
        )

        # 3. Generation, from the retrieved passages only.
        try:
            answer = await run_in_threadpool(
                self._generator.generate, question, retrieval.chunks, history
            )
        except LLMError as exc:
            logger.error("llm_call_failed", error=str(exc), status_code=exc.status_code)
            raise ServiceUnavailableError(
                "The language model is temporarily unavailable. Please retry later.",
                code="llm_unavailable",
            ) from exc

        # 4. Persistence: the question and its answer are written together, so a failed
        #    exchange leaves no orphan question in the history.
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        answered_at = max(datetime.now(UTC), asked_at + timedelta(microseconds=1))
        if conversation is None:
            conversation = self._conversations.add(
                Conversation(id=uuid.uuid4(), owner_id=user.id, title=make_title(question))
            )
        conversation.updated_at = answered_at
        self._conversations.add_message(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=question,
                citations=[],
                meta={"document_ids": [str(d) for d in document_ids]} if document_ids else {},
                created_at=asked_at,
            )
        )
        reply = self._conversations.add_message(
            Message(
                id=uuid.uuid4(),
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=answer.text,
                citations=[source_snapshot(source) for source in answer.sources],
                meta={
                    "answered": answer.answered,
                    "model": answer.model,
                    "retrieval_ms": retrieval.timings_ms.get("total", 0.0),
                    "llm_ms": answer.llm_ms,
                    "total_ms": total_ms,
                    "degraded": retrieval.degraded,
                    "citation_fallback": answer.citation_fallback,
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                },
                created_at=answered_at,
            )
        )
        try:
            await self._session.commit()
        except (IntegrityError, StaleDataError) as exc:
            # The conversation was deleted while the answer was being generated.
            await self._session.rollback()
            raise NotFoundError("Conversation not found.", code="conversation_not_found") from exc

        # Neither the question nor the answer is logged: only sizes and timings.
        logger.info(
            "chat_completed",
            conversation_id=str(conversation.id),
            answered=answer.answered,
            sources=len(answer.sources),
            passages=len(retrieval.chunks),
            degraded=retrieval.degraded,
            model=answer.model,
            llm_ms=answer.llm_ms,
            duration_ms=total_ms,
        )
        return ChatResult(
            conversation_id=conversation.id,
            message_id=reply.id,
            answer=answer,
            retrieval=retrieval,
            total_ms=total_ms,
        )

    # ----------------------------------------------------------- conversations
    async def list_conversations(
        self, user: User, *, limit: int, offset: int
    ) -> tuple[list[tuple[Conversation, int]], int]:
        return await self._conversations.list_owned(user.id, limit=limit, offset=offset)

    async def get_conversation(
        self, user: User, conversation_id: uuid.UUID
    ) -> tuple[Conversation, list[Message]]:
        conversation = await self._get_owned(user, conversation_id)
        messages = await self._conversations.list_messages(user.id, conversation_id)
        return conversation, messages

    async def delete_conversation(self, user: User, conversation_id: uuid.UUID) -> None:
        deleted = await self._conversations.delete_owned(user.id, conversation_id)
        if not deleted:
            raise NotFoundError("Conversation not found.", code="conversation_not_found")
        await self._session.commit()
        logger.info("conversation_deleted", conversation_id=str(conversation_id))

    async def _get_owned(self, user: User, conversation_id: uuid.UUID) -> Conversation:
        conversation = await self._conversations.get_owned(user.id, conversation_id)
        if conversation is None:
            # Same answer whether it does not exist or belongs to someone else.
            raise NotFoundError("Conversation not found.", code="conversation_not_found")
        return conversation
