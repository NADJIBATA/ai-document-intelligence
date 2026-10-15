"""Answer generation: passages + question -> grounded answer with verified sources."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from app.llm.base import ChatMessage, LLMProvider
from app.rag.citations import parse_citations
from app.rag.prompts import NO_ANSWER_PATTERN, build_messages, refusal_message, source_ref
from app.rag.types import RetrievedChunk

logger = structlog.get_logger(__name__)

EXCERPT_CHARS = 300


@dataclass(frozen=True, slots=True)
class AnswerSource:
    """A passage backing the answer. Every field comes from the chunk metadata."""

    ref: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    section: str | None
    excerpt: str
    score: float


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str
    # False when the documents do not contain the answer (refusal message).
    answered: bool
    sources: list[AnswerSource]
    model: str | None = None
    llm_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # True when the model answered without any valid marker: every passage of the
    # context is then listed as a source.
    citation_fallback: bool = False


def make_excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit].rsplit(" ", 1)[0] or collapsed[:limit]
    return cut.rstrip(",;:") + "…"


def _source(ref: str, chunk: RetrievedChunk) -> AnswerSource:
    return AnswerSource(
        ref=ref,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page=chunk.page,
        section=chunk.section,
        excerpt=make_excerpt(chunk.text),
        score=round(chunk.score, 6),
    )


class AnswerGenerator:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        history_turns: int = 4,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self._history_turns = history_turns
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[ChatMessage] = (),
    ) -> GeneratedAnswer:
        """Answer ``question`` from ``chunks`` only. Raises ``LLMError`` if the model
        cannot be reached."""
        if not chunks:
            # Nothing relevant was retrieved: no need to call (and pay for) the model.
            return GeneratedAnswer(text=refusal_message(question), answered=False, sources=[])

        recent = list(history)[-2 * self._history_turns :] if self._history_turns else []
        messages = build_messages(question, chunks, recent)

        started = time.perf_counter()
        response = self._llm.generate(
            messages, temperature=self._temperature, max_tokens=self._max_tokens
        )
        llm_ms = round((time.perf_counter() - started) * 1000, 2)

        by_ref = {source_ref(position): chunk for position, chunk in enumerate(chunks, start=1)}
        parsed = parse_citations(response.content, by_ref.keys())

        def result(
            text: str, answered: bool, sources: list[AnswerSource], fallback: bool = False
        ) -> GeneratedAnswer:
            return GeneratedAnswer(
                text=text,
                answered=answered,
                sources=sources,
                model=response.model,
                llm_ms=llm_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                citation_fallback=fallback,
            )

        text = NO_ANSWER_PATTERN.sub("", parsed.text).strip()
        refused = NO_ANSWER_PATTERN.search(parsed.text) is not None and not parsed.refs
        if refused or not text:
            return result(refusal_message(question), False, [])

        if parsed.refs:
            return result(
                text,
                True,
                [
                    _source(source_ref(position), by_ref[original])
                    for position, original in enumerate(parsed.refs, start=1)
                ],
            )

        # The model answered without citing: the answer still comes from the context, so
        # every passage is listed rather than presenting an unsourced answer.
        logger.warning("answer_without_citations", model=response.model)
        return result(
            text, True, [_source(ref, chunk) for ref, chunk in by_ref.items()], fallback=True
        )
