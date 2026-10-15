"""Hybrid search use case on the API side."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import structlog
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import BadRequestError
from app.models.user import User
from app.rag.pipeline import RAGPipeline
from app.rag.types import RetrievalResult

logger = structlog.get_logger(__name__)


class SearchService:
    def __init__(self, pipeline: RAGPipeline, settings: Settings) -> None:
        self._pipeline = pipeline
        self._settings = settings

    async def search(
        self,
        user: User,
        query: str,
        *,
        document_ids: Sequence[uuid.UUID] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Search the documents of ``user`` — and only those.

        The user id comes from the access token, never from the request body, and is
        applied by every retrieval stage (BM25 corpus, Qdrant filter, hydration).
        """
        if len(query) > self._settings.max_question_chars:
            raise BadRequestError(
                f"The query exceeds {self._settings.max_question_chars} characters.",
                code="query_too_long",
            )
        # The pipeline is synchronous (CPU-bound BM25, blocking HTTP calls).
        result = await run_in_threadpool(
            self._pipeline.retrieve,
            query,
            user.id,
            document_ids=document_ids,
            top_k=top_k,
        )
        # The query text is user content: only its size is logged.
        logger.info(
            "search_completed",
            query_chars=len(result.query),
            results=len(result.chunks),
            candidates=result.candidate_count,
            degraded=result.degraded,
            duration_ms=result.timings_ms.get("total"),
        )
        return result
