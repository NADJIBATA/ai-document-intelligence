"""Question answering over the user's documents."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import ChatServiceDep, CurrentUser
from app.core.rate_limit import rate_limit
from app.rag.generation import AnswerSource
from app.schemas.chat import ChatMetadata, ChatRequest, ChatResponse, SourceRead
from app.schemas.common import ERROR_RESPONSES, ErrorResponse

router = APIRouter(prefix="/chat", tags=["chat"])


def _source(source: AnswerSource) -> SourceRead:
    return SourceRead(
        ref=source.ref,
        document_id=source.document_id,
        document=source.filename,
        page=source.page,
        section=source.section,
        chunk_id=source.chunk_id,
        excerpt=source.excerpt,
        score=source.score,
    )


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question about my documents",
    description=(
        "Retrieves the most relevant passages of the authenticated user's documents "
        "(hybrid search + reranking), then asks the language model to answer **only** "
        "from them. The answer contains markers such as `[S1]` that match the `ref` of "
        "the returned `sources`. When the documents do not contain the information, "
        "`answered` is `false` and `sources` is empty. Omit `conversation_id` to start a "
        "new conversation."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Question too long"},
        404: {"model": ErrorResponse, "description": "Conversation not found"},
        503: {"model": ErrorResponse, "description": "Language model unavailable"},
        **ERROR_RESPONSES,
    },
)
async def chat(
    payload: ChatRequest,
    current_user: CurrentUser,
    _: Annotated[None, Depends(rate_limit("chat", "rate_limit_chat", key="user"))],
    service: ChatServiceDep,
) -> ChatResponse:
    result = await service.ask(
        current_user,
        payload.question,
        conversation_id=payload.conversation_id,
        document_ids=payload.document_ids,
    )
    return ChatResponse(
        conversation_id=result.conversation_id,
        message_id=result.message_id,
        answer=result.answer.text,
        answered=result.answer.answered,
        sources=[_source(source) for source in result.answer.sources],
        metadata=ChatMetadata(
            model=result.answer.model,
            retrieval_ms=result.retrieval.timings_ms.get("total", 0.0),
            llm_ms=result.answer.llm_ms,
            total_ms=result.total_ms,
            degraded=result.retrieval.degraded,
        ),
    )
