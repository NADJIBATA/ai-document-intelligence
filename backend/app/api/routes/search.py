"""Hybrid search endpoint (retrieval only, no LLM)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import CurrentUser, SearchServiceDep
from app.core.rate_limit import rate_limit
from app.schemas.common import ERROR_RESPONSES, ErrorResponse
from app.schemas.search import SearchRequest, SearchResponse, SearchResultRead

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search my documents",
    description=(
        "Runs the retrieval pipeline used by the chat, without generating an answer: "
        "BM25 and dense search in parallel, Reciprocal Rank Fusion, cross-encoder "
        "reranking. Only the documents of the authenticated user are searched. "
        "`degraded` lists the stages that were skipped because a model server was "
        "unavailable."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Query too long"},
        **ERROR_RESPONSES,
    },
)
async def search_documents(
    payload: SearchRequest,
    current_user: CurrentUser,
    _: Annotated[None, Depends(rate_limit("search", "rate_limit_search", key="user"))],
    service: SearchServiceDep,
) -> SearchResponse:
    result = await service.search(
        current_user, payload.query, document_ids=payload.document_ids, top_k=payload.top_k
    )
    return SearchResponse(
        query=result.query,
        results=[
            SearchResultRead(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=chunk.score,
                rerank_score=chunk.rerank_score,
                rrf_score=chunk.rrf_score,
                bm25_rank=chunk.bm25_rank,
                dense_rank=chunk.dense_rank,
            )
            for chunk in result.chunks
        ],
        candidate_count=result.candidate_count,
        degraded=result.degraded,
        timings_ms=result.timings_ms,
    )
