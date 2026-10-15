"""Document comparison. Every route is scoped to the authenticated user."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response, status

from app.api.deps import ComparisonServiceDep, CurrentUser
from app.core.rate_limit import rate_limit
from app.models.comparison import DocumentComparison
from app.schemas.common import ERROR_RESPONSES, ErrorResponse, Page
from app.schemas.comparison import (
    ComparedDocumentRead,
    CompareRequest,
    ComparisonRead,
    ComparisonResultRead,
    ComparisonSummaryRead,
)

router = APIRouter(tags=["comparisons"])

NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Comparison not found"}
}


def _summary_fields(comparison: DocumentComparison) -> dict[str, Any]:
    return {
        "id": comparison.id,
        "status": comparison.status,
        "document_a": ComparedDocumentRead(
            id=comparison.document_a_id, name=comparison.document_a_name
        ),
        "document_b": ComparedDocumentRead(
            id=comparison.document_b_id, name=comparison.document_b_name
        ),
        "aspects": comparison.aspects,
        "error_message": comparison.error_message,
        "created_at": comparison.created_at,
        "completed_at": comparison.completed_at,
    }


def _read(comparison: DocumentComparison) -> ComparisonRead:
    result = (
        ComparisonResultRead.model_validate(comparison.result)
        if comparison.result is not None
        else None
    )
    return ComparisonRead(**_summary_fields(comparison), result=result)


@router.post(
    "/documents/compare",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ComparisonRead,
    summary="Compare two documents",
    description=(
        "Starts the comparison of two of my indexed documents (prices, dates, conditions, "
        "clauses, deadlines, obligations). The comparison runs in the background: poll "
        "`GET /api/comparisons/{id}` until the status is `completed` or `failed`. Every "
        "difference carries the passages (document, page, excerpt) that support each side."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        409: {"model": ErrorResponse, "description": "A document is not indexed yet"},
        503: {"model": ErrorResponse, "description": "Job queue unavailable"},
        **ERROR_RESPONSES,
    },
)
async def compare_documents(
    payload: CompareRequest,
    current_user: CurrentUser,
    _: Annotated[None, Depends(rate_limit("analysis", "rate_limit_analysis", key="user"))],
    service: ComparisonServiceDep,
) -> ComparisonRead:
    comparison = await service.create(
        current_user, payload.document_a_id, payload.document_b_id, payload.aspects
    )
    return _read(comparison)


@router.get(
    "/comparisons",
    response_model=Page[ComparisonSummaryRead],
    summary="List my comparisons",
    description="Most recent first, without the results.",
    responses=ERROR_RESPONSES,
)
async def list_comparisons(
    current_user: CurrentUser,
    service: ComparisonServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[ComparisonSummaryRead]:
    comparisons, total = await service.list(current_user, limit=limit, offset=offset)
    return Page[ComparisonSummaryRead](
        items=[ComparisonSummaryRead(**_summary_fields(c)) for c in comparisons],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/comparisons/{comparison_id}",
    response_model=ComparisonRead,
    summary="Get a comparison and its result",
    responses={**NOT_FOUND, **ERROR_RESPONSES},
)
async def get_comparison(
    comparison_id: uuid.UUID, current_user: CurrentUser, service: ComparisonServiceDep
) -> ComparisonRead:
    return _read(await service.get(current_user, comparison_id))


@router.delete(
    "/comparisons/{comparison_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a comparison",
    responses={**NOT_FOUND, **ERROR_RESPONSES},
)
async def delete_comparison(
    comparison_id: uuid.UUID, current_user: CurrentUser, service: ComparisonServiceDep
) -> Response:
    await service.delete(current_user, comparison_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
