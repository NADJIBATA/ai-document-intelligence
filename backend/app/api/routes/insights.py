"""Document Intelligence: structured information extracted from a document."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from app.api.deps import CurrentUser, InsightServiceDep
from app.core.rate_limit import rate_limit
from app.models.insight import DocumentInsight
from app.schemas.common import ERROR_RESPONSES, ErrorResponse
from app.schemas.insight import InsightRead

router = APIRouter(prefix="/documents", tags=["intelligence"])

NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Document not found"}
}


def _insight_read(insight: DocumentInsight) -> InsightRead:
    data = insight.data or {}
    return InsightRead(
        document_id=insight.document_id,
        status=insight.status,
        document_type=insight.document_type,  # type: ignore[arg-type]  # checked by Pydantic
        title=insight.title,
        language=insight.language,
        summary=insight.summary,
        authors=data.get("authors", []),
        keywords=data.get("keywords", []),
        entities=data.get("entities", []),
        important_dates=data.get("important_dates", []),
        monetary_amounts=data.get("monetary_amounts", []),
        key_points=data.get("key_points", []),
        truncated=bool(data.get("coverage", {}).get("truncated", False)),
        model=insight.model,
        error_message=insight.error_message,
        updated_at=insight.updated_at,
    )


@router.get(
    "/{document_id}/insights",
    response_model=InsightRead,
    summary="Information extracted from a document",
    description=(
        "Title, type, summary, keywords, people and organizations, important dates, "
        "amounts and key points extracted by the language model once the document is "
        "indexed. `status` is `pending`/`processing` while the extraction is running. "
        "The pages attached to dates and amounts come from the indexed passages."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Document or insights not found"},
        **ERROR_RESPONSES,
    },
)
async def get_insights(
    document_id: uuid.UUID, current_user: CurrentUser, service: InsightServiceDep
) -> InsightRead:
    return _insight_read(await service.get(current_user, document_id))


@router.post(
    "/{document_id}/insights/regenerate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=InsightRead,
    summary="Extract the information again",
    description="Starts a new extraction in the background (poll `GET .../insights`).",
    responses={
        **NOT_FOUND,
        409: {
            "model": ErrorResponse,
            "description": "Document not indexed, or extraction already in progress",
        },
        503: {"model": ErrorResponse, "description": "Job queue unavailable"},
        **ERROR_RESPONSES,
    },
)
async def regenerate_insights(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    _: Annotated[None, Depends(rate_limit("analysis", "rate_limit_analysis", key="user"))],
    service: InsightServiceDep,
) -> InsightRead:
    return _insight_read(await service.regenerate(current_user, document_id))
