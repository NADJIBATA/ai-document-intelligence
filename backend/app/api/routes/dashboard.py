"""Dashboard figures."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUser, DashboardServiceDep
from app.schemas.common import ERROR_RESPONSES
from app.schemas.dashboard import DashboardStats

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get(
    "/stats",
    response_model=DashboardStats,
    summary="My statistics",
    description="Number of documents by processing status, conversations, comparisons, "
    "indexed passages and storage used by the authenticated user.",
    responses=ERROR_RESPONSES,
)
async def get_stats(current_user: CurrentUser, service: DashboardServiceDep) -> DashboardStats:
    return await service.stats(current_user)
