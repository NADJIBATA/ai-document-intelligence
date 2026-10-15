"""Aggregates every business router under the ``/api`` prefix."""

from fastapi import APIRouter

from app.api.routes import (
    auth,
    chat,
    comparisons,
    conversations,
    dashboard,
    documents,
    insights,
    search,
)

api_router = APIRouter(prefix="/api")
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(insights.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)
api_router.include_router(conversations.router)
api_router.include_router(comparisons.router)
api_router.include_router(dashboard.router)
