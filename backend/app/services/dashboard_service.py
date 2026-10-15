"""Figures displayed on the dashboard, for the authenticated user only."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.comparison import DocumentComparison
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.user import User
from app.repositories.document import DocumentRepository
from app.schemas.dashboard import DashboardStats, DocumentStats


class DashboardService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._documents = DocumentRepository(session)

    async def stats(self, user: User) -> DashboardStats:
        by_status = await self._documents.count_by_status(user.id)
        passages, storage_bytes = (
            await self._session.execute(
                select(
                    func.coalesce(
                        func.sum(Document.chunk_count).filter(
                            Document.status == DocumentStatus.INDEXED
                        ),
                        0,
                    ),
                    func.coalesce(func.sum(Document.size_bytes), 0),
                ).where(Document.owner_id == user.id)
            )
        ).one()
        conversations = await self._session.scalar(
            select(func.count()).select_from(Conversation).where(Conversation.owner_id == user.id)
        )
        comparisons = await self._session.scalar(
            select(func.count())
            .select_from(DocumentComparison)
            .where(DocumentComparison.owner_id == user.id)
        )
        return DashboardStats(
            documents=DocumentStats(
                total=sum(by_status.values()),
                uploaded=by_status.get(DocumentStatus.UPLOADED, 0),
                processing=by_status.get(DocumentStatus.PROCESSING, 0),
                indexed=by_status.get(DocumentStatus.INDEXED, 0),
                failed=by_status.get(DocumentStatus.FAILED, 0),
            ),
            conversations=int(conversations or 0),
            comparisons=int(comparisons or 0),
            passages=int(passages),
            storage_bytes=int(storage_bytes),
        )
