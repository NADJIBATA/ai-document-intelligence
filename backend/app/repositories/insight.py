from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.insight import DocumentInsight


class InsightRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_owned(
        self, owner_id: uuid.UUID, document_id: uuid.UUID, *, refresh: bool = False
    ) -> DocumentInsight | None:
        """``refresh=True`` re-reads a row already loaded in this session (it may have
        been changed by a Core statement or by the background job)."""
        statement = select(DocumentInsight).where(
            DocumentInsight.document_id == document_id,
            DocumentInsight.owner_id == owner_id,
        )
        if refresh:
            statement = statement.execution_options(populate_existing=True)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()
