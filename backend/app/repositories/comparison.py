"""Comparison persistence. Every query is scoped by ``owner_id``."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.comparison import DocumentComparison


class ComparisonRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, comparison: DocumentComparison) -> DocumentComparison:
        self._session.add(comparison)
        return comparison

    async def get_owned(
        self, owner_id: uuid.UUID, comparison_id: uuid.UUID
    ) -> DocumentComparison | None:
        result = await self._session.execute(
            select(DocumentComparison).where(
                DocumentComparison.id == comparison_id, DocumentComparison.owner_id == owner_id
            )
        )
        return result.scalar_one_or_none()

    async def list_owned(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[DocumentComparison], int]:
        total = await self._session.scalar(
            select(func.count())
            .select_from(DocumentComparison)
            .where(DocumentComparison.owner_id == owner_id)
        )
        result = await self._session.execute(
            select(DocumentComparison)
            .where(DocumentComparison.owner_id == owner_id)
            .order_by(DocumentComparison.created_at.desc(), DocumentComparison.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all()), int(total or 0)

    async def count_owned(self, owner_id: uuid.UUID) -> int:
        total = await self._session.scalar(
            select(func.count())
            .select_from(DocumentComparison)
            .where(DocumentComparison.owner_id == owner_id)
        )
        return int(total or 0)

    async def delete_owned(self, owner_id: uuid.UUID, comparison_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            delete(DocumentComparison).where(
                DocumentComparison.id == comparison_id, DocumentComparison.owner_id == owner_id
            )
        )
        return isinstance(result, CursorResult) and result.rowcount > 0
