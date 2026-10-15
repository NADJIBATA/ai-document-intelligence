from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import DocumentChunk


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_owned(
        self, owner_id: uuid.UUID, document_id: uuid.UUID, chunk_id: uuid.UUID
    ) -> DocumentChunk | None:
        result = await self._session.execute(
            select(DocumentChunk).where(
                DocumentChunk.id == chunk_id,
                DocumentChunk.document_id == document_id,
                DocumentChunk.owner_id == owner_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_owned_by_ids(
        self, owner_id: uuid.UUID, chunk_ids: list[uuid.UUID]
    ) -> list[DocumentChunk]:
        """Hydrate search hits. Ids that do not belong to ``owner_id`` are silently dropped."""
        if not chunk_ids:
            return []
        result = await self._session.execute(
            select(DocumentChunk).where(
                DocumentChunk.id.in_(chunk_ids), DocumentChunk.owner_id == owner_id
            )
        )
        return list(result.scalars().all())
