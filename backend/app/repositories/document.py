"""Document persistence. Every query is scoped by ``owner_id``: a user can only ever
reach its own rows, whatever identifier the client sends."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.enums import DocumentStatus


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, document: Document) -> Document:
        self._session.add(document)
        return document

    async def get_owned(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document | None:
        result = await self._session.execute(
            select(Document).where(Document.id == document_id, Document.owner_id == owner_id)
        )
        return result.scalar_one_or_none()

    async def get_owned_by_sha256(self, owner_id: uuid.UUID, sha256: str) -> Document | None:
        result = await self._session.execute(
            select(Document).where(Document.owner_id == owner_id, Document.sha256 == sha256)
        )
        return result.scalar_one_or_none()

    async def list_owned(
        self,
        owner_id: uuid.UUID,
        *,
        limit: int,
        offset: int,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], int]:
        conditions = [Document.owner_id == owner_id]
        if status is not None:
            conditions.append(Document.status == status)
        total = await self._session.scalar(
            select(func.count()).select_from(Document).where(*conditions)
        )
        result = await self._session.execute(
            select(Document)
            .where(*conditions)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all()), int(total or 0)

    async def count_by_status(self, owner_id: uuid.UUID) -> dict[DocumentStatus, int]:
        result = await self._session.execute(
            select(Document.status, func.count())
            .where(Document.owner_id == owner_id)
            .group_by(Document.status)
        )
        return {status: int(count) for status, count in result.all()}

    async def delete_owned(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> None:
        await self._session.execute(
            delete(Document).where(Document.id == document_id, Document.owner_id == owner_id)
        )
