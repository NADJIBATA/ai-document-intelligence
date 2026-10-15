"""Document use cases on the API side: upload, listing, status, download, deletion."""

from __future__ import annotations

import hashlib
import io
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import structlog
from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    UnsupportedMediaTypeError,
)
from app.document_processing.validation import InvalidFileError, detect_file_type, sanitize_filename
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.user import User
from app.rag.vector_store import VectorStore, VectorStoreError
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.services.storage_service import (
    ObjectNotFoundError,
    ObjectStorage,
    StorageError,
    document_storage_key,
)
from app.services.task_dispatcher import TaskDispatcher, TaskDispatchError

logger = structlog.get_logger(__name__)

READ_CHUNK_SIZE = 1024 * 1024
SCHEDULING_FAILED_MESSAGE = "Processing could not be scheduled. Please retry later."


@dataclass(frozen=True, slots=True)
class FileDownload:
    filename: str
    content_type: str
    size_bytes: int
    body: Iterator[bytes]


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        storage: ObjectStorage,
        dispatcher: TaskDispatcher,
        vector_store: VectorStore,
    ) -> None:
        self._session = session
        self._settings = settings
        self._storage = storage
        self._dispatcher = dispatcher
        self._vector_store = vector_store
        self._documents = DocumentRepository(session)
        self._chunks = ChunkRepository(session)

    # ------------------------------------------------------------------ upload
    async def upload(self, owner: User, upload: UploadFile) -> Document:
        data, sha256 = await self._read_bounded(upload)
        if not data:
            raise BadRequestError("The uploaded file is empty.", code="empty_file")

        filename = sanitize_filename(upload.filename)
        try:
            detected = detect_file_type(data, filename, self._settings.allowed_extensions)
        except InvalidFileError as exc:
            raise UnsupportedMediaTypeError(exc.message, code=exc.code) from exc

        existing = await self._documents.get_owned_by_sha256(owner.id, sha256)
        if existing is not None:
            raise ConflictError(
                f"This file has already been uploaded (document {existing.id}).",
                code="duplicate_document",
            )

        document_id = uuid.uuid4()
        storage_key = document_storage_key(owner.id, document_id, detected.extension)
        try:
            await run_in_threadpool(
                self._storage.put_object,
                storage_key,
                io.BytesIO(data),
                len(data),
                detected.content_type,
            )
        except StorageError as exc:
            logger.error("document_storage_failed", error=str(exc))
            raise ServiceUnavailableError("File storage is temporarily unavailable.") from exc

        document = self._documents.add(
            Document(
                id=document_id,
                owner_id=owner.id,
                filename=filename,
                content_type=detected.content_type,
                extension=detected.extension,
                size_bytes=len(data),
                sha256=sha256,
                storage_key=storage_key,
                status=DocumentStatus.UPLOADED,
            )
        )
        try:
            await self._session.commit()
        except IntegrityError as exc:  # concurrent upload of the same file
            await self._session.rollback()
            await self._delete_object_quietly(storage_key)
            raise ConflictError(
                "This file has already been uploaded.", code="duplicate_document"
            ) from exc

        logger.info(
            "document_uploaded",
            document_id=str(document.id),
            size_bytes=document.size_bytes,
            extension=document.extension,
        )
        await self._schedule_processing(document)
        return document

    # ------------------------------------------------------------------- reads
    async def list(
        self, owner: User, *, limit: int, offset: int, status: DocumentStatus | None
    ) -> tuple[list[Document], int]:
        return await self._documents.list_owned(owner.id, limit=limit, offset=offset, status=status)

    async def get(self, owner: User, document_id: uuid.UUID) -> Document:
        document = await self._documents.get_owned(owner.id, document_id)
        if document is None:
            # Same answer whether the document does not exist or belongs to someone else.
            raise NotFoundError("Document not found.", code="document_not_found")
        return document

    async def get_chunk(
        self, owner: User, document_id: uuid.UUID, chunk_id: uuid.UUID
    ) -> DocumentChunk:
        chunk = await self._chunks.get_owned(owner.id, document_id, chunk_id)
        if chunk is None:
            raise NotFoundError("Passage not found.", code="chunk_not_found")
        return chunk

    async def open_file(self, owner: User, document_id: uuid.UUID) -> FileDownload:
        document = await self.get(owner, document_id)
        iterator = self._storage.iter_object(document.storage_key)
        try:
            first = await run_in_threadpool(next, iterator, b"")
        except ObjectNotFoundError as exc:
            raise NotFoundError("The file is no longer available.", code="file_missing") from exc
        except StorageError as exc:
            raise ServiceUnavailableError("File storage is temporarily unavailable.") from exc

        def body() -> Iterator[bytes]:
            if first:
                yield first
            yield from iterator

        return FileDownload(
            filename=document.filename,
            content_type=document.content_type,
            size_bytes=document.size_bytes,
            body=body(),
        )

    # ----------------------------------------------------------------- changes
    async def delete(self, owner: User, document_id: uuid.UUID) -> None:
        document = await self.get(owner, document_id)
        storage_key = document.storage_key
        await self._documents.delete_owned(owner.id, document_id)  # chunks cascade
        await self._session.commit()
        logger.info("document_deleted", document_id=str(document_id))

        try:
            await run_in_threadpool(self._dispatcher.cleanup_document, document_id, storage_key)
        except TaskDispatchError:
            logger.warning("cleanup_dispatch_failed_inline_fallback", document_id=str(document_id))
            await self._delete_object_quietly(storage_key)
            await self._delete_vectors_quietly(document_id)

    async def reprocess(self, owner: User, document_id: uuid.UUID) -> Document:
        document = await self.get(owner, document_id)
        if document.status != DocumentStatus.FAILED:
            raise ConflictError(
                "Only documents whose processing failed can be reprocessed.",
                code="invalid_document_status",
            )
        document.status = DocumentStatus.UPLOADED
        document.error_message = None
        await self._session.commit()
        await self._schedule_processing(document)
        return document

    # ----------------------------------------------------------------- helpers
    async def _read_bounded(self, upload: UploadFile) -> tuple[bytes, str]:
        limit = self._settings.max_upload_size_bytes
        if upload.size is not None and upload.size > limit:
            raise self._too_large()
        digest = hashlib.sha256()
        buffer = bytearray()
        while chunk := await upload.read(READ_CHUNK_SIZE):
            buffer.extend(chunk)
            if len(buffer) > limit:
                raise self._too_large()
            digest.update(chunk)
        return bytes(buffer), digest.hexdigest()

    def _too_large(self) -> PayloadTooLargeError:
        return PayloadTooLargeError(
            f"The file exceeds the maximum size of {self._settings.max_upload_size_mb} MB."
        )

    async def _schedule_processing(self, document: Document) -> None:
        try:
            task_id = await run_in_threadpool(self._dispatcher.process_document, document.id)
        except TaskDispatchError as exc:
            logger.error("processing_dispatch_failed", document_id=str(document.id))
            document.status = DocumentStatus.FAILED
            document.error_message = SCHEDULING_FAILED_MESSAGE
            await self._session.commit()
            raise ServiceUnavailableError(SCHEDULING_FAILED_MESSAGE) from exc
        if task_id:
            document.celery_task_id = task_id
            await self._session.commit()

    async def _delete_object_quietly(self, storage_key: str) -> None:
        try:
            await run_in_threadpool(self._storage.delete_object, storage_key)
        except StorageError:
            logger.warning("storage_delete_failed", storage_key=storage_key)

    async def _delete_vectors_quietly(self, document_id: uuid.UUID) -> None:
        # Orphan vectors are never served (results are re-checked against PostgreSQL).
        try:
            await run_in_threadpool(self._vector_store.delete_document, document_id)
        except VectorStoreError:
            logger.warning("vector_delete_failed", document_id=str(document_id))
