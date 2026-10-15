"""Document management endpoints. Every route is scoped to the authenticated user."""

from __future__ import annotations

import uuid
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, DocumentServiceDep
from app.core.rate_limit import rate_limit
from app.models.enums import DocumentStatus
from app.schemas.common import ERROR_RESPONSES, ErrorResponse, Page
from app.schemas.document import ChunkRead, DocumentRead, DocumentStatusRead

router = APIRouter(prefix="/documents", tags=["documents"])

NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Document not found"}
}
INLINE_CONTENT_TYPES = {"application/pdf", "text/plain"}


def _content_disposition(filename: str, inline: bool) -> str:
    ascii_name = filename.encode("ascii", "ignore").decode() or "document"
    ascii_name = ascii_name.replace('"', "").replace("\\", "")
    kind = "inline" if inline else "attachment"
    return f"{kind}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentRead,
    summary="Upload a document",
    description=(
        "Uploads a PDF, DOCX or TXT file (multipart/form-data, field `file`). The file is "
        "stored, then processed asynchronously: poll `GET /api/documents/{id}/status` until "
        "the status is `indexed` or `failed`."
    ),
    responses={
        409: {"model": ErrorResponse, "description": "The same file was already uploaded"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported or mismatching file type"},
        503: {"model": ErrorResponse, "description": "Storage or queue unavailable"},
        **ERROR_RESPONSES,
    },
)
async def upload_document(
    current_user: CurrentUser,
    _: Annotated[None, Depends(rate_limit("documents_upload", "rate_limit_upload", key="user"))],
    service: DocumentServiceDep,
    file: Annotated[UploadFile, File(description="PDF, DOCX or TXT file")],
) -> DocumentRead:
    document = await service.upload(current_user, file)
    return DocumentRead.model_validate(document)


@router.get(
    "",
    response_model=Page[DocumentRead],
    summary="List my documents",
    description="Most recent first. Optional filter on the processing status.",
    responses=ERROR_RESPONSES,
)
async def list_documents(
    current_user: CurrentUser,
    service: DocumentServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
) -> Page[DocumentRead]:
    documents, total = await service.list(
        current_user, limit=limit, offset=offset, status=status_filter
    )
    return Page[DocumentRead](
        items=[DocumentRead.model_validate(d) for d in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Get a document",
    responses={**NOT_FOUND, **ERROR_RESPONSES},
)
async def get_document(
    document_id: uuid.UUID, current_user: CurrentUser, service: DocumentServiceDep
) -> DocumentRead:
    return DocumentRead.model_validate(await service.get(current_user, document_id))


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusRead,
    summary="Processing status",
    description="Lightweight endpoint designed for polling: uploaded → processing → "
    "indexed | failed.",
    responses={**NOT_FOUND, **ERROR_RESPONSES},
)
async def get_document_status(
    document_id: uuid.UUID, current_user: CurrentUser, service: DocumentServiceDep
) -> DocumentStatusRead:
    return DocumentStatusRead.model_validate(await service.get(current_user, document_id))


@router.get(
    "/{document_id}/file",
    summary="Download the original file",
    description="Streams the original file. PDF and TXT are served inline so that the "
    "browser can open them (a PDF can be opened at a given page with `#page=N`).",
    response_class=StreamingResponse,
    responses={**NOT_FOUND, **ERROR_RESPONSES},
)
async def download_document(
    document_id: uuid.UUID, current_user: CurrentUser, service: DocumentServiceDep
) -> StreamingResponse:
    download = await service.open_file(current_user, document_id)
    inline = download.content_type in INLINE_CONTENT_TYPES
    return StreamingResponse(
        download.body,
        media_type=download.content_type,
        headers={
            "Content-Disposition": _content_disposition(download.filename, inline),
            "Content-Length": str(download.size_bytes),
            "X-Content-Type-Options": "nosniff",
            # User content must never run scripts in the application's origin.
            "Content-Security-Policy": "sandbox",
            "Cache-Control": "private, max-age=0, no-store",
        },
    )


@router.get(
    "/{document_id}/chunks/{chunk_id}",
    response_model=ChunkRead,
    summary="Get a passage",
    description="Returns one passage of the document, e.g. the target of a citation.",
    responses={
        404: {"model": ErrorResponse, "description": "Passage not found"},
        **ERROR_RESPONSES,
    },
)
async def get_chunk(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    current_user: CurrentUser,
    service: DocumentServiceDep,
) -> ChunkRead:
    return ChunkRead.model_validate(await service.get_chunk(current_user, document_id, chunk_id))


@router.post(
    "/{document_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentRead,
    summary="Retry a failed document",
    responses={
        **NOT_FOUND,
        409: {"model": ErrorResponse, "description": "The document is not in `failed` status"},
        **ERROR_RESPONSES,
    },
)
async def reprocess_document(
    document_id: uuid.UUID, current_user: CurrentUser, service: DocumentServiceDep
) -> DocumentRead:
    return DocumentRead.model_validate(await service.reprocess(current_user, document_id))


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Deletes the document and its passages immediately; the stored file (and "
    "vectors) are removed asynchronously.",
    responses={**NOT_FOUND, **ERROR_RESPONSES},
)
async def delete_document(
    document_id: uuid.UUID, current_user: CurrentUser, service: DocumentServiceDep
) -> Response:
    await service.delete(current_user, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
