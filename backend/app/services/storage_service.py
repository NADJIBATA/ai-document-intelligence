"""Object storage (MinIO or any S3-compatible service).

The rest of the application depends on the :class:`ObjectStorage` protocol only, so
switching to AWS S3, SeaweedFS, Garage... is a configuration change. The MinIO SDK
is synchronous: async callers must run these methods in a thread
(``starlette.concurrency.run_in_threadpool``).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, BinaryIO, Protocol

import structlog

from app.core.config import Settings

if TYPE_CHECKING:
    from minio import Minio

logger = structlog.get_logger(__name__)

STREAM_CHUNK_SIZE = 1024 * 1024


class StorageError(Exception):
    """The storage backend is unavailable or returned an unexpected error (transient)."""


class ObjectNotFoundError(StorageError):
    """The requested object does not exist."""


class ObjectStorage(Protocol):
    def ensure_bucket(self) -> None: ...

    def ping(self) -> None: ...

    def put_object(self, key: str, data: BinaryIO, length: int, content_type: str) -> None: ...

    def get_object(self, key: str) -> bytes: ...

    def iter_object(self, key: str, chunk_size: int = STREAM_CHUNK_SIZE) -> Iterator[bytes]: ...

    def delete_object(self, key: str) -> None: ...


def document_storage_key(owner_id: uuid.UUID, document_id: uuid.UUID, extension: str) -> str:
    """Server-generated key: the user-provided file name is never part of the path."""
    return f"users/{owner_id}/documents/{document_id}/original.{extension}"


class MinioStorage:
    def __init__(self, settings: Settings) -> None:
        # Imported lazily: unit tests and tools that use another backend do not need it.
        import urllib3
        from minio import Minio

        http_client = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=3.0, read=60.0),
            retries=urllib3.Retry(
                total=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504)
            ),
            maxsize=10,
        )
        self._bucket = settings.minio_bucket
        self._client: Minio = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key.get_secret_value(),
            secret_key=settings.minio_secret_key.get_secret_value(),
            secure=settings.minio_secure,
            region=settings.minio_region,
            http_client=http_client,
        )

    def ensure_bucket(self) -> None:
        try:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                logger.info("storage_bucket_created", bucket=self._bucket)
        except Exception as exc:
            raise StorageError(f"Cannot ensure bucket: {type(exc).__name__}") from exc

    def ping(self) -> None:
        """Readiness check. Creates the bucket when it is missing: MinIO may not have been
        ready when the application started (containers starting together)."""
        self.ensure_bucket()

    def put_object(self, key: str, data: BinaryIO, length: int, content_type: str) -> None:
        from minio.error import S3Error

        try:
            self._client.put_object(
                self._bucket, key, data, length=length, content_type=content_type
            )
            return
        except S3Error as exc:
            if exc.code != "NoSuchBucket":
                raise StorageError(f"Upload failed: {exc.code}") from exc
        except Exception as exc:
            raise StorageError(f"Upload failed: {type(exc).__name__}") from exc
        # The bucket could not be created at start-up: create it and retry once.
        self.ensure_bucket()
        data.seek(0)
        try:
            self._client.put_object(
                self._bucket, key, data, length=length, content_type=content_type
            )
        except Exception as exc:
            raise StorageError(f"Upload failed: {type(exc).__name__}") from exc

    def get_object(self, key: str) -> bytes:
        return b"".join(self.iter_object(key))

    def iter_object(self, key: str, chunk_size: int = STREAM_CHUNK_SIZE) -> Iterator[bytes]:
        from minio.error import S3Error

        try:
            response = self._client.get_object(self._bucket, key)
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject"):
                raise ObjectNotFoundError(key) from exc
            raise StorageError(f"Download failed: {exc.code}") from exc
        except Exception as exc:
            raise StorageError(f"Download failed: {type(exc).__name__}") from exc
        try:
            yield from response.stream(chunk_size)
        finally:
            response.close()
            response.release_conn()

    def delete_object(self, key: str) -> None:
        try:
            self._client.remove_object(self._bucket, key)
        except Exception as exc:
            raise StorageError(f"Delete failed: {type(exc).__name__}") from exc
