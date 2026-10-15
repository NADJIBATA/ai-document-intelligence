"""MinioStorage against a real S3-compatible server (``minio-test`` in docker-compose.test.yml).

Skipped when the server is not reachable, unless ``REQUIRE_MINIO=1``.
"""

from __future__ import annotations

import io
import os
import uuid

import pytest

pytest.importorskip("minio")

from app.services.storage_service import MinioStorage, ObjectNotFoundError
from tests.conftest import build_settings


@pytest.fixture(scope="module")
def minio_storage() -> MinioStorage:
    settings = build_settings(
        minio_endpoint=os.environ.get("TEST_MINIO_ENDPOINT", "127.0.0.1:9002"),
        minio_access_key=os.environ.get("TEST_MINIO_ACCESS_KEY", "minioadmin"),
        minio_secret_key=os.environ.get("TEST_MINIO_SECRET_KEY", "minioadmin"),
        minio_bucket=f"test-{uuid.uuid4().hex[:8]}",
    )
    storage = MinioStorage(settings)
    try:
        storage.ensure_bucket()
    except Exception as exc:
        if os.environ.get("REQUIRE_MINIO") == "1":
            pytest.fail(f"MinIO is not reachable: {exc}")
        pytest.skip(f"MinIO is not reachable: {exc}")
    return storage


def test_round_trip(minio_storage: MinioStorage) -> None:
    key = f"users/{uuid.uuid4()}/documents/{uuid.uuid4()}/original.pdf"
    payload = os.urandom(3 * 1024 * 1024 + 17)  # several stream chunks
    minio_storage.put_object(key, io.BytesIO(payload), len(payload), "application/pdf")

    assert minio_storage.get_object(key) == payload
    assert b"".join(minio_storage.iter_object(key, chunk_size=1024 * 1024)) == payload

    minio_storage.delete_object(key)
    with pytest.raises(ObjectNotFoundError):
        minio_storage.get_object(key)


def test_ping_and_idempotent_bucket_creation(minio_storage: MinioStorage) -> None:
    minio_storage.ensure_bucket()
    minio_storage.ping()


def test_deleting_a_missing_object_is_not_an_error(minio_storage: MinioStorage) -> None:
    minio_storage.delete_object("does/not/exist")


def test_missing_bucket_is_created_by_the_first_upload_and_by_ping(
    minio_storage: MinioStorage,
) -> None:
    # Containers start together: MinIO may not accept connections yet when the API
    # starts, so a missing bucket must be created later instead of failing every upload.
    settings = build_settings(
        minio_endpoint=os.environ.get("TEST_MINIO_ENDPOINT", "127.0.0.1:9002"),
        minio_access_key=os.environ.get("TEST_MINIO_ACCESS_KEY", "minioadmin"),
        minio_secret_key=os.environ.get("TEST_MINIO_SECRET_KEY", "minioadmin"),
        minio_bucket=f"late-{uuid.uuid4().hex[:8]}",
    )
    payload = b"created lazily"
    storage = MinioStorage(settings)
    storage.put_object("a/b.txt", io.BytesIO(payload), len(payload), "text/plain")
    assert storage.get_object("a/b.txt") == payload

    pinged = MinioStorage(
        settings.model_copy(update={"minio_bucket": f"ping-{uuid.uuid4().hex[:8]}"})
    )
    pinged.ping()
    pinged.ping()
