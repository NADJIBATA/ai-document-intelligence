"""Vector storage.

``QdrantVectorStore`` uses Qdrant's REST API directly (no gRPC client to install).
Multi-tenancy follows Qdrant's recommended design: **one collection, one payload
index on ``user_id`` declared as tenant key**, and a mandatory ``user_id`` filter
applied by Qdrant itself during the search (never as a post-filter).

``InMemoryVectorStore`` implements the same protocol with exact cosine search; it is
used by tests and by the offline evaluation.
"""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import numpy as np

from app.core.http import client_timeout
from app.rag.types import SearchHit

UPSERT_BATCH_SIZE = 128


class VectorStoreError(Exception):
    """The vector database is unavailable or rejected the request."""


@dataclass(frozen=True, slots=True)
class VectorPoint:
    chunk_id: uuid.UUID
    vector: Sequence[float]
    user_id: uuid.UUID
    document_id: uuid.UUID
    payload: dict[str, Any]


class VectorStore(Protocol):
    def ensure_collection(self, dimension: int) -> None: ...

    def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    def search(
        self,
        vector: Sequence[float],
        user_id: uuid.UUID,
        *,
        limit: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[SearchHit]: ...

    def delete_document(self, document_id: uuid.UUID) -> None: ...

    def ping(self) -> None: ...


def collection_name(prefix: str, model_name: str, dimension: int) -> str:
    """One collection per (model, dimension): vectors of different models never mix."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return f"{prefix}__{slug}__{dimension}"


def build_filter(
    user_id: uuid.UUID, document_ids: Sequence[uuid.UUID] | None = None
) -> dict[str, Any]:
    """Qdrant filter restricting a search to one user (and optionally some documents)."""
    must: list[dict[str, Any]] = [{"key": "user_id", "match": {"value": str(user_id)}}]
    if document_ids is not None:
        must.append({"key": "document_id", "match": {"any": [str(d) for d in document_ids]}})
    return {"must": must}


class QdrantVectorStore:
    def __init__(
        self,
        *,
        url: str,
        collection: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._collection = collection
        headers = {"api-key": api_key} if api_key else {}
        self._client = client or httpx.Client(
            base_url=url.rstrip("/"), timeout=client_timeout(timeout_seconds), headers=headers
        )
        self._ready = False
        self._lock = threading.Lock()

    @property
    def collection(self) -> str:
        return self._collection

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise VectorStoreError(f"qdrant: {type(exc).__name__}") from exc

    def _checked(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._request(method, path, **kwargs)
        if response.status_code >= 400:
            raise VectorStoreError(f"qdrant: HTTP {response.status_code}: {response.text[:200]}")
        data: dict[str, Any] = response.json()
        return data

    def ping(self) -> None:
        response = self._request("GET", "/readyz")
        if response.status_code >= 400:
            raise VectorStoreError(f"qdrant: HTTP {response.status_code}")

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection and its payload indexes if needed (idempotent)."""
        with self._lock:
            if self._ready:
                return
            path = f"/collections/{self._collection}"
            existing = self._request("GET", path)
            if existing.status_code == 404:
                created = self._request(
                    "PUT", path, json={"vectors": {"size": dimension, "distance": "Cosine"}}
                )
                # A failure is fine if another process created it in the meantime.
                if created.status_code >= 400 and self._request("GET", path).status_code != 200:
                    raise VectorStoreError(
                        f"qdrant: cannot create collection: {created.text[:200]}"
                    )
            elif existing.status_code >= 400:
                raise VectorStoreError(f"qdrant: HTTP {existing.status_code}")
            else:
                size = existing.json()["result"]["config"]["params"]["vectors"].get("size")
                if size != dimension:
                    raise VectorStoreError(
                        f"qdrant: collection '{self._collection}' has dimension {size}, "
                        f"the embedding model produces {dimension}"
                    )
            self._checked(
                "PUT",
                f"{path}/index",
                params={"wait": "true"},
                json={
                    "field_name": "user_id",
                    "field_schema": {"type": "keyword", "is_tenant": True},
                },
            )
            self._checked(
                "PUT",
                f"{path}/index",
                params={"wait": "true"},
                json={"field_name": "document_id", "field_schema": "keyword"},
            )
            self._ready = True

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            self._checked(
                "PUT",
                f"/collections/{self._collection}/points",
                params={"wait": "true"},
                json={
                    "points": [
                        {
                            "id": str(point.chunk_id),
                            "vector": list(point.vector),
                            "payload": {
                                **point.payload,
                                # Set last: the tenant keys can never be overridden.
                                "user_id": str(point.user_id),
                                "document_id": str(point.document_id),
                                "chunk_id": str(point.chunk_id),
                            },
                        }
                        for point in batch
                    ]
                },
            )

    def search(
        self,
        vector: Sequence[float],
        user_id: uuid.UUID,
        *,
        limit: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[SearchHit]:
        if document_ids is not None and not document_ids:
            return []
        response = self._request(
            "POST",
            f"/collections/{self._collection}/points/query",
            json={
                "query": list(vector),
                # Applied by Qdrant during the search: never a post-filter.
                "filter": build_filter(user_id, document_ids),
                "limit": limit,
                "with_payload": False,
                "with_vector": False,
            },
        )
        if response.status_code == 404:
            return []  # collection not created yet: nothing has been indexed
        if response.status_code >= 400:
            raise VectorStoreError(f"qdrant: HTTP {response.status_code}: {response.text[:200]}")
        try:
            return [
                SearchHit(uuid.UUID(str(point["id"])), float(point["score"]))
                for point in response.json()["result"]["points"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorStoreError("qdrant: malformed search response") from exc

    def delete_document(self, document_id: uuid.UUID) -> None:
        response = self._request(
            "POST",
            f"/collections/{self._collection}/points/delete",
            params={"wait": "true"},
            json={
                "filter": {"must": [{"key": "document_id", "match": {"value": str(document_id)}}]}
            },
        )
        # 404: the collection does not exist yet, so there is nothing to delete.
        if response.status_code >= 400 and response.status_code != 404:
            raise VectorStoreError(f"qdrant: HTTP {response.status_code}: {response.text[:200]}")


class InMemoryVectorStore:
    """Exact cosine search over vectors kept in memory (tests, offline evaluation)."""

    def __init__(self) -> None:
        self._points: dict[uuid.UUID, VectorPoint] = {}
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def ensure_collection(self, dimension: int) -> None:
        if self._dimension not in (None, dimension):
            raise VectorStoreError(
                f"collection has dimension {self._dimension}, model produces {dimension}"
            )
        self._dimension = dimension

    def ping(self) -> None:
        return None

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        with self._lock:
            for point in points:
                self._points[point.chunk_id] = point

    def search(
        self,
        vector: Sequence[float],
        user_id: uuid.UUID,
        *,
        limit: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[SearchHit]:
        allowed = set(document_ids) if document_ids is not None else None
        with self._lock:
            candidates = [
                point
                for point in self._points.values()
                if point.user_id == user_id and (allowed is None or point.document_id in allowed)
            ]
        if not candidates:
            return []
        matrix = np.asarray([point.vector for point in candidates], dtype=np.float64)
        scores = matrix @ np.asarray(vector, dtype=np.float64)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [SearchHit(candidates[i].chunk_id, float(scores[i])) for i in order]

    def delete_document(self, document_id: uuid.UUID) -> None:
        with self._lock:
            self._points = {
                chunk_id: point
                for chunk_id, point in self._points.items()
                if point.document_id != document_id
            }

    def __len__(self) -> int:
        return len(self._points)
