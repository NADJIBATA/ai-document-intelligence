"""Small helper for calls to model servers (embeddings, reranker, LLM): retries with
exponential backoff on network errors, 429 and 5xx responses."""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
CONNECT_TIMEOUT_SECONDS = 5.0


def client_timeout(total_seconds: float) -> httpx.Timeout:
    """Generous read timeout (model inference), short connect timeout (server down)."""
    return httpx.Timeout(total_seconds, connect=min(CONNECT_TIMEOUT_SECONDS, total_seconds))


class UpstreamError(Exception):
    """A model server / external API call failed after all retries."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    max_retries: int = 2,
    backoff_seconds: float = 0.5,
    service: str = "upstream",
    retry_read_timeout: bool = True,
) -> dict[str, Any]:
    """POST ``payload`` and return the decoded JSON object, or raise ``UpstreamError``.

    ``retry_read_timeout=False`` is meant for slow generative calls: when the server did
    not answer within the (long) timeout, asking again only multiplies the wait.
    """
    last_error = "unknown error"
    status_code: int | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, json=payload)
        except httpx.ReadTimeout as exc:
            last_error, status_code = type(exc).__name__, None
            if not retry_read_timeout:
                break
        except httpx.HTTPError as exc:
            last_error, status_code = type(exc).__name__, None
        else:
            if response.status_code < 400:
                data = response.json()
                if not isinstance(data, dict):
                    raise UpstreamError(f"{service}: unexpected response format")
                return data
            status_code = response.status_code
            # Authentication errors may echo (part of) the API key: never keep the body.
            detail = "" if status_code in (401, 403) else f": {response.text[:200]}"
            last_error = f"HTTP {status_code}{detail}"
            if status_code not in RETRYABLE_STATUS:
                break
        if attempt < max_retries:
            logger.warning("upstream_retry", service=service, attempt=attempt + 1, error=last_error)
            time.sleep(backoff_seconds * 2**attempt)
    raise UpstreamError(f"{service}: {last_error}", status_code=status_code)
