"""Fixed-window rate limiting backed by Redis (or memory for tests / single process).

A rule is expressed as ``"<count>/<period>"``, e.g. ``"10/minute"``.
The limiter fails *open* when Redis is unreachable: availability of the API is
preferred over strict throttling, and the incident is logged.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import structlog
from fastapi import Request
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.exceptions import RateLimitExceededError

logger = structlog.get_logger(__name__)

_PERIODS: dict[str, int] = {
    "second": 1,
    "seconds": 1,
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
    "day": 86400,
    "days": 86400,
}


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    limit: int
    window_seconds: int

    @classmethod
    def parse(cls, value: str) -> RateLimitRule:
        try:
            count_str, period = value.strip().lower().split("/", maxsplit=1)
            count = int(count_str)
            window = _PERIODS[period.strip()]
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Invalid rate limit '{value}'. Expected '<count>/<second|minute|hour|day>'."
            ) from exc
        if count < 1:
            raise ValueError(f"Invalid rate limit '{value}': count must be >= 1")
        return cls(limit=count, window_seconds=window)


class RateLimitBackend(Protocol):
    async def increment(self, key: str, window_seconds: int) -> int:
        """Increment the counter for ``key`` and return its new value."""
        ...


class InMemoryRateLimitBackend:
    """Process-local backend. Suitable for tests and single-process development."""

    def __init__(self) -> None:
        self._counters: dict[str, tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    async def increment(self, key: str, window_seconds: int) -> int:
        async with self._lock:
            now = time.monotonic()
            count, expires_at = self._counters.get(key, (0, now + window_seconds))
            if expires_at <= now:
                count, expires_at = 0, now + window_seconds
            count += 1
            self._counters[key] = (count, expires_at)
            if len(self._counters) > 10_000:
                self._counters = {k: v for k, v in self._counters.items() if v[1] > now}
            return count


class RedisRateLimitBackend:
    """Shared backend: counters are visible to every API replica."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def increment(self, key: str, window_seconds: int) -> int:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, window_seconds)
            count, _ = await pipe.execute()
        return int(count)


class RateLimiter:
    def __init__(
        self, backend: RateLimitBackend, *, enabled: bool = True, prefix: str = "ratelimit"
    ) -> None:
        self._backend = backend
        self._enabled = enabled
        self._prefix = prefix

    async def check(self, scope: str, identifier: str, rule: RateLimitRule) -> None:
        """Raise :class:`RateLimitExceededError` when ``identifier`` exceeded ``rule``."""
        if not self._enabled:
            return
        now = time.time()
        window_index = math.floor(now / rule.window_seconds)
        key = f"{self._prefix}:{scope}:{identifier}:{window_index}"
        try:
            count = await self._backend.increment(key, rule.window_seconds)
        except RedisError:
            logger.warning("rate_limit_backend_unavailable", scope=scope)
            return
        if count > rule.limit:
            retry_after = max(1, math.ceil((window_index + 1) * rule.window_seconds - now))
            logger.info("rate_limit_exceeded", scope=scope, limit=rule.limit)
            raise RateLimitExceededError(retry_after=retry_after)


def client_ip(request: Request) -> str:
    """Client address as seen by the ASGI server.

    Uvicorn runs with ``--proxy-headers`` behind nginx, so ``request.client`` already
    contains the real client IP taken from ``X-Forwarded-For`` by a trusted proxy.
    """
    return request.client.host if request.client else "unknown"


def rate_limit(
    scope: str, setting_name: str, key: Literal["ip", "user"] = "ip"
) -> Callable[[Request], Awaitable[None]]:
    """Build a FastAPI dependency enforcing the rule stored in ``Settings.<setting_name>``.

    ``key="user"`` must be used on routes where the current user dependency runs first
    (it stores ``request.state.user_id``); it falls back to the IP otherwise.
    """

    async def dependency(request: Request) -> None:
        settings = request.app.state.settings
        limiter: RateLimiter = request.app.state.rate_limiter
        rule = RateLimitRule.parse(getattr(settings, setting_name))
        identifier = client_ip(request)
        if key == "user":
            identifier = getattr(request.state, "user_id", None) or identifier
        await limiter.check(scope, identifier, rule)

    return dependency
