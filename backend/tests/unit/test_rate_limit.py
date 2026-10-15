from __future__ import annotations

import asyncio

import pytest

from app.core.exceptions import RateLimitExceededError
from app.core.rate_limit import InMemoryRateLimitBackend, RateLimiter, RateLimitRule


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5/minute", RateLimitRule(5, 60)),
        ("10/second", RateLimitRule(10, 1)),
        ("100/hour", RateLimitRule(100, 3600)),
        (" 3 / day ", RateLimitRule(3, 86400)),
    ],
)
def test_parse(value: str, expected: RateLimitRule) -> None:
    assert RateLimitRule.parse(value) == expected


@pytest.mark.parametrize("value", ["", "5", "five/minute", "0/minute", "5/week"])
def test_parse_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        RateLimitRule.parse(value)


def test_limit_is_enforced_per_identifier() -> None:
    limiter = RateLimiter(InMemoryRateLimitBackend())
    rule = RateLimitRule(limit=2, window_seconds=3600)

    async def scenario() -> None:
        await limiter.check("login", "1.1.1.1", rule)
        await limiter.check("login", "1.1.1.1", rule)
        with pytest.raises(RateLimitExceededError) as exc_info:
            await limiter.check("login", "1.1.1.1", rule)
        assert exc_info.value.headers is not None
        assert int(exc_info.value.headers["Retry-After"]) >= 1
        # Another client is not affected.
        await limiter.check("login", "2.2.2.2", rule)

    asyncio.run(scenario())


def test_disabled_limiter_never_raises() -> None:
    limiter = RateLimiter(InMemoryRateLimitBackend(), enabled=False)
    rule = RateLimitRule(limit=1, window_seconds=60)

    async def scenario() -> None:
        for _ in range(5):
            await limiter.check("login", "1.1.1.1", rule)

    asyncio.run(scenario())
