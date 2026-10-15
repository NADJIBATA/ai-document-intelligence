"""Shared fixtures for unit and integration tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from app.core.config import Settings

TEST_JWT_SECRET = "test-secret-" + "x" * 40

# Defaults used by every test; integration tests override the URLs from the environment.
# 127.0.0.1 (not "localhost"): on Windows, localhost resolves to ::1 first and every new
# connection to a Docker-published port then waits ~2 s before falling back to IPv4.
BASE_TEST_SETTINGS: dict[str, Any] = {
    "app_env": "test",
    "log_json": True,
    "log_level": "WARNING",
    "jwt_secret": TEST_JWT_SECRET,
    "cookie_secure": False,
    "rate_limit_storage": "memory",
    # No model server in tests: deterministic local stand-ins.
    "embedding_provider": "fake",
    "embedding_dimension": 256,
    "reranker_provider": "fake",
    "llm_provider": "fake",
    "database_url": os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+psycopg://docintel:docintel@127.0.0.1:5433/docintel_test",
    ),
    "redis_url": os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6380/0"),
}


def build_settings(**overrides: Any) -> Settings:
    """Create isolated settings (the .env file of the developer is ignored)."""
    values = {**BASE_TEST_SETTINGS, **overrides}
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def settings_factory() -> Callable[..., Settings]:
    return build_settings


@pytest.fixture
def settings() -> Settings:
    return build_settings()
