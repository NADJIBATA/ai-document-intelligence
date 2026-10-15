"""Engine and session factories.

The API uses an *async* engine; Celery workers and Alembic use a *sync* engine.
Both use the same driver (psycopg 3) and the same ``DATABASE_URL``.
Engines are created explicitly (no module-level globals) so that each process /
event loop owns its own connection pool.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings

# Every session works in UTC, whatever the server's default TimeZone is, and a
# connection attempt never hangs forever when the database is unreachable.
_CONNECT_ARGS = {"options": "-c timezone=utc", "connect_timeout": 10}


def create_async_db_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args=_CONNECT_ARGS,
        echo=False,
    )


def create_async_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


def create_sync_db_engine(settings: Settings, *, pool_size: int | None = None) -> Engine:
    return create_engine(
        settings.database_url,
        pool_size=pool_size or settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args=_CONNECT_ARGS,
        echo=False,
    )


def create_sync_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False, autoflush=False)
