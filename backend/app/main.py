"""FastAPI application factory.

Run with: ``uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.concurrency import run_in_threadpool

from app.api.router import api_router
from app.api.routes.health import ReadinessCheck
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.metrics import APP_INFO
from app.core.middleware import REQUEST_ID_HEADER, MetricsMiddleware, RequestContextMiddleware
from app.core.rate_limit import (
    InMemoryRateLimitBackend,
    RateLimitBackend,
    RateLimiter,
    RedisRateLimitBackend,
)
from app.db.session import (
    create_async_db_engine,
    create_async_session_factory,
    create_sync_db_engine,
    create_sync_session_factory,
)
from app.llm.base import LLMProvider
from app.llm.factory import build_llm
from app.rag.factory import RetrievalComponents, build_retrieval
from app.rag.vector_store import VectorStore, VectorStoreError
from app.services.metrics_service import (
    Collector,
    database_collector,
    job_collector,
    queue_collector,
)
from app.services.storage_service import MinioStorage, ObjectStorage, StorageError
from app.services.task_dispatcher import (
    ANALYSIS_QUEUE,
    INGESTION_QUEUE,
    CeleryTaskDispatcher,
    TaskDispatcher,
)

logger = structlog.get_logger(__name__)

API_DESCRIPTION = """
Upload documents, index them and query them in natural language with a hybrid
(BM25 + dense) retrieval pipeline, neural reranking and cited LLM answers.

**Authentication**: call `POST /api/auth/login`, then click *Authorize* and paste the
`access_token`. Every document, conversation and search result is scoped to the
authenticated user.
"""

OPENAPI_TAGS = [
    {"name": "auth", "description": "Registration, login and token rotation."},
    {"name": "documents", "description": "Upload, processing status, download, deletion."},
    {"name": "search", "description": "Hybrid retrieval (BM25 + dense, RRF, reranking)."},
    {"name": "chat", "description": "Answers grounded in my documents, with cited sources."},
    {"name": "conversations", "description": "Conversation history."},
    {"name": "intelligence", "description": "Structured information extracted from documents."},
    {"name": "comparisons", "description": "Differences between two documents, with sources."},
    {"name": "dashboard", "description": "Figures of the authenticated user."},
    {"name": "health", "description": "Liveness and readiness probes."},
]


def _database_check(engine: AsyncEngine) -> ReadinessCheck:
    async def check() -> None:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    return check


def _redis_check(redis: Redis) -> ReadinessCheck:
    async def check() -> None:
        # redis-py types ping() as "Awaitable[bool] | bool" for both clients.
        result = redis.ping()
        if inspect.isawaitable(result):
            await result

    return check


def _storage_check(storage: ObjectStorage) -> ReadinessCheck:
    async def check() -> None:
        await run_in_threadpool(storage.ping)

    return check


def _vector_store_check(vector_store: VectorStore) -> ReadinessCheck:
    async def check() -> None:
        await run_in_threadpool(vector_store.ping)

    return check


def create_app(
    settings: Settings | None = None,
    *,
    storage: ObjectStorage | None = None,
    task_dispatcher: TaskDispatcher | None = None,
    retrieval: RetrievalComponents | None = None,
    llm: LLMProvider | None = None,
) -> FastAPI:
    """Build the application.

    ``storage``, ``task_dispatcher``, ``retrieval`` and ``llm`` can be injected (tests);
    an injected ``retrieval`` is owned, and closed, by the caller.
    """
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_async_db_engine(settings)
        redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
        backend: RateLimitBackend = (
            RedisRateLimitBackend(redis)
            if settings.rate_limit_storage == "redis"
            else InMemoryRateLimitBackend()
        )
        app.state.db_engine = engine
        app.state.session_factory = create_async_session_factory(engine)
        app.state.redis = redis
        app.state.rate_limiter = RateLimiter(backend, enabled=settings.rate_limit_enabled)
        object_storage = storage or MinioStorage(settings)
        try:
            await run_in_threadpool(object_storage.ensure_bucket)
        except StorageError as exc:  # reported by /ready, retried on next start
            logger.warning("storage_bucket_check_failed", error=str(exc))
        app.state.storage = object_storage
        app.state.task_dispatcher = task_dispatcher or CeleryTaskDispatcher()

        # The retrieval engine is synchronous (BM25 is CPU-bound, model servers are
        # called with blocking HTTP clients) and runs in the threadpool: it gets its
        # own small synchronous connection pool.
        sync_engine = None
        components = retrieval
        if components is None:
            sync_engine = create_sync_db_engine(settings, pool_size=5)
            components = build_retrieval(settings, create_sync_session_factory(sync_engine))
            try:
                await run_in_threadpool(
                    components.vector_store.ensure_collection, components.embedder.dimension
                )
            except VectorStoreError as exc:  # reported by /ready, retried when indexing
                logger.warning("vector_store_check_failed", error=str(exc))
        app.state.retrieval = components
        app.state.llm = llm if llm is not None else build_llm(settings)

        # Celery's Redis broker (another database than the application data).
        broker = Redis.from_url(
            settings.celery_broker_url or settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        collectors: dict[str, Collector] = {
            "database": database_collector(app.state.session_factory),
            "queues": queue_collector(broker, (INGESTION_QUEUE, ANALYSIS_QUEUE)),
            "jobs": job_collector(redis),
        }
        app.state.metrics_collectors = collectors
        APP_INFO.set(1, version=settings.app_version, env=settings.app_env)

        app.state.readiness_checks = {
            "database": _database_check(engine),
            "redis": _redis_check(redis),
            "storage": _storage_check(object_storage),
            "vector_store": _vector_store_check(components.vector_store),
        }
        logger.info("application_startup", env=settings.app_env, version=settings.app_version)
        try:
            yield
        finally:
            if sync_engine is not None:
                components.close()
                sync_engine.dispose()
            await broker.aclose()
            await redis.aclose()
            await engine.dispose()
            logger.info("application_shutdown")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
    )
    app.state.settings = settings

    register_exception_handlers(app)

    # Starlette runs the *last added* middleware first: RequestContext wraps everything.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
    )
    if settings.metrics_enabled:
        app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health_router)
    if settings.metrics_enabled:
        app.include_router(metrics_router)
    app.include_router(api_router)
    return app
