"""Integration fixtures: real PostgreSQL (migrated with Alembic) and real Redis.

Start the dependencies with::

    docker compose -f docker-compose.test.yml up -d

When they are not reachable the integration tests are *skipped* locally, but they
*fail* when ``REQUIRE_INTEGRATION=1`` (set in CI) so that a broken setup is never hidden.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import redis
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.base import Base
from app.db.session import create_sync_session_factory
from app.llm.fake import FakeLLMProvider
from app.main import create_app
from app.rag.embeddings import EmbeddingProvider
from app.rag.factory import RetrievalComponents, build_embedder, build_retrieval
from app.rag.vector_store import InMemoryVectorStore
from app.repositories.corpus import PostgresChunkStore
from app.services.comparison_service import ComparisonJobService, build_document_comparator
from app.services.indexing_service import ChunkIndexer
from app.services.ingestion_service import IngestionService, build_document_processor
from app.services.intelligence_service import IntelligenceService, build_insight_extractor
from tests.conftest import build_settings
from tests.fakes import InlineDispatcher, InMemoryStorage

BACKEND_DIR = Path(__file__).resolve().parents[2]

# psycopg's async driver cannot run on Windows' default ProactorEventLoop.
TESTCLIENT_BACKEND_OPTIONS: dict[str, Any] = (
    {"loop_factory": asyncio.SelectorEventLoop} if sys.platform == "win32" else {}
)
DEFAULT_PASSWORD = "S3cure-password"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "integration" in item.path.parts:
            item.add_marker(pytest.mark.integration)


_REDIS_CLIENTS: dict[str, redis.Redis] = {}


def _redis_client(settings: Settings) -> redis.Redis:
    """One shared connection for fixture housekeeping (a new one per test is slow)."""
    if settings.redis_url not in _REDIS_CLIENTS:
        _REDIS_CLIENTS[settings.redis_url] = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=3, socket_timeout=5
        )
    return _REDIS_CLIENTS[settings.redis_url]


def _unavailable(reason: str) -> None:
    if os.environ.get("REQUIRE_INTEGRATION") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def integration_settings() -> Settings:
    return build_settings()


@pytest.fixture(scope="session")
def db_engine(integration_settings: Settings) -> Iterator[Engine]:
    engine = create_engine(
        integration_settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError as exc:
        engine.dispose()
        _unavailable(f"PostgreSQL is not reachable for integration tests: {exc.orig}")
    try:
        _redis_client(integration_settings).ping()
    except redis.RedisError as exc:
        engine.dispose()
        _unavailable(f"Redis is not reachable for integration tests: {exc}")

    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.attributes["database_url"] = integration_settings.database_url
    config.attributes["configure_logger"] = False
    command.downgrade(config, "base")
    command.upgrade(config, "head")

    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def _clean_database(request: pytest.FixtureRequest) -> Iterator[None]:
    yield
    if "db_engine" not in request.fixturenames:
        return
    engine: Engine = request.getfixturevalue("db_engine")
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    _redis_client(request.getfixturevalue("integration_settings")).flushdb()


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def session_factory(db_engine: Engine) -> sessionmaker[Session]:
    return create_sync_session_factory(db_engine)


@pytest.fixture
def embedder(integration_settings: Settings) -> EmbeddingProvider:
    return build_embedder(integration_settings)  # EMBEDDING_PROVIDER=fake


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Shared by the "worker" (ingestion) and the API, like Qdrant in production."""
    return InMemoryVectorStore()


@pytest.fixture
def indexer(embedder: EmbeddingProvider, vector_store: InMemoryVectorStore) -> ChunkIndexer:
    return ChunkIndexer(embedder, vector_store)


@pytest.fixture
def ingestion_service(
    session_factory: sessionmaker[Session],
    storage: InMemoryStorage,
    integration_settings: Settings,
    indexer: ChunkIndexer,
) -> IngestionService:
    return IngestionService(
        session_factory, storage, build_document_processor(integration_settings), indexer
    )


@pytest.fixture
def llm() -> FakeLLMProvider:
    """Deterministic model; ``llm.calls`` exposes the prompts it received."""
    return FakeLLMProvider()


@pytest.fixture
def retrieval(
    integration_settings: Settings,
    session_factory: sessionmaker[Session],
    embedder: EmbeddingProvider,
    vector_store: InMemoryVectorStore,
) -> Iterator[RetrievalComponents]:
    """Real pipeline (PostgreSQL corpus, BM25, RRF) with the fake model servers."""
    components = build_retrieval(
        integration_settings, session_factory, embedder=embedder, vector_store=vector_store
    )
    yield components
    components.close()


@pytest.fixture
def intelligence_service(
    session_factory: sessionmaker[Session], llm: FakeLLMProvider, integration_settings: Settings
) -> IntelligenceService:
    return IntelligenceService(session_factory, build_insight_extractor(llm, integration_settings))


@pytest.fixture
def comparison_jobs(
    session_factory: sessionmaker[Session],
    llm: FakeLLMProvider,
    retrieval: RetrievalComponents,
    integration_settings: Settings,
) -> ComparisonJobService:
    comparator = build_document_comparator(
        llm, retrieval.pipeline, PostgresChunkStore(session_factory), integration_settings
    )
    return ComparisonJobService(session_factory, comparator)


@pytest.fixture
def dispatcher(
    ingestion_service: IngestionService,
    storage: InMemoryStorage,
    vector_store: InMemoryVectorStore,
    intelligence_service: IntelligenceService,
    comparison_jobs: ComparisonJobService,
) -> InlineDispatcher:
    """Jobs run synchronously inside the request, like Celery's eager mode."""
    return InlineDispatcher(
        ingestion_service, storage, vector_store, intelligence_service, comparison_jobs
    )


@pytest.fixture
def make_client(
    db_engine: Engine,
    storage: InMemoryStorage,
    dispatcher: InlineDispatcher,
    session_factory: sessionmaker[Session],
    embedder: EmbeddingProvider,
    vector_store: InMemoryVectorStore,
    llm: FakeLLMProvider,
) -> Iterator[Any]:
    """Factory returning started TestClients (lifespan executed), closed after the test.

    ``storage_backend`` / ``task_dispatcher`` / ``retrieval`` / ``llm_backend`` keyword arguments
    replace the default fakes; other keyword arguments override settings.
    """
    clients: list[TestClient] = []
    owned: list[RetrievalComponents] = []

    def factory(
        *,
        storage_backend: Any = None,
        task_dispatcher: Any = None,
        retrieval: RetrievalComponents | None = None,
        llm_backend: Any = None,
        **overrides: Any,
    ) -> TestClient:
        settings = build_settings(**overrides)
        if retrieval is None:
            retrieval = build_retrieval(
                settings, session_factory, embedder=embedder, vector_store=vector_store
            )
            owned.append(retrieval)
        app = create_app(
            settings,
            storage=storage_backend or storage,
            task_dispatcher=task_dispatcher or dispatcher,
            retrieval=retrieval,
            llm=llm_backend if llm_backend is not None else llm,
        )
        client = TestClient(app, backend_options=TESTCLIENT_BACKEND_OPTIONS)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)
    for components in owned:
        components.close()


@pytest.fixture
def client(make_client: Any) -> TestClient:
    result: TestClient = make_client()
    return result


def register_user(
    client: TestClient, email: str, password: str = DEFAULT_PASSWORD
) -> dict[str, Any]:
    response = client.post("/api/auth/register", json={"email": email, "password": password})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def login_user(client: TestClient, email: str, password: str = DEFAULT_PASSWORD) -> dict[str, Any]:
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def auth_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def signup(client: TestClient, email: str) -> dict[str, str]:
    """Register + log in; returns the Authorization header."""
    register_user(client, email)
    return auth_headers(login_user(client, email)["access_token"])


def upload(
    client: TestClient,
    headers: dict[str, str],
    filename: str,
    content: bytes,
    content_type: str = "application/octet-stream",
) -> Any:
    return client.post(
        "/api/documents", headers=headers, files={"file": (filename, content, content_type)}
    )
