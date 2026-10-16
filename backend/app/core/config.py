"""Application configuration.

All settings are read from environment variables (and an optional ``.env`` file).
Nothing secret is hard-coded: secrets are typed as ``SecretStr`` so that they are
never rendered by ``repr()`` or accidentally written to the logs.
"""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.core.rate_limit import RateLimitRule

_INSECURE_JWT_SECRETS = {"", "change-me", "changeme", "secret", "please-change-me"}


def _split_csv(value: object) -> object:
    """Accept either a JSON-less comma separated string or a real list."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Typed, validated configuration for the backend and the Celery worker."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "AI Document Intelligence Platform"
    app_version: str = "0.1.0"
    app_env: Literal["development", "test", "production"] = "development"
    service_name: str = "backend"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    enable_docs: bool = True
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost", "http://localhost:5173"]
    )

    # ------------------------------------------------------------- database
    database_url: str = "postgresql+psycopg://docintel:docintel@127.0.0.1:5432/docintel"
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_echo: bool = False

    # ---------------------------------------------------------------- redis
    redis_url: str = "redis://127.0.0.1:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # --------------------------------------------------------------- qdrant
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_prefix: str = "chunks"

    # ---------------------------------------------------------- minio / s3
    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: SecretStr = SecretStr("minioadmin")
    minio_secret_key: SecretStr = SecretStr("minioadmin")
    minio_bucket: str = "documents"
    minio_secure: bool = False
    minio_region: str | None = None

    # ----------------------------------------------------------------- auth
    jwt_secret: SecretStr = SecretStr("change-me")
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    jwt_issuer: str = "ai-document-intelligence"
    access_token_expire_minutes: int = Field(default=15, ge=1, le=24 * 60)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=90)
    refresh_cookie_name: str = "refresh_token"
    cookie_secure: bool = True
    cookie_samesite: Literal["lax", "strict", "none"] = "strict"

    # --------------------------------------------------------------- upload
    max_upload_size_mb: int = Field(default=50, ge=1, le=1024)
    allowed_extensions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["pdf", "txt", "docx"]
    )

    # ------------------------------------------------------------- chunking
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    chunk_min_size: int = Field(default=100, ge=0, le=2000)

    # ----------------------------------------------------------- embeddings
    embedding_provider: Literal["vllm", "openai", "fake"] = "vllm"
    embedding_base_url: str = "http://vllm-embedding:8000/v1"
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    # Output size of the model. It is part of the Qdrant collection name and every
    # vector returned by the embedding server is checked against it.
    embedding_dimension: int = Field(default=1024, ge=8, le=8192)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    embedding_query_instruction: str = (
        "Given a user question, retrieve relevant passages from the user's documents "
        "that answer the question"
    )

    # ------------------------------------------------------------- reranker
    reranker_provider: Literal["vllm", "none", "fake"] = "vllm"
    reranker_base_url: str = "http://vllm-reranker:8000"
    reranker_api_key: SecretStr | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_timeout_seconds: float = Field(default=60.0, gt=0)
    rerank_min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # ------------------------------------------------------------ retrieval
    bm25_top_k: int = Field(default=30, ge=1, le=500)
    dense_top_k: int = Field(default=30, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1)
    rrf_max_candidates: int = Field(default=60, ge=1, le=1000)
    rerank_top_k: int = Field(default=5, ge=1, le=100)
    bm25_languages: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["fr", "en"])
    bm25_cache_max_users: int = Field(default=64, ge=1)

    # ------------------------------------------------------------------ llm
    llm_provider: Literal["openai", "ollama", "vllm", "fake"] = "openai"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=16, le=32768)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    # Token budget of structured outputs (document insights, comparisons), which are
    # longer than a chat answer.
    llm_structured_max_tokens: int = Field(default=3000, ge=256, le=32768)

    # ----------------------------------------------------------------- chat
    chat_history_turns: int = Field(default=4, ge=0, le=20)
    max_question_chars: int = Field(default=2000, ge=10, le=20000)

    # --------------------------------------------------------- intelligence
    intelligence_enabled: bool = True
    intelligence_max_chars_per_call: int = Field(default=12000, ge=1000)
    intelligence_max_windows: int = Field(default=8, ge=1, le=50)

    # ----------------------------------------------------------- comparison
    comparison_passages_per_aspect: int = Field(default=3, ge=1, le=10)
    comparison_max_passages_per_document: int = Field(default=16, ge=2, le=60)

    # ----------------------------------------------------------- rate limit
    rate_limit_enabled: bool = True
    rate_limit_storage: Literal["redis", "memory"] = "redis"
    rate_limit_register: str = "5/minute"
    rate_limit_login: str = "10/minute"
    rate_limit_refresh: str = "30/minute"
    rate_limit_upload: str = "20/minute"
    rate_limit_chat: str = "30/minute"
    rate_limit_search: str = "60/minute"
    # Jobs that call the LLM several times (insight regeneration, comparisons).
    rate_limit_analysis: str = "10/minute"

    # ------------------------------------------------------------ monitoring
    # GET /metrics (Prometheus). Only reachable on the internal network (nginx blocks it).
    metrics_enabled: bool = True

    # ------------------------------------------------------------ validators
    @field_validator("cors_origins", "allowed_extensions", "bm25_languages", mode="before")
    @classmethod
    def _parse_csv_lists(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("allowed_extensions", "bm25_languages")
    @classmethod
    def _lowercase_items(cls, value: list[str]) -> list[str]:
        return [item.lower().lstrip(".") for item in value]

    @field_validator("database_url")
    @classmethod
    def _check_database_url(cls, value: str) -> str:
        if not value.startswith("postgresql+psycopg://"):
            raise ValueError(
                "DATABASE_URL must use the psycopg 3 driver: postgresql+psycopg://user:pass@host/db"
            )
        return value

    @field_validator(
        "rate_limit_register",
        "rate_limit_login",
        "rate_limit_refresh",
        "rate_limit_upload",
        "rate_limit_chat",
        "rate_limit_search",
        "rate_limit_analysis",
    )
    @classmethod
    def _check_rate_limit(cls, value: str) -> str:
        RateLimitRule.parse(value)
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be strictly smaller than CHUNK_SIZE")
        if self.chunk_min_size >= self.chunk_size:
            raise ValueError("CHUNK_MIN_SIZE must be strictly smaller than CHUNK_SIZE")
        if self.rerank_top_k > self.rrf_max_candidates:
            raise ValueError("RERANK_TOP_K must be <= RRF_MAX_CANDIDATES")
        if self.cookie_samesite == "none" and not self.cookie_secure:
            raise ValueError("COOKIE_SAMESITE=none requires COOKIE_SECURE=true")
        if self.celery_broker_url is None:
            self.celery_broker_url = self.redis_url
        if self.celery_result_backend is None:
            self.celery_result_backend = self.redis_url
        if self.app_env == "production":
            secret = self.jwt_secret.get_secret_value()
            if secret.lower() in _INSECURE_JWT_SECRETS or len(secret) < 32:
                raise ValueError(
                    "JWT_SECRET must be a random string of at least 32 characters in production"
                )
            if not self.cookie_secure:
                raise ValueError("COOKIE_SECURE must be true in production")
            if self.minio_secret_key.get_secret_value() == "minioadmin":
                raise ValueError("MINIO_SECRET_KEY must be changed in production")
            if self.llm_provider == "openai" and self.llm_api_key is None:
                raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai")
        return self

    # ------------------------------------------------------------ helpers
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def access_token_ttl(self) -> timedelta:
        return timedelta(minutes=self.access_token_expire_minutes)

    @property
    def refresh_token_ttl(self) -> timedelta:
        return timedelta(days=self.refresh_token_expire_days)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (read once at startup)."""
    return Settings()
