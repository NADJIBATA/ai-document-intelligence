from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.conftest import build_settings


def test_defaults_match_the_specification() -> None:
    settings = build_settings()
    assert settings.bm25_top_k == 30
    assert settings.dense_top_k == 30
    assert settings.rrf_k == 60
    assert settings.rerank_top_k == 5
    assert settings.rrf_max_candidates == 60
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150


def test_comma_separated_lists_are_parsed() -> None:
    settings = build_settings(
        cors_origins="http://a.test, http://b.test", allowed_extensions="PDF,.Txt"
    )
    assert settings.cors_origins == ["http://a.test", "http://b.test"]
    assert settings.allowed_extensions == ["pdf", "txt"]


def test_celery_urls_default_to_redis_url() -> None:
    settings = build_settings(redis_url="redis://cache:6379/0")
    assert settings.celery_broker_url == "redis://cache:6379/0"
    assert settings.celery_result_backend == "redis://cache:6379/0"


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        build_settings(chunk_size=500, chunk_overlap=500)


def test_rerank_top_k_cannot_exceed_candidates() -> None:
    with pytest.raises(ValidationError, match="RERANK_TOP_K"):
        build_settings(rerank_top_k=70, rrf_max_candidates=60)


def test_database_url_requires_psycopg_driver() -> None:
    with pytest.raises(ValidationError, match="psycopg"):
        build_settings(database_url="postgresql://u:p@localhost/db")


def test_invalid_rate_limit_is_rejected() -> None:
    with pytest.raises(ValidationError, match="rate limit"):
        build_settings(rate_limit_login="ten per minute")


@pytest.mark.parametrize("secret", ["change-me", "short-secret"])
def test_production_requires_strong_jwt_secret(secret: str) -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        build_settings(
            app_env="production",
            jwt_secret=secret,
            cookie_secure=True,
            minio_secret_key="a-real-minio-secret",
            llm_api_key="sk-test",
        )


def test_valid_production_settings() -> None:
    settings = build_settings(
        app_env="production",
        jwt_secret="p" * 48,
        cookie_secure=True,
        minio_secret_key="a-real-minio-secret",
        llm_api_key="sk-test",
    )
    assert settings.is_production


def test_secrets_are_not_rendered() -> None:
    settings = build_settings(llm_api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(settings)
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "sk-very-secret"
