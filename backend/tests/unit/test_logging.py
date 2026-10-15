from __future__ import annotations

from typing import Any

from app.core.logging import REDACTED, redact_sensitive_data

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJlLXNpZ25hdHVyZQ"


def _redact(event: dict[str, Any]) -> dict[str, Any]:
    return dict(redact_sensitive_data(None, "info", event))


def test_sensitive_keys_are_redacted() -> None:
    event = _redact(
        {
            "event": "login",
            "password": "S3cure-password",
            "access_token": "abc",
            "Authorization": "Bearer abc",
            "llm_api_key": "sk-123",
            "jwt_secret": "secret",
            "user_id": "42",
        }
    )
    assert event["password"] == REDACTED
    assert event["access_token"] == REDACTED
    assert event["Authorization"] == REDACTED
    assert event["llm_api_key"] == REDACTED
    assert event["jwt_secret"] == REDACTED
    assert event["user_id"] == "42"
    assert event["event"] == "login"


def test_nested_structures_are_redacted() -> None:
    event = _redact({"event": "x", "payload": {"user": {"password": "p"}, "items": [{"token": 1}]}})
    assert event["payload"]["user"]["password"] == REDACTED
    assert event["payload"]["items"][0]["token"] == REDACTED


def test_jwt_and_bearer_values_are_masked_everywhere() -> None:
    event = _redact({"event": f"received {JWT}", "header": f"Bearer {JWT}"})
    assert JWT not in event["event"]
    assert JWT not in event["header"]


def test_non_sensitive_keys_that_look_similar_are_kept() -> None:
    event = _redact({"event": "x", "passages_count": 5, "document_id": "d"})
    assert event["passages_count"] == 5
    assert event["document_id"] == "d"
