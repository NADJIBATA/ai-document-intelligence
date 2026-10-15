from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import Settings
from app.core.exceptions import AuthenticationError
from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from tests.conftest import build_settings


class TestPasswords:
    def test_hash_is_argon2id_and_salted(self) -> None:
        first = hash_password("S3cure-password")
        second = hash_password("S3cure-password")
        assert first.startswith("$argon2id$")
        assert first != second
        assert "S3cure-password" not in first

    def test_verify(self) -> None:
        hashed = hash_password("S3cure-password")
        assert verify_password("S3cure-password", hashed)[0] is True
        assert verify_password("wrong-password1", hashed)[0] is False

    def test_unknown_hash_format_is_rejected_without_error(self) -> None:
        assert verify_password("anything", "plain-text-not-a-hash") == (False, None)


class TestAccessTokens:
    def test_round_trip(self, settings: Settings) -> None:
        user_id = uuid.uuid4()
        token, expires_in = create_access_token(user_id, settings)
        payload = decode_access_token(token, settings)
        assert payload.user_id == user_id
        assert expires_in == settings.access_token_expire_minutes * 60

    def test_expired_token(self, settings: Settings) -> None:
        now = datetime.now(UTC)
        token = jwt.encode(
            {
                "sub": str(uuid.uuid4()),
                "type": "access",
                "iss": settings.jwt_issuer,
                "iat": int((now - timedelta(hours=1)).timestamp()),
                "exp": int((now - timedelta(minutes=1)).timestamp()),
                "jti": "x",
            },
            settings.jwt_secret.get_secret_value(),
            algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(AuthenticationError) as exc_info:
            decode_access_token(token, settings)
        assert exc_info.value.code == "token_expired"

    def test_token_signed_with_another_secret_is_rejected(self, settings: Settings) -> None:
        other = build_settings(jwt_secret="another-secret-" + "y" * 40)
        token, _ = create_access_token(uuid.uuid4(), other)
        with pytest.raises(AuthenticationError):
            decode_access_token(token, settings)

    def test_tampered_token_is_rejected(self, settings: Settings) -> None:
        token, _ = create_access_token(uuid.uuid4(), settings)
        header, payload, signature = token.split(".")
        tampered = f"{header}.{payload}.{signature[::-1]}"
        with pytest.raises(AuthenticationError):
            decode_access_token(tampered, settings)

    def test_wrong_token_type_is_rejected(self, settings: Settings) -> None:
        now = datetime.now(UTC)
        token = jwt.encode(
            {
                "sub": str(uuid.uuid4()),
                "type": "refresh",
                "iss": settings.jwt_issuer,
                "iat": int(now.timestamp()),
                "exp": int((now + timedelta(minutes=5)).timestamp()),
                "jti": "x",
            },
            settings.jwt_secret.get_secret_value(),
            algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(AuthenticationError):
            decode_access_token(token, settings)

    def test_none_algorithm_is_rejected(self, settings: Settings) -> None:
        token = jwt.encode({"sub": str(uuid.uuid4()), "type": "access"}, key="", algorithm="none")
        with pytest.raises(AuthenticationError):
            decode_access_token(token, settings)


class TestRefreshTokens:
    def test_tokens_are_random_and_long(self) -> None:
        tokens = {generate_refresh_token() for _ in range(100)}
        assert len(tokens) == 100
        assert all(len(token) >= 64 for token in tokens)

    def test_hash_is_deterministic_sha256(self) -> None:
        token = generate_refresh_token()
        assert hash_refresh_token(token) == hash_refresh_token(token)
        assert len(hash_refresh_token(token)) == 64
        assert token not in hash_refresh_token(token)
