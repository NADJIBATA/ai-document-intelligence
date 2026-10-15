"""Password hashing (Argon2id), JWT access tokens and opaque refresh tokens."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

import jwt
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError
from pwdlib.hashers.argon2 import Argon2Hasher

from app.core.config import Settings
from app.core.exceptions import AuthenticationError

ACCESS_TOKEN_TYPE = "access"  # noqa: S105 - token *type* claim, not a secret

_password_hash = PasswordHash((Argon2Hasher(),))


# --------------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> tuple[bool, str | None]:
    """Verify a password.

    Returns ``(is_valid, new_hash)``; ``new_hash`` is set when the stored hash uses
    outdated parameters and should be replaced (transparent re-hashing).
    """
    try:
        return _password_hash.verify_and_update(password, hashed_password)
    except UnknownHashError:
        return False, None


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return _password_hash.hash(secrets.token_urlsafe(16))


def burn_password_check(password: str) -> None:
    """Spend the same time as a real verification (anti user-enumeration by timing)."""
    _password_hash.verify(password, _dummy_hash())


# ---------------------------------------------------------------- access tokens
@dataclass(frozen=True, slots=True)
class AccessTokenPayload:
    user_id: uuid.UUID
    token_id: str
    issued_at: datetime
    expires_at: datetime


def create_access_token(user_id: uuid.UUID, settings: Settings) -> tuple[str, int]:
    """Return ``(jwt, expires_in_seconds)``."""
    now = datetime.now(UTC)
    expires_at = now + settings.access_token_ttl
    claims = {
        "sub": str(user_id),
        "type": ACCESS_TOKEN_TYPE,
        "iss": settings.jwt_issuer,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": uuid.uuid4().hex,
    }
    token = jwt.encode(
        claims, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )
    return token, int(settings.access_token_ttl.total_seconds())


def decode_access_token(token: str, settings: Settings) -> AccessTokenPayload:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            options={"require": ["sub", "type", "iss", "iat", "exp", "jti"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("The access token has expired.", code="token_expired") from exc
    except jwt.PyJWTError as exc:
        raise AuthenticationError("The access token is invalid.", code="invalid_token") from exc

    if claims.get("type") != ACCESS_TOKEN_TYPE:
        raise AuthenticationError("The access token is invalid.", code="invalid_token")
    try:
        user_id = uuid.UUID(str(claims["sub"]))
    except ValueError as exc:
        raise AuthenticationError("The access token is invalid.", code="invalid_token") from exc

    return AccessTokenPayload(
        user_id=user_id,
        token_id=str(claims["jti"]),
        issued_at=datetime.fromtimestamp(int(claims["iat"]), tz=UTC),
        expires_at=datetime.fromtimestamp(int(claims["exp"]), tz=UTC),
    )


# --------------------------------------------------------------- refresh tokens
def generate_refresh_token() -> str:
    """Opaque, URL-safe, 384-bit random token. Only its hash is persisted."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
