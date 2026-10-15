"""Authentication use cases: registration, login, refresh-token rotation, logout."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.security import (
    burn_password_check,
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.models.user import User
from app.repositories.refresh_token import RefreshTokenRepository
from app.repositories.user import UserRepository

logger = structlog.get_logger(__name__)

INVALID_CREDENTIALS_MESSAGE = "Invalid email or password."


@dataclass(frozen=True, slots=True)
class ClientInfo:
    user_agent: str | None
    ip_address: str | None


@dataclass(frozen=True, slots=True)
class IssuedTokens:
    access_token: str
    access_expires_in: int
    refresh_token: str
    refresh_expires_in: int


class AuthService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._users = UserRepository(session)
        self._tokens = RefreshTokenRepository(session)

    # ----------------------------------------------------------------- register
    async def register(self, *, email: str, password: str, full_name: str | None) -> User:
        if await self._users.get_by_email(email) is not None:
            raise ConflictError(
                "An account with this email already exists.", code="email_already_registered"
            )
        user = await self._users.add(
            email=email, hashed_password=hash_password(password), full_name=full_name
        )
        try:
            await self._session.commit()
        except IntegrityError as exc:  # concurrent registration with the same email
            await self._session.rollback()
            raise ConflictError(
                "An account with this email already exists.", code="email_already_registered"
            ) from exc
        logger.info("user_registered", user_id=str(user.id))
        return user

    # -------------------------------------------------------------------- login
    async def login(
        self, *, email: str, password: str, client: ClientInfo
    ) -> tuple[User, IssuedTokens]:
        user = await self._users.get_by_email(email)
        if user is None:
            burn_password_check(password)
            logger.info("login_failed", reason="unknown_email")
            raise AuthenticationError(INVALID_CREDENTIALS_MESSAGE, code="invalid_credentials")

        is_valid, new_hash = verify_password(password, user.hashed_password)
        if not is_valid:
            logger.info("login_failed", reason="wrong_password", user_id=str(user.id))
            raise AuthenticationError(INVALID_CREDENTIALS_MESSAGE, code="invalid_credentials")
        if not user.is_active:
            raise AuthenticationError("This account is disabled.", code="account_disabled")

        if new_hash is not None:
            user.hashed_password = new_hash
        user.last_login_at = datetime.now(UTC)
        tokens, _ = await self._issue_tokens(user, family_id=uuid.uuid4(), client=client)
        await self._session.commit()
        logger.info("login_succeeded", user_id=str(user.id))
        return user, tokens

    # ------------------------------------------------------------------ refresh
    async def refresh(self, *, refresh_token: str, client: ClientInfo) -> IssuedTokens:
        now = datetime.now(UTC)
        record = await self._tokens.get_by_hash_for_update(hash_refresh_token(refresh_token))
        if record is None:
            raise AuthenticationError("Invalid refresh token.", code="invalid_refresh_token")

        if record.revoked_at is not None:
            # A rotated token is being replayed: assume theft, kill the whole session family.
            await self._tokens.revoke_family(record.family_id, now)
            await self._session.commit()
            logger.warning(
                "refresh_token_reuse_detected",
                user_id=str(record.user_id),
                family_id=str(record.family_id),
            )
            raise AuthenticationError("Invalid refresh token.", code="invalid_refresh_token")

        if record.expires_at <= now:
            raise AuthenticationError("The refresh token has expired.", code="refresh_expired")

        user = await self._users.get_by_id(record.user_id)
        if user is None or not user.is_active:
            await self._tokens.revoke_family(record.family_id, now)
            await self._session.commit()
            raise AuthenticationError("Invalid refresh token.", code="invalid_refresh_token")

        tokens, new_record_id = await self._issue_tokens(
            user, family_id=record.family_id, client=client
        )
        record.revoked_at = now
        record.replaced_by_id = new_record_id
        await self._session.commit()
        logger.info("refresh_token_rotated", user_id=str(user.id))
        return tokens

    # ------------------------------------------------------------------- logout
    async def logout(self, *, refresh_token: str) -> None:
        """Revoke the session family of this token. Idempotent."""
        record = await self._tokens.get_by_hash_for_update(hash_refresh_token(refresh_token))
        if record is None:
            return
        await self._tokens.revoke_family(record.family_id, datetime.now(UTC))
        await self._session.commit()
        logger.info("logout", user_id=str(record.user_id))

    # ------------------------------------------------------------ access tokens
    async def get_user_from_access_token(self, access_token: str) -> User:
        payload = decode_access_token(access_token, self._settings)
        user = await self._users.get_by_id(payload.user_id)
        if user is None or not user.is_active:
            raise AuthenticationError("The access token is invalid.", code="invalid_token")
        return user

    # ------------------------------------------------------------------ helpers
    async def _issue_tokens(
        self, user: User, *, family_id: uuid.UUID, client: ClientInfo
    ) -> tuple[IssuedTokens, uuid.UUID]:
        """Create an access token and persist a new refresh token (not committed)."""
        access_token, access_expires_in = create_access_token(user.id, self._settings)
        raw_refresh = generate_refresh_token()
        record = await self._tokens.add(
            user_id=user.id,
            token_hash=hash_refresh_token(raw_refresh),
            family_id=family_id,
            expires_at=datetime.now(UTC) + self._settings.refresh_token_ttl,
            user_agent=(client.user_agent or "")[:512] or None,
            ip_address=(client.ip_address or "")[:64] or None,
        )
        tokens = IssuedTokens(
            access_token=access_token,
            access_expires_in=access_expires_in,
            refresh_token=raw_refresh,
            refresh_expires_in=int(self._settings.refresh_token_ttl.total_seconds()),
        )
        return tokens, record.id
