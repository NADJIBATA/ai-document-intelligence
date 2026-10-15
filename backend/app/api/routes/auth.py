"""Authentication endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request, Response, status

from app.api.deps import AuthServiceDep, CurrentUser, SettingsDep
from app.core.config import Settings
from app.core.exceptions import AuthenticationError
from app.core.rate_limit import client_ip, rate_limit
from app.schemas.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse
from app.schemas.common import ERROR_RESPONSES, ErrorResponse
from app.schemas.user import UserRead
from app.services.auth_service import ClientInfo, IssuedTokens

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE_PATH = "/api/auth"


def _client_info(request: Request) -> ClientInfo:
    return ClientInfo(user_agent=request.headers.get("user-agent"), ip_address=client_ip(request))


def _set_refresh_cookie(response: Response, tokens: IssuedTokens, settings: Settings) -> None:
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=tokens.refresh_token,
        max_age=tokens.refresh_expires_in,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path=REFRESH_COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path=REFRESH_COOKIE_PATH,
    )


def _token_response(tokens: IssuedTokens) -> TokenResponse:
    return TokenResponse(
        access_token=tokens.access_token,
        expires_in=tokens.access_expires_in,
        refresh_token=tokens.refresh_token,
        refresh_expires_in=tokens.refresh_expires_in,
    )


def _extract_refresh_token(
    request: Request, payload: RefreshRequest | None, settings: Settings
) -> str:
    token = payload.refresh_token if payload and payload.refresh_token else None
    token = token or request.cookies.get(settings.refresh_cookie_name)
    if not token:
        raise AuthenticationError("A refresh token is required.", code="missing_refresh_token")
    return token


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=UserRead,
    summary="Create an account",
    description="Creates a user account. Passwords are hashed with Argon2id; "
    "the email address is case-insensitive.",
    responses={
        409: {"model": ErrorResponse, "description": "Email already registered"},
        **ERROR_RESPONSES,
    },
    dependencies=[Depends(rate_limit("auth_register", "rate_limit_register"))],
)
async def register(payload: RegisterRequest, auth_service: AuthServiceDep) -> UserRead:
    user = await auth_service.register(
        email=payload.email,
        password=payload.password.get_secret_value(),
        full_name=payload.full_name,
    )
    return UserRead.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in",
    description="Returns a short-lived JWT access token and a rotating refresh token. "
    "The refresh token is also set as an HttpOnly cookie scoped to /api/auth.",
    responses=ERROR_RESPONSES,
    dependencies=[Depends(rate_limit("auth_login", "rate_limit_login"))],
)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    auth_service: AuthServiceDep,
    settings: SettingsDep,
) -> TokenResponse:
    _, tokens = await auth_service.login(
        email=payload.email,
        password=payload.password.get_secret_value(),
        client=_client_info(request),
    )
    _set_refresh_cookie(response, tokens, settings)
    return _token_response(tokens)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate the refresh token",
    description="Exchanges a valid refresh token (cookie or JSON body) for a new token pair. "
    "The presented token is revoked; replaying it revokes the whole session.",
    responses=ERROR_RESPONSES,
    dependencies=[Depends(rate_limit("auth_refresh", "rate_limit_refresh"))],
)
async def refresh(
    request: Request,
    response: Response,
    auth_service: AuthServiceDep,
    settings: SettingsDep,
    payload: Annotated[RefreshRequest | None, Body()] = None,
) -> TokenResponse:
    raw_token = _extract_refresh_token(request, payload, settings)
    tokens = await auth_service.refresh(refresh_token=raw_token, client=_client_info(request))
    _set_refresh_cookie(response, tokens, settings)
    return _token_response(tokens)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log out",
    description="Revokes the refresh token session (cookie or JSON body) and clears the cookie.",
    dependencies=[Depends(rate_limit("auth_refresh", "rate_limit_refresh"))],
)
async def logout(
    request: Request,
    auth_service: AuthServiceDep,
    settings: SettingsDep,
    payload: Annotated[RefreshRequest | None, Body()] = None,
) -> Response:
    token = payload.refresh_token if payload and payload.refresh_token else None
    token = token or request.cookies.get(settings.refresh_cookie_name)
    if token:
        await auth_service.logout(refresh_token=token)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    _clear_refresh_cookie(response, settings)
    return response


@router.get(
    "/me",
    response_model=UserRead,
    summary="Current user",
    description="Returns the profile of the authenticated user.",
    responses=ERROR_RESPONSES,
)
async def me(current_user: CurrentUser) -> UserRead:
    return UserRead.model_validate(current_user)
