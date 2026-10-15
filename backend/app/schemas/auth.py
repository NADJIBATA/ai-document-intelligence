from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field, SecretStr, field_validator

PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128  # bounds Argon2 work per request (DoS protection)


def _normalize_email(value: str) -> str:
    return value.strip().lower()


class RegisterRequest(BaseModel):
    email: EmailStr = Field(examples=["alice@example.org"])
    password: SecretStr = Field(
        description=f"{PASSWORD_MIN_LENGTH}-{PASSWORD_MAX_LENGTH} characters, "
        "at least one letter and one digit.",
        examples=["S3cure-password"],
    )
    full_name: str | None = Field(default=None, max_length=255, examples=["Alice Martin"])

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        return _normalize_email(value)

    @field_validator("full_name")
    @classmethod
    def _full_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        return value or None

    @field_validator("password")
    @classmethod
    def _password_policy(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if not PASSWORD_MIN_LENGTH <= len(raw) <= PASSWORD_MAX_LENGTH:
            raise ValueError(
                f"Password must contain between {PASSWORD_MIN_LENGTH} and "
                f"{PASSWORD_MAX_LENGTH} characters."
            )
        if not any(c.isalpha() for c in raw) or not any(c.isdigit() for c in raw):
            raise ValueError("Password must contain at least one letter and one digit.")
        return value


class LoginRequest(BaseModel):
    email: EmailStr = Field(examples=["alice@example.org"])
    password: SecretStr = Field(examples=["S3cure-password"])

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        return _normalize_email(value)

    @field_validator("password")
    @classmethod
    def _password_length(cls, value: SecretStr) -> SecretStr:
        if not 1 <= len(value.get_secret_value()) <= PASSWORD_MAX_LENGTH:
            raise ValueError("Invalid password length.")
        return value


class RefreshRequest(BaseModel):
    refresh_token: str | None = Field(
        default=None,
        max_length=512,
        description="Optional: browsers send it automatically in the HttpOnly cookie.",
    )


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105
    expires_in: int = Field(description="Access token lifetime in seconds")
    refresh_token: str = Field(
        description="Also set as an HttpOnly cookie. Browser clients should ignore this field."
    )
    refresh_expires_in: int = Field(description="Refresh token lifetime in seconds")
