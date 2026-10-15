from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from tests.integration.conftest import (
    DEFAULT_PASSWORD,
    auth_headers,
    login_user,
    register_user,
)


class TestRegister:
    def test_register_returns_public_profile(self, client: TestClient) -> None:
        body = register_user(client, "alice@docintel.dev")
        assert body["email"] == "alice@docintel.dev"
        assert body["is_active"] is True
        assert "password" not in body
        assert "hashed_password" not in body

    def test_password_is_stored_hashed(self, client: TestClient, db_engine: Engine) -> None:
        register_user(client, "alice@docintel.dev")
        with db_engine.connect() as connection:
            stored = connection.execute(
                text("SELECT hashed_password FROM users WHERE email = 'alice@docintel.dev'")
            ).scalar_one()
        assert stored.startswith("$argon2id$")
        assert DEFAULT_PASSWORD not in stored

    def test_email_is_case_insensitive_and_unique(self, client: TestClient) -> None:
        register_user(client, "Alice@DocIntel.dev")
        response = client.post(
            "/api/auth/register",
            json={"email": "alice@docintel.dev", "password": DEFAULT_PASSWORD},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "email_already_registered"

    def test_weak_password_is_rejected_without_echoing_it(self, client: TestClient) -> None:
        response = client.post(
            "/api/auth/register", json={"email": "bob@docintel.dev", "password": "short"}
        )
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "validation_error"
        assert "short" not in response.text

    def test_invalid_email_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/auth/register", json={"email": "not-an-email", "password": DEFAULT_PASSWORD}
        )
        assert response.status_code == 422


class TestLogin:
    def test_login_returns_tokens_and_sets_http_only_cookie(self, client: TestClient) -> None:
        register_user(client, "alice@docintel.dev")
        response = client.post(
            "/api/auth/login",
            json={"email": "ALICE@docintel.dev", "password": DEFAULT_PASSWORD},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == 15 * 60
        assert body["access_token"]
        assert body["refresh_token"]
        cookie_header = response.headers["set-cookie"].lower()
        assert "refresh_token=" in cookie_header
        assert "httponly" in cookie_header
        assert "path=/api/auth" in cookie_header
        assert "samesite=strict" in cookie_header

    def test_wrong_password_and_unknown_email_look_identical(self, client: TestClient) -> None:
        register_user(client, "alice@docintel.dev")
        wrong_password = client.post(
            "/api/auth/login", json={"email": "alice@docintel.dev", "password": "Wrong-pass1"}
        )
        unknown_email = client.post(
            "/api/auth/login", json={"email": "nobody@docintel.dev", "password": "Wrong-pass1"}
        )
        assert wrong_password.status_code == unknown_email.status_code == 401
        assert wrong_password.json()["error"]["code"] == unknown_email.json()["error"]["code"]
        assert wrong_password.json()["error"]["message"] == unknown_email.json()["error"]["message"]

    def test_login_is_rate_limited(self, make_client: Any) -> None:
        client: TestClient = make_client(rate_limit_login="3/hour")
        for _ in range(3):
            response = client.post(
                "/api/auth/login", json={"email": "x@docintel.dev", "password": "Wrong-pass1"}
            )
            assert response.status_code == 401
        response = client.post(
            "/api/auth/login", json={"email": "x@docintel.dev", "password": "Wrong-pass1"}
        )
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1
        assert response.json()["error"]["code"] == "rate_limited"

    def test_login_rate_limit_is_shared_through_redis(self, make_client: Any) -> None:
        first_replica: TestClient = make_client(
            rate_limit_storage="redis", rate_limit_login="2/hour"
        )
        second_replica: TestClient = make_client(
            rate_limit_storage="redis", rate_limit_login="2/hour"
        )
        payload = {"email": "x@docintel.dev", "password": "Wrong-pass1"}
        assert first_replica.post("/api/auth/login", json=payload).status_code == 401
        assert second_replica.post("/api/auth/login", json=payload).status_code == 401
        # Third attempt on the first replica: the counter lives in Redis, not in memory.
        assert first_replica.post("/api/auth/login", json=payload).status_code == 429


class TestMe:
    def test_me_with_valid_token(self, client: TestClient) -> None:
        register_user(client, "alice@docintel.dev")
        tokens = login_user(client, "alice@docintel.dev")
        response = client.get("/api/auth/me", headers=auth_headers(tokens["access_token"]))
        assert response.status_code == 200
        body = response.json()
        assert body["email"] == "alice@docintel.dev"
        assert body["created_at"].endswith("Z")  # database sessions run in UTC

    def test_me_without_token(self, client: TestClient) -> None:
        response = client.get("/api/auth/me")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["error"]["request_id"] == response.headers["x-request-id"]

    def test_me_with_garbage_token(self, client: TestClient) -> None:
        response = client.get("/api/auth/me", headers=auth_headers("not.a.jwt"))
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_token"

    def test_refresh_token_cannot_be_used_as_access_token(self, client: TestClient) -> None:
        register_user(client, "alice@docintel.dev")
        tokens = login_user(client, "alice@docintel.dev")
        response = client.get("/api/auth/me", headers=auth_headers(tokens["refresh_token"]))
        assert response.status_code == 401


class TestRefresh:
    def test_refresh_with_body_rotates_the_token(self, client: TestClient) -> None:
        register_user(client, "alice@docintel.dev")
        tokens = login_user(client, "alice@docintel.dev")
        client.cookies.clear()

        response = client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        assert response.status_code == 200
        new_tokens = response.json()
        assert new_tokens["refresh_token"] != tokens["refresh_token"]
        me = client.get("/api/auth/me", headers=auth_headers(new_tokens["access_token"]))
        assert me.status_code == 200

    def test_refresh_with_cookie(self, client: TestClient) -> None:
        register_user(client, "alice@docintel.dev")
        login_user(client, "alice@docintel.dev")  # stores the HttpOnly cookie in the client jar
        response = client.post("/api/auth/refresh")
        assert response.status_code == 200
        assert response.json()["access_token"]

    def test_reusing_a_rotated_token_revokes_the_whole_family(self, client: TestClient) -> None:
        register_user(client, "alice@docintel.dev")
        first = login_user(client, "alice@docintel.dev")["refresh_token"]
        client.cookies.clear()

        second = client.post("/api/auth/refresh", json={"refresh_token": first}).json()[
            "refresh_token"
        ]
        client.cookies.clear()

        replay = client.post("/api/auth/refresh", json={"refresh_token": first})
        assert replay.status_code == 401
        client.cookies.clear()

        # The legitimate latest token has been revoked too (possible theft detected).
        after_replay = client.post("/api/auth/refresh", json={"refresh_token": second})
        assert after_replay.status_code == 401

    def test_refresh_without_token(self, client: TestClient) -> None:
        response = client.post("/api/auth/refresh")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "missing_refresh_token"

    def test_unknown_refresh_token(self, client: TestClient) -> None:
        response = client.post("/api/auth/refresh", json={"refresh_token": "unknown-token"})
        assert response.status_code == 401


class TestLogout:
    def test_logout_revokes_the_session_and_clears_the_cookie(self, client: TestClient) -> None:
        register_user(client, "alice@docintel.dev")
        tokens = login_user(client, "alice@docintel.dev")
        client.cookies.clear()

        response = client.post("/api/auth/logout", json={"refresh_token": tokens["refresh_token"]})
        assert response.status_code == 204
        assert "refresh_token=" in response.headers["set-cookie"]

        refresh = client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        assert refresh.status_code == 401

    def test_logout_is_idempotent(self, client: TestClient) -> None:
        assert client.post("/api/auth/logout").status_code == 204
        response = client.post("/api/auth/logout", json={"refresh_token": "unknown"})
        assert response.status_code == 204
