from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-request-id"]


def test_ready_when_dependencies_are_up(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["database"]["status"] == "ok"
    assert body["checks"]["redis"]["status"] == "ok"


def test_ready_reports_unavailable_dependency(make_client: Any) -> None:
    client: TestClient = make_client(redis_url="redis://127.0.0.1:1/0")
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["redis"]["status"] == "error"
    assert body["checks"]["database"]["status"] == "ok"


def test_client_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "my-trace-id-1234"})
    assert response.headers["x-request-id"] == "my-trace-id-1234"


def test_unsafe_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert response.headers["x-request-id"] != "bad id with spaces"


def test_openapi_documents_the_auth_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/auth/register", "/api/auth/login", "/api/auth/refresh", "/api/auth/me"):
        assert path in paths
