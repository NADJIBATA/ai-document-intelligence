from __future__ import annotations

import httpx
import pytest

from app.core import http as http_module
from app.core.http import UpstreamError, post_json


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(http_module.time, "sleep", delays.append)
    return delays


def client_for(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_returns_the_json_object() -> None:
    client = client_for(httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": 1})))
    assert post_json(client, "http://model/v1/x", {"a": 1}) == {"ok": 1}


def test_retries_on_5xx_then_succeeds(_no_sleep: list[float]) -> None:
    statuses = iter([503, 500, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), json={"ok": True})

    client = client_for(httpx.MockTransport(handler))
    assert post_json(client, "http://model/v1/x", {}, max_retries=2) == {"ok": True}
    assert _no_sleep == [0.5, 1.0]  # exponential backoff


def test_retries_on_network_errors_then_gives_up() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused")

    client = client_for(httpx.MockTransport(handler))
    with pytest.raises(UpstreamError, match="embeddings: ConnectError") as error:
        post_json(client, "http://model/v1/x", {}, max_retries=2, service="embeddings")
    assert calls == 3
    assert error.value.status_code is None


def test_client_errors_are_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="model not found")

    client = client_for(httpx.MockTransport(handler))
    with pytest.raises(UpstreamError, match="HTTP 400") as error:
        post_json(client, "http://model/v1/x", {}, max_retries=3)
    assert calls == 1
    assert error.value.status_code == 400


def test_rate_limit_is_retried() -> None:
    statuses = iter([429, 200])
    client = client_for(
        httpx.MockTransport(lambda request: httpx.Response(next(statuses), json={}))
    )
    assert post_json(client, "http://model/v1/x", {}, max_retries=1) == {}


def test_non_object_response_is_rejected() -> None:
    client = client_for(httpx.MockTransport(lambda request: httpx.Response(200, json=[1, 2])))
    with pytest.raises(UpstreamError, match="unexpected response format"):
        post_json(client, "http://model/v1/x", {})


def test_authentication_errors_do_not_expose_the_response_body() -> None:
    client = client_for(
        httpx.MockTransport(
            lambda request: httpx.Response(401, text="Incorrect API key provided: k-secret")
        )
    )
    with pytest.raises(UpstreamError) as error:
        post_json(client, "http://model/v1/x", {}, service="embeddings")
    assert str(error.value) == "embeddings: HTTP 401"
    assert "k-secret" not in str(error.value)


def test_read_timeouts_are_retried_unless_disabled() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out")

    client = client_for(httpx.MockTransport(handler))
    with pytest.raises(UpstreamError, match="ReadTimeout"):
        post_json(client, "http://model/v1/x", {}, max_retries=2)
    assert calls == 3

    calls = 0
    with pytest.raises(UpstreamError, match="ReadTimeout"):
        post_json(client, "http://model/v1/x", {}, max_retries=2, retry_read_timeout=False)
    assert calls == 1
