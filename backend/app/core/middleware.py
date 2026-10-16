"""ASGI middlewares.

``RequestContextMiddleware`` is a *pure* ASGI middleware (not ``BaseHTTPMiddleware``):
it assigns a request id, exposes it in the ``X-Request-ID`` response header, binds it
to the logging context, writes one structured access log per request and converts
unhandled exceptions into a JSON 500 that still carries the request id.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.context import bind_request_context, clear_request_context, sanitize_request_id
from app.core.exceptions import error_body
from app.core.metrics import HTTP_DURATION, HTTP_IN_PROGRESS, HTTP_REQUESTS

logger = structlog.get_logger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"
_QUIET_PATHS = frozenset({"/health", "/ready", "/metrics"})


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = sanitize_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id
        bind_request_context(request_id)

        status_code = 500
        response_started = False
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_started = True
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("unhandled_exception", path=scope.get("path"))
            if response_started:
                raise
            status_code = 500
            response = JSONResponse(
                status_code=500,
                content=error_body("internal_error", "An unexpected error occurred.", request_id),
                headers={REQUEST_ID_HEADER: request_id},
            )
            await response(scope, receive, send)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            path = scope.get("path", "")
            client = scope.get("client")
            log = logger.debug if path in _QUIET_PATHS and status_code < 400 else logger.info
            log(
                "http_request",
                method=scope.get("method"),
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
                user_id=state.get("user_id"),
                client_ip=client[0] if client else None,
            )
            clear_request_context()


class MetricsMiddleware:
    """Counts the HTTP requests and measures their duration (pure ASGI).

    Requests are labelled with the *route template* (``/api/documents/{document_id}``),
    never with the raw path: identifiers would create an unbounded number of series.
    Paths that match no route are grouped under ``unmatched``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == "/metrics":
            await self.app(scope, receive, send)
            return

        status_code = 500
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        HTTP_IN_PROGRESS.inc()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            HTTP_IN_PROGRESS.dec()
            route = route_template(scope)
            method = scope.get("method", "")
            HTTP_REQUESTS.inc(method=method, route=route, status=str(status_code))
            HTTP_DURATION.observe(time.perf_counter() - start, method=method, route=route)


def route_template(scope: Scope) -> str:
    """Template of the matched route, e.g. ``/api/documents/{document_id}``.

    Depending on the FastAPI version, the route stored in the scope holds either the full
    template or only the part below the prefixes of the routers that include it
    (``/documents/{document_id}``). The missing prefix is then taken from the request
    path: it is the part before the suffix that the route matches. The prefixes of this
    application are literal (``/api``), so no identifier can end up in the label.
    """
    route = scope.get("route")
    template = getattr(route, "path", None)
    if not isinstance(template, str) or not template:
        return "unmatched"
    path: str = scope.get("path", "")
    regex = getattr(route, "path_regex", None)
    if regex is None:
        return template
    for index, char in enumerate(path):
        if char == "/" and regex.fullmatch(path[index:]):
            return path[:index] + template
    return template
