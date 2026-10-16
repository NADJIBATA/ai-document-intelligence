"""Prometheus endpoint.

Not part of the public API: nginx answers 404 for /metrics, Prometheus scrapes the
backend directly on the internal Docker network. Disabled with METRICS_ENABLED=false.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, Response

from app.core.metrics import CONTENT_TYPE
from app.services.metrics_service import render_metrics

router = APIRouter(tags=["health"])


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    body = await render_metrics(request.app.state.metrics_collectors)
    return PlainTextResponse(body, media_type=CONTENT_TYPE)
