"""Liveness and readiness probes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])

ReadinessCheck = Callable[[], Awaitable[None]]
CHECK_TIMEOUT_SECONDS = 3.0


@router.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 as long as the process is able to serve requests. "
    "Does not check external dependencies.",
)
async def health(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    return {"status": "ok", "service": settings.service_name, "version": settings.app_version}


async def _run_check(check: ReadinessCheck) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        await asyncio.wait_for(check(), timeout=CHECK_TIMEOUT_SECONDS)
    except Exception as exc:  # the probe must never raise
        return {
            "status": "error",
            "error": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    return {"status": "ok", "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Checks every external dependency (PostgreSQL, Redis, object storage, Qdrant). "
    "Returns 503 if at least one of them is unavailable. The model servers (embeddings, "
    "reranker) are not checked: the search degrades gracefully without them.",
    responses={503: {"description": "At least one dependency is unavailable"}},
)
async def ready(request: Request) -> JSONResponse:
    checks: dict[str, ReadinessCheck] = request.app.state.readiness_checks
    names = list(checks)
    results = await asyncio.gather(*(_run_check(checks[name]) for name in names))
    report = dict(zip(names, results, strict=True))
    is_ready = all(result["status"] == "ok" for result in results)
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={"status": "ready" if is_ready else "not_ready", "checks": report},
    )
