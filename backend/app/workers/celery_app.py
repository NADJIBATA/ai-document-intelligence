"""Celery application.

Start a worker with::

    celery -A app.workers.celery_app:celery_app worker --queues ingestion,analysis --loglevel INFO

Two queues: ``ingestion`` (document processing) and ``analysis`` (LLM jobs: document
insights and comparisons). They can be served by the same worker or by separate ones.
"""

from __future__ import annotations

import time
from typing import Any

import redis
import structlog
from celery import Celery, Task
from celery.signals import setup_logging, task_postrun, task_prerun

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.metrics import JobMetricsStore
from app.services.task_dispatcher import ANALYSIS_QUEUE, INGESTION_QUEUE

settings = get_settings()

celery_app = Celery(
    "docintel",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.tasks.documents",
        "app.workers.tasks.intelligence",
        "app.workers.tasks.comparisons",
    ],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue=INGESTION_QUEUE,
    task_routes={
        "documents.*": {"queue": INGESTION_QUEUE},
        "intelligence.*": {"queue": ANALYSIS_QUEUE},
        "comparisons.*": {"queue": ANALYSIS_QUEUE},
    },
    # Reliability: a job is acknowledged only once finished; if the worker dies the job
    # is redelivered (the ingestion is idempotent).
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=900,
    task_soft_time_limit=840,
    result_expires=24 * 3600,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 3600},
    worker_hijack_root_logger=False,
    enable_utc=True,
    timezone="UTC",
)


@setup_logging.connect
def _configure_worker_logging(**_: Any) -> None:
    configure_logging(settings.model_copy(update={"service_name": "celery-worker"}))


# Start time of the jobs running in this process (prerun and postrun run in the same
# child process), and the Redis client of the job metrics, created after the fork.
_started: dict[str, float] = {}
_metrics_store: JobMetricsStore | None = None


def _job_metrics() -> JobMetricsStore:
    global _metrics_store
    if _metrics_store is None:
        client = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=1, socket_timeout=1
        )
        _metrics_store = JobMetricsStore(client)
    return _metrics_store


@task_prerun.connect
def _bind_task_context(task_id: str, task: Task, args: tuple[Any, ...], **_: Any) -> None:
    _started[task_id] = time.perf_counter()
    headers = task.request.headers or {}
    request_id = getattr(task.request, "request_id", None) or headers.get("request_id")
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        task_id=task_id, task_name=task.name, request_id=request_id or task_id
    )


@task_postrun.connect
def _record_task(task_id: str, task: Task, state: str | None = None, **_: Any) -> None:
    started = _started.pop(task_id, None)
    if started is not None and settings.metrics_enabled:
        try:
            _job_metrics().record(task.name, state or "unknown", time.perf_counter() - started)
        except Exception as exc:  # metrics must never make a job fail
            structlog.get_logger(__name__).warning(
                "job_metrics_not_recorded", error=type(exc).__name__
            )
    structlog.contextvars.clear_contextvars()
