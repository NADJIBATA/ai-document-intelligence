"""Celery application.

Start a worker with::

    celery -A app.workers.celery_app:celery_app worker --queues ingestion,analysis --loglevel INFO

Two queues: ``ingestion`` (document processing) and ``analysis`` (LLM jobs: document
insights and comparisons). They can be served by the same worker or by separate ones.
"""

from __future__ import annotations

from typing import Any

import structlog
from celery import Celery, Task
from celery.signals import setup_logging, task_postrun, task_prerun

from app.core.config import get_settings
from app.core.logging import configure_logging
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


@task_prerun.connect
def _bind_task_context(task_id: str, task: Task, args: tuple[Any, ...], **_: Any) -> None:
    headers = task.request.headers or {}
    request_id = getattr(task.request, "request_id", None) or headers.get("request_id")
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        task_id=task_id, task_name=task.name, request_id=request_id or task_id
    )


@task_postrun.connect
def _clear_task_context(**_: Any) -> None:
    structlog.contextvars.clear_contextvars()
