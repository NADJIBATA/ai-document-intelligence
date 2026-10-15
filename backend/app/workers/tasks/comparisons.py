"""Document comparison task."""

from __future__ import annotations

import uuid

import structlog
from celery import Task

from app.services.comparison_service import TransientComparisonError
from app.services.task_dispatcher import COMPARE_DOCUMENTS_TASK
from app.workers.celery_app import celery_app
from app.workers.resources import get_comparison_service

MAX_RETRIES = 3


def _retry_delay(retries: int) -> int:
    """Exponential backoff: 15 s, 30 s, 60 s... capped at 5 minutes."""
    return int(min(15 * 2**retries, 300))


@celery_app.task(name=COMPARE_DOCUMENTS_TASK, bind=True, max_retries=MAX_RETRIES)
def compare_documents(self: Task, comparison_id: str) -> dict[str, str]:
    structlog.contextvars.bind_contextvars(comparison_id=comparison_id)
    final_attempt = self.request.retries >= MAX_RETRIES
    try:
        outcome = get_comparison_service().run(
            uuid.UUID(comparison_id), final_attempt=final_attempt
        )
    except TransientComparisonError as exc:
        raise self.retry(exc=exc, countdown=_retry_delay(self.request.retries)) from exc
    return {"comparison_id": comparison_id, "outcome": outcome.value}
