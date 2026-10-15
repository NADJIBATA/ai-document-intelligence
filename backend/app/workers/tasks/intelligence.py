"""Document Intelligence task: structured extraction with the LLM."""

from __future__ import annotations

import uuid

import structlog
from celery import Task

from app.services.intelligence_service import TransientInsightError
from app.services.task_dispatcher import EXTRACT_INSIGHTS_TASK
from app.workers.celery_app import celery_app
from app.workers.resources import get_intelligence_service

MAX_RETRIES = 3


def _retry_delay(retries: int) -> int:
    """Exponential backoff: 15 s, 30 s, 60 s... capped at 5 minutes."""
    return int(min(15 * 2**retries, 300))


@celery_app.task(name=EXTRACT_INSIGHTS_TASK, bind=True, max_retries=MAX_RETRIES)
def extract_insights(self: Task, document_id: str) -> dict[str, str]:
    structlog.contextvars.bind_contextvars(document_id=document_id)
    final_attempt = self.request.retries >= MAX_RETRIES
    try:
        outcome = get_intelligence_service().extract(
            uuid.UUID(document_id), final_attempt=final_attempt
        )
    except TransientInsightError as exc:
        raise self.retry(exc=exc, countdown=_retry_delay(self.request.retries)) from exc
    return {"document_id": document_id, "outcome": outcome.value}
