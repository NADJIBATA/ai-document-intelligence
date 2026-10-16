"""Values read when Prometheus scrapes ``GET /metrics``.

Each collector is independent: if PostgreSQL or Redis is unavailable, its metrics are
missing from that scrape (and ``docintel_metrics_collector_errors_total`` increases)
but the other metrics are still served.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

import structlog
from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.metrics import (
    JOB_BUCKET_KEY,
    JOB_COUNT_KEY,
    JOB_SUM_KEY,
    REGISTRY,
    MetricFamily,
    Sample,
    job_metric_families,
)
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.user import User

logger = structlog.get_logger(__name__)

COLLECTOR_ERRORS = REGISTRY.counter(
    "docintel_metrics_collector_errors_total",
    "Scrapes during which a collector failed (its metrics were left out).",
    ("collector",),
)

Collector = Callable[[], Awaitable[list[MetricFamily]]]
COLLECTOR_TIMEOUT_SECONDS = 3.0


def database_collector(session_factory: async_sessionmaker[AsyncSession]) -> Collector:
    async def collect() -> list[MetricFamily]:
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(Document.status, func.count()).group_by(Document.status)
                )
            ).all()
            users = await session.scalar(select(func.count()).select_from(User))
        by_status = {status: 0 for status in DocumentStatus}
        for status, count in rows:
            by_status[DocumentStatus(status)] = count
        documents = MetricFamily(
            "docintel_documents", "gauge", "Documents stored, by processing status."
        )
        documents.samples = [
            Sample("", {"status": status.value}, count) for status, count in by_status.items()
        ]
        accounts = MetricFamily("docintel_users", "gauge", "Registered user accounts.")
        accounts.samples = [Sample("", {}, users or 0)]
        return [documents, accounts]

    return collect


def queue_collector(broker: Redis, queues: Sequence[str]) -> Collector:
    """Jobs waiting in the Celery queues (Redis broker: one list per queue)."""

    async def collect() -> list[MetricFamily]:
        family = MetricFamily(
            "docintel_queue_length", "gauge", "Background jobs waiting in each Celery queue."
        )
        for queue in queues:
            length = await broker.llen(queue)  # type: ignore[misc]
            family.samples.append(Sample("", {"queue": queue}, float(length)))
        return [family]

    return collect


def job_collector(redis: Redis) -> Collector:
    async def collect() -> list[MetricFamily]:
        counts = await redis.hgetall(JOB_COUNT_KEY)  # type: ignore[misc]
        sums = await redis.hgetall(JOB_SUM_KEY)  # type: ignore[misc]
        buckets = await redis.hgetall(JOB_BUCKET_KEY)  # type: ignore[misc]
        return job_metric_families(counts, sums, buckets)

    return collect


async def render_metrics(collectors: dict[str, Collector]) -> str:
    families: list[MetricFamily] = []
    for name, collect in collectors.items():
        try:
            families.extend(await asyncio.wait_for(collect(), COLLECTOR_TIMEOUT_SECONDS))
        except Exception as exc:  # a scrape must never fail because of one source
            COLLECTOR_ERRORS.inc(collector=name)
            logger.warning("metrics_collector_failed", collector=name, error=type(exc).__name__)
    return REGISTRY.render(families)
