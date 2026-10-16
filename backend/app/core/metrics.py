"""Prometheus metrics, without a client library.

A small registry (counters, gauges, histograms) rendered in the Prometheus text format
by ``GET /metrics``. Three sources:

* in-process metrics of the API (HTTP requests, retrieval, chat answers, LLM calls);
* values read at scrape time (documents by status, queue lengths) - see
  ``app.services.metrics_service``;
* metrics of the Celery jobs. The worker runs its jobs in several child processes, so
  it cannot expose an in-memory registry: each job adds its outcome and duration to a few
  Redis hashes (``JobMetricsStore``) and the API reads them when it is scraped.

Labels never contain user data (no user id, no file name, no query): only bounded
values such as a route template, a status code or a pipeline stage.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

MetricType = Literal["counter", "gauge", "histogram"]

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
# Seconds. Wide range: an HTTP request takes milliseconds, an LLM answer seconds and an
# ingestion job up to minutes.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
)  # fmt: skip

LabelValues = tuple[str, ...]


# ----------------------------------------------------------------------- format
def format_value(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_help(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def _format_labels(names: Sequence[str], values: Sequence[str]) -> str:
    if not names:
        return ""
    pairs = ",".join(
        f'{name}="{_escape_label(value)}"' for name, value in zip(names, values, strict=True)
    )
    return "{" + pairs + "}"


@dataclass(frozen=True, slots=True)
class Sample:
    suffix: str  # "", "_bucket", "_sum", "_count"
    labels: Mapping[str, str]
    value: float


@dataclass(slots=True)
class MetricFamily:
    """A metric with its samples, ready to be rendered."""

    name: str
    type: MetricType
    help: str
    samples: list[Sample] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} {self.type}"]
        for sample in self.samples:
            labels = _format_labels(list(sample.labels), list(sample.labels.values()))
            lines.append(f"{self.name}{sample.suffix}{labels} {format_value(sample.value)}")
        return "\n".join(lines)


def histogram_samples(
    labels: Mapping[str, str],
    buckets: Sequence[float],
    bucket_counts: Sequence[float],
    total: float,
    count: float,
) -> list[Sample]:
    """Samples of one histogram series; ``bucket_counts`` are per bucket (not cumulative),
    one per bound of ``buckets`` plus one for +Inf."""
    samples: list[Sample] = []
    cumulative = 0.0
    for bound, bucket_count in zip((*buckets, math.inf), bucket_counts, strict=True):
        cumulative += bucket_count
        samples.append(Sample("_bucket", {**labels, "le": format_value(bound)}, cumulative))
    samples.append(Sample("_sum", dict(labels), total))
    samples.append(Sample("_count", dict(labels), count))
    return samples


def bucket_index(buckets: Sequence[float], value: float) -> int:
    """Index of the first bucket whose upper bound contains ``value`` (len = +Inf)."""
    for index, bound in enumerate(buckets):
        if value <= bound:
            return index
    return len(buckets)


# ---------------------------------------------------------------------- metrics
class _Metric:
    type: MetricType

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str]) -> None:
        self.name = name
        self.help = help_text
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: Mapping[str, object]) -> LabelValues:
        if set(labels) != set(self.labelnames):
            raise ValueError(f"{self.name}: expected labels {self.labelnames}, got {tuple(labels)}")
        return tuple(str(labels[name]) for name in self.labelnames)

    def _labels(self, key: LabelValues) -> dict[str, str]:
        return dict(zip(self.labelnames, key, strict=True))

    def collect(self) -> MetricFamily:
        raise NotImplementedError


class Counter(_Metric):
    type: MetricType = "counter"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labelnames)
        self._values: dict[LabelValues, float] = {}

    def inc(self, amount: float = 1.0, **labels: object) -> None:
        if amount < 0:
            raise ValueError("a counter can only increase")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: object) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def collect(self) -> MetricFamily:
        with self._lock:
            items = sorted(self._values.items())
        samples = [Sample("", self._labels(key), value) for key, value in items]
        return MetricFamily(self.name, self.type, self.help, samples)


class Gauge(_Metric):
    type: MetricType = "gauge"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labelnames)
        self._values: dict[LabelValues, float] = {}

    def set(self, value: float, **labels: object) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, **labels: object) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: object) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: object) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def collect(self) -> MetricFamily:
        with self._lock:
            items = sorted(self._values.items())
        samples = [Sample("", self._labels(key), value) for key, value in items]
        return MetricFamily(self.name, self.type, self.help, samples)


@dataclass(slots=True)
class _HistogramState:
    buckets: list[float]
    total: float = 0.0
    count: float = 0.0


class Histogram(_Metric):
    type: MetricType = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, help_text, labelnames)
        if list(buckets) != sorted(buckets) or not buckets:
            raise ValueError("buckets must be sorted and not empty")
        self.buckets = tuple(float(bound) for bound in buckets)
        self._states: dict[LabelValues, _HistogramState] = {}

    def observe(self, value: float, **labels: object) -> None:
        key = self._key(labels)
        index = bucket_index(self.buckets, value)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _HistogramState(buckets=[0.0] * (len(self.buckets) + 1))
                self._states[key] = state
            state.buckets[index] += 1
            state.total += value
            state.count += 1

    def count(self, **labels: object) -> float:
        with self._lock:
            state = self._states.get(self._key(labels))
            return state.count if state else 0.0

    def collect(self) -> MetricFamily:
        with self._lock:
            items = [
                (key, list(state.buckets), state.total, state.count)
                for key, state in sorted(self._states.items())
            ]
        samples: list[Sample] = []
        for key, buckets, total, count in items:
            samples.extend(
                histogram_samples(self._labels(key), self.buckets, buckets, total, count)
            )
        return MetricFamily(self.name, self.type, self.help, samples)


class MetricsRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _register(self, metric: _Metric) -> None:
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"metric {metric.name} is already registered")
            self._metrics[metric.name] = metric

    def counter(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> Counter:
        metric = Counter(name, help_text, labelnames)
        self._register(metric)
        return metric

    def gauge(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> Gauge:
        metric = Gauge(name, help_text, labelnames)
        self._register(metric)
        return metric

    def histogram(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> Histogram:
        metric = Histogram(name, help_text, labelnames, buckets)
        self._register(metric)
        return metric

    def collect(self) -> list[MetricFamily]:
        with self._lock:
            metrics = list(self._metrics.values())
        return [metric.collect() for metric in metrics]

    def render(self, extra: Iterable[MetricFamily] = ()) -> str:
        families = [*self.collect(), *extra]
        return "\n".join(family.render() for family in families) + "\n"


# --------------------------------------------------------- application metrics
REGISTRY = MetricsRegistry()

APP_INFO = REGISTRY.gauge(
    "docintel_app_info", "Version and environment of the running API.", ("version", "env")
)
HTTP_REQUESTS = REGISTRY.counter(
    "docintel_http_requests_total",
    "HTTP requests handled by the API, by route template and status code.",
    ("method", "route", "status"),
)
HTTP_DURATION = REGISTRY.histogram(
    "docintel_http_request_duration_seconds",
    "Duration of the HTTP requests handled by the API.",
    ("method", "route"),
)
HTTP_IN_PROGRESS = REGISTRY.gauge(
    "docintel_http_requests_in_progress", "HTTP requests currently being handled."
)
RETRIEVALS = REGISTRY.counter(
    "docintel_retrievals_total", "Hybrid retrievals, by origin (search, chat).", ("origin",)
)
RETRIEVAL_STAGE_DURATION = REGISTRY.histogram(
    "docintel_retrieval_stage_duration_seconds",
    "Duration of each stage of the retrieval pipeline (bm25, dense, fusion, rerank, total).",
    ("stage",),
)
RETRIEVAL_DEGRADED = REGISTRY.counter(
    "docintel_retrieval_degraded_total",
    "Retrievals that skipped a stage because its model server was unavailable.",
    ("stage",),
)
CHAT_ANSWERS = REGISTRY.counter(
    "docintel_chat_answers_total",
    "Chat questions by outcome: answered, refused (not in the documents) or llm_error.",
    ("outcome",),
)
LLM_DURATION = REGISTRY.histogram(
    "docintel_llm_request_duration_seconds",
    "Duration of the language model calls made by the API.",
    ("operation",),
)

LLM_TOKENS = REGISTRY.counter(
    "docintel_llm_tokens_total",
    "Tokens consumed by the language model calls of the API (cost follow-up).",
    ("type",),
)


def record_retrieval(origin: str, timings_ms: Mapping[str, float], degraded: Iterable[str]) -> None:
    RETRIEVALS.inc(origin=origin)
    for stage, milliseconds in timings_ms.items():
        RETRIEVAL_STAGE_DURATION.observe(milliseconds / 1000, stage=stage)
    for stage in degraded:
        RETRIEVAL_DEGRADED.inc(stage=stage)


def record_answer(
    *,
    answered: bool,
    model: str | None,
    llm_ms: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    CHAT_ANSWERS.inc(outcome="answered" if answered else "refused")
    if model is None:  # no passage: the refusal is produced without calling the model
        return
    LLM_DURATION.observe(llm_ms / 1000, operation="chat")
    if prompt_tokens:
        LLM_TOKENS.inc(prompt_tokens, type="prompt")
    if completion_tokens:
        LLM_TOKENS.inc(completion_tokens, type="completion")


# ------------------------------------------------------------ Celery job metrics
JOB_COUNT_KEY = "docintel:metrics:jobs:count"
JOB_SUM_KEY = "docintel:metrics:jobs:seconds_sum"
JOB_BUCKET_KEY = "docintel:metrics:jobs:seconds_bucket"
JOB_BUCKETS: tuple[float, ...] = (1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0)


class SyncHashStore(Protocol):
    def hincrby(self, name: str, key: str, amount: int = 1) -> object: ...

    def hincrbyfloat(self, name: str, key: str, amount: float = 1.0) -> object: ...


class JobMetricsStore:
    """Job outcomes and durations, aggregated in Redis by the worker processes."""

    def __init__(self, client: SyncHashStore) -> None:
        self._client = client

    def record(self, task: str, state: str, seconds: float) -> None:
        self._client.hincrby(JOB_COUNT_KEY, f"{task}|{state.lower()}", 1)
        self._client.hincrbyfloat(JOB_SUM_KEY, task, seconds)
        bound = bucket_index(JOB_BUCKETS, seconds)
        self._client.hincrby(JOB_BUCKET_KEY, f"{task}|{bound}", 1)


def job_metric_families(
    counts: Mapping[str, str], sums: Mapping[str, str], buckets: Mapping[str, str]
) -> list[MetricFamily]:
    """Families built from the three Redis hashes written by ``JobMetricsStore``.

    Malformed fields (written by another version, edited by hand) are ignored.
    """
    jobs = MetricFamily(
        "docintel_jobs_total", "counter", "Background jobs finished, by task and final state."
    )
    for field_name, raw in sorted(counts.items()):
        task, _, state = field_name.rpartition("|")
        value = _to_float(raw)
        if task and state and value is not None:
            jobs.samples.append(Sample("", {"task": task, "state": state}, value))

    per_task: dict[str, list[float]] = {}
    for field_name, raw in buckets.items():
        task, _, index = field_name.rpartition("|")
        value = _to_float(raw)
        if not task or not index.isdigit() or int(index) > len(JOB_BUCKETS) or value is None:
            continue
        per_task.setdefault(task, [0.0] * (len(JOB_BUCKETS) + 1))[int(index)] += value

    durations = MetricFamily(
        "docintel_job_duration_seconds", "histogram", "Duration of the background jobs."
    )
    for task in sorted(per_task):
        counted = per_task[task]
        durations.samples.extend(
            histogram_samples(
                {"task": task},
                JOB_BUCKETS,
                counted,
                _to_float(sums.get(task, "0")) or 0.0,
                sum(counted),
            )
        )
    return [jobs, durations]


def _to_float(raw: str | bytes | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
