"""Metrics registry, Prometheus text format and Celery job metrics."""

from __future__ import annotations

import math

import pytest

from app.core.metrics import (
    JOB_BUCKET_KEY,
    JOB_BUCKETS,
    JOB_COUNT_KEY,
    JOB_SUM_KEY,
    JobMetricsStore,
    MetricsRegistry,
    bucket_index,
    format_value,
    job_metric_families,
)


def test_counter_and_gauge_render_in_prometheus_format() -> None:
    registry = MetricsRegistry()
    requests = registry.counter("app_requests_total", "Requests.\nSecond line", ("route",))
    in_progress = registry.gauge("app_in_progress", "Requests in progress.")
    requests.inc(route="/api/documents/{document_id}")
    requests.inc(2, route="/api/documents/{document_id}")
    requests.inc(route='a"b\\c')
    in_progress.inc()
    in_progress.inc()
    in_progress.dec()

    text = registry.render()
    assert "# HELP app_requests_total Requests.\\nSecond line" in text
    assert "# TYPE app_requests_total counter" in text
    assert 'app_requests_total{route="/api/documents/{document_id}"} 3' in text
    assert 'app_requests_total{route="a\\"b\\\\c"} 1' in text
    assert "# TYPE app_in_progress gauge\napp_in_progress 1" in text
    assert text.endswith("\n")


def test_histogram_buckets_are_cumulative() -> None:
    registry = MetricsRegistry()
    duration = registry.histogram("op_seconds", "Duration.", ("op",), buckets=(0.1, 1.0))
    for value in (0.05, 0.1, 0.5, 3.0):
        duration.observe(value, op="x")

    text = registry.render()
    assert 'op_seconds_bucket{op="x",le="0.1"} 2' in text
    assert 'op_seconds_bucket{op="x",le="1"} 3' in text
    assert 'op_seconds_bucket{op="x",le="+Inf"} 4' in text
    assert 'op_seconds_sum{op="x"} 3.65' in text
    assert 'op_seconds_count{op="x"} 4' in text
    assert duration.count(op="x") == 4


def test_labels_must_match_the_declaration() -> None:
    registry = MetricsRegistry()
    counter = registry.counter("c_total", "C.", ("a",))
    with pytest.raises(ValueError, match="expected labels"):
        counter.inc(b="1")
    with pytest.raises(ValueError, match="only increase"):
        counter.inc(-1, a="1")
    with pytest.raises(ValueError, match="already registered"):
        registry.counter("c_total", "C.")
    with pytest.raises(ValueError, match="sorted"):
        registry.histogram("h", "H.", buckets=(2.0, 1.0))


def test_value_formatting() -> None:
    assert format_value(3.0) == "3"
    assert format_value(0.25) == "0.25"
    assert format_value(math.inf) == "+Inf"
    assert format_value(math.nan) == "NaN"
    assert bucket_index((1.0, 2.0), 1.0) == 0
    assert bucket_index((1.0, 2.0), 1.5) == 1
    assert bucket_index((1.0, 2.0), 9.0) == 2


class FakeHashes:
    """Same semantics as Redis HINCRBY / HINCRBYFLOAT with decode_responses=True."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, str]] = {}

    def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        current = int(float(self.data.setdefault(name, {}).get(key, "0")))
        self.data[name][key] = str(current + amount)
        return current + amount

    def hincrbyfloat(self, name: str, key: str, amount: float = 1.0) -> float:
        current = float(self.data.setdefault(name, {}).get(key, "0"))
        self.data[name][key] = repr(current + amount)
        return current + amount


def test_job_metrics_round_trip_through_the_redis_hashes() -> None:
    hashes = FakeHashes()
    store = JobMetricsStore(hashes)
    store.record("documents.process", "SUCCESS", 4.0)
    store.record("documents.process", "SUCCESS", 40.0)
    store.record("documents.process", "FAILURE", 0.5)
    store.record("comparisons.run", "SUCCESS", 2000.0)

    jobs, durations = job_metric_families(
        hashes.data[JOB_COUNT_KEY], hashes.data[JOB_SUM_KEY], hashes.data[JOB_BUCKET_KEY]
    )
    text = jobs.render() + "\n" + durations.render()
    assert 'docintel_jobs_total{task="documents.process",state="success"} 2' in text
    assert 'docintel_jobs_total{task="documents.process",state="failure"} 1' in text
    assert 'docintel_job_duration_seconds_bucket{task="documents.process",le="1"} 1' in text
    assert 'docintel_job_duration_seconds_bucket{task="documents.process",le="5"} 2' in text
    assert 'docintel_job_duration_seconds_bucket{task="documents.process",le="+Inf"} 3' in text
    assert 'docintel_job_duration_seconds_sum{task="documents.process"} 44.5' in text
    assert 'docintel_job_duration_seconds_count{task="documents.process"} 3' in text
    # Longer than the last bound: only in +Inf.
    last_bound = int(JOB_BUCKETS[-1])
    series = 'docintel_job_duration_seconds_bucket{task="comparisons.run",le='
    assert f'{series}"{last_bound}"}} 0' in text
    assert 'docintel_job_duration_seconds_bucket{task="comparisons.run",le="+Inf"} 1' in text


def test_malformed_job_fields_are_ignored() -> None:
    jobs, durations = job_metric_families(
        {"no-separator": "1", "task|state": "not-a-number", "ok|success": "2"},
        {"ok": "nan"},
        {"ok|99": "1", "ok|x": "1", "ok|0": "1"},
    )
    assert [sample.labels for sample in jobs.samples] == [{"task": "ok", "state": "success"}]
    sums = [sample for sample in durations.samples if sample.suffix == "_sum"]
    assert [sample.value for sample in sums] == [0.0]


def test_route_template_restores_router_prefixes() -> None:
    from types import SimpleNamespace

    from starlette.routing import compile_path

    from app.core.middleware import route_template

    def route(path: str) -> SimpleNamespace:
        return SimpleNamespace(path=path, path_regex=compile_path(path)[0])

    # Route relative to the including router (recent FastAPI) or full template.
    nested = {"route": route("/documents/{document_id}"), "path": "/api/documents/42"}
    full = {"route": route("/api/documents/{document_id}"), "path": "/api/documents/42"}
    assert route_template(nested) == "/api/documents/{document_id}"
    assert route_template(full) == "/api/documents/{document_id}"
    assert route_template({"path": "/nowhere"}) == "unmatched"
