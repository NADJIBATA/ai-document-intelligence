"""Structured JSON logging with secret redaction.

Every log line contains at least: ``timestamp``, ``level``, ``service``,
``request_id``, ``user_id`` (when known) and ``message``.
Values stored under sensitive keys (password, token, authorization, api key, ...)
and anything that looks like a JWT are replaced by ``***``.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog
from structlog.types import EventDict, Processor, WrappedLogger

from app.core.config import Settings

REDACTED = "***"
_SENSITIVE_KEY = re.compile(
    r"password|passwd|secret|token|authorization|api[_-]?key|cookie|jwt|credential",
    re.IGNORECASE,
)
_JWT_PATTERN = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
_PRESERVED_KEYS = {"event", "message", "timestamp", "level", "service", "logger"}


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        value = _JWT_PATTERN.sub(REDACTED, value)
        return _BEARER_PATTERN.sub(f"Bearer {REDACTED}", value)
    if isinstance(value, Mapping):
        return {k: _redact_item(str(k), v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact_value(item) for item in value]
    return value


def _redact_item(key: str, value: Any) -> Any:
    if key not in _PRESERVED_KEYS and _SENSITIVE_KEY.search(key):
        return REDACTED
    return _redact_value(value)


def redact_sensitive_data(
    _logger: WrappedLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """structlog processor removing secrets from the event before rendering."""
    for key in list(event_dict.keys()):
        event_dict[key] = _redact_item(key, event_dict[key])
    return event_dict


def _drop_color_message(
    _logger: WrappedLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Uvicorn adds an ANSI-colored duplicate of the message: useless in JSON logs."""
    event_dict.pop("color_message", None)
    return event_dict


def _add_service(service_name: str) -> Processor:
    def processor(
        _logger: WrappedLogger, _method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        event_dict.setdefault("service", service_name)
        return event_dict

    return processor


def configure_logging(settings: Settings) -> None:
    """Configure structlog + the standard ``logging`` module (uvicorn, sqlalchemy, celery)."""
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        _add_service(settings.service_name),
        structlog.stdlib.ExtraAdder(),
    ]

    renderer: Processor
    if settings.log_json:
        renderer = structlog.processors.JSONRenderer()
        exception_processor: Processor = structlog.processors.dict_tracebacks
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        exception_processor = structlog.processors.format_exc_info

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            exception_processor,
            _drop_color_message,
            redact_sensitive_data,
            structlog.processors.EventRenamer("message"),
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # Access logs are produced by RequestContextMiddleware (with request_id / user_id).
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    for noisy in ("httpx", "httpcore", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.db_echo else logging.WARNING
    )
