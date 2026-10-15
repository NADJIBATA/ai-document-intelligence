# syntax=docker/dockerfile:1
# Backend image: FastAPI API, Celery worker and Alembic migrations (one image, three
# commands - see docker-compose.yml). No ML framework inside: the models are served by
# vLLM (or a remote API) over HTTP.

ARG PYTHON_IMAGE=python:3.13-slim-trixie

# ----------------------------------------------------------------------------- build
FROM ${PYTHON_IMAGE} AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY requirements.txt requirements.lock ./
# Every dependency ships binary wheels for linux/amd64 and linux/arm64: no compiler.
RUN pip install -r requirements.txt -c requirements.lock

# --------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime

ARG APP_VERSION=0.1.0
LABEL org.opencontainers.image.title="ai-document-intelligence-backend" \
      org.opencontainers.image.version="${APP_VERSION}"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_VERSION=${APP_VERSION}

# Unprivileged user with a fixed uid (volume permissions are predictable).
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts

USER app
EXPOSE 8000

# Liveness only (no dependency check): /ready is for load balancers and humans.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# --proxy-headers: the client IP (rate limiting, logs) comes from X-Forwarded-For set
# by nginx. The API port is never published, only nginx can reach it.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header"]
