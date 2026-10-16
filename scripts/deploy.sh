#!/usr/bin/env bash
# Deployment on the server, run by .github/workflows/deploy.yml over SSH:
#
#   { printf '%s\n' "$REGISTRY_TOKEN"; cat scripts/deploy.sh; } |
#       ssh user@server 'read -r REGISTRY_TOKEN; export REGISTRY_TOKEN; bash -s -- ARGS'
#
# Arguments: <app dir> <version> <git commit> <image prefix> <registry user>
#   app dir       clone of the repository on the server, with its .env (never in Git)
#   version       image tag to run (1.2.3, main...)
#   git commit    commit whose Compose files and configuration are used
#   image prefix  ghcr.io/<owner>
# REGISTRY_TOKEN (stdin, optional): short-lived token used to pull private images; the
# registry session is closed at the end.
#
# Steps: backup of PostgreSQL, checkout of the commit, pull of the images, start (the
# migrate service applies the database migrations first), readiness check.
# Rollback: run the workflow again with the previous version.
set -euo pipefail

# Everything is inside a function: bash reads the whole script from stdin before
# running it, so commands that read stdin (docker compose exec) cannot consume it.
main() {
    exec < /dev/null

    local app_dir="$1" version="$2" commit="$3" image_prefix="$4" registry_user="${5:-}"

    cd "$app_dir"
    if [ ! -f .env ]; then
        log "missing $app_dir/.env: create it from .env.example first"
        exit 1
    fi

    local compose=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)

    if [ -n "$("${compose[@]}" ps --quiet postgres 2>/dev/null)" ]; then
        log "backup of PostgreSQL"
        KEEP="${KEEP:-14}" ./scripts/backup_postgres.sh
    fi

    log "checkout $commit"
    git fetch --quiet --tags origin
    git checkout --quiet --detach "$commit"

    export APP_VERSION="$version"
    export BACKEND_IMAGE="$image_prefix/docintel-backend"
    export FRONTEND_IMAGE="$image_prefix/docintel-frontend"

    if [ -n "${REGISTRY_TOKEN:-}" ] && [ -n "$registry_user" ]; then
        printf '%s' "$REGISTRY_TOKEN" | docker login ghcr.io -u "$registry_user" --password-stdin > /dev/null
        trap 'docker logout ghcr.io > /dev/null 2>&1 || true' EXIT
    fi

    log "pull $BACKEND_IMAGE:$version and $FRONTEND_IMAGE:$version"
    "${compose[@]}" pull --quiet backend frontend

    log "start"
    "${compose[@]}" up -d --no-build --remove-orphans

    # Readiness through nginx, like a user (PostgreSQL, Redis, MinIO, Qdrant).
    local port url
    port="$(sed -n 's/^HTTP_PORT=//p' .env | tail -n 1)"
    url="http://127.0.0.1:${port:-80}/ready"
    for _ in $(seq 1 60); do
        if curl -fsS --max-time 5 "$url" > /dev/null 2>&1; then
            log "ready: version $version is running"
            "${compose[@]}" ps
            exit 0
        fi
        sleep 5
    done

    log "not ready after 5 minutes"
    "${compose[@]}" ps
    "${compose[@]}" logs --no-color --tail=100 migrate backend worker
    exit 1
}

log() { printf '[deploy] %s\n' "$*"; }

main "$@"
