#!/usr/bin/env bash
# Smoke test of the running stack (docker compose up -d), through nginx.
#   ./scripts/smoke_test.sh          # infrastructure, upload, indexing, search, isolation
#   ./scripts/smoke_test.sh --chat   # + a cited answer from the LLM (paid API call)
# Windows (PowerShell): docker compose exec backend python -m scripts.smoke_test --base-url http://nginx
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker compose exec -T backend python -m scripts.smoke_test --base-url http://nginx "$@"
