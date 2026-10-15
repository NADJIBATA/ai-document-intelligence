#!/usr/bin/env bash
# PostgreSQL backup (custom format, compressed) of the running stack.
#   ./scripts/backup_postgres.sh                 # -> backups/docintel_<UTC timestamp>.dump
#   KEEP=14 ./scripts/backup_postgres.sh         # also deletes all but the 14 newest dumps
#
# Restore (stops the application services first):
#   docker compose stop backend worker
#   docker compose exec -T postgres sh -c 'pg_restore --clean --if-exists --no-owner \
#       -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < backups/docintel_<timestamp>.dump
#   docker compose start backend worker
#
# PostgreSQL is the source of truth: after a restore, rebuild the vectors if the Qdrant
# volume does not match (docker compose exec backend python -m scripts.reindex).
# Uploaded files live in MinIO (volume minio-data): back the volume up as well.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKUP_DIR="${BACKUP_DIR:-backups}"
KEEP="${KEEP:-0}"
mkdir -p "$BACKUP_DIR"
target="$BACKUP_DIR/docintel_$(date -u +%Y%m%dT%H%M%SZ).dump"

# Credentials are read inside the container (never on the command line of the host).
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom --compress=6' > "$target.partial"
mv "$target.partial" "$target"
echo "backup written: $target ($(du -h "$target" | cut -f1))"

if [ "$KEEP" -gt 0 ]; then
    ls -1t "$BACKUP_DIR"/docintel_*.dump | tail -n +"$((KEEP + 1))" | xargs -r rm -f --
fi
