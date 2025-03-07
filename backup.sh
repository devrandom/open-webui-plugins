#!/bin/bash
set -euo pipefail

COMPOSE="docker-compose.yaml"
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/backup_${TIMESTAMP}.tar.gz"

[ -f "${COMPOSE}" ] || { echo "Error: ${COMPOSE} not found"; exit 1; }
mkdir -p "${BACKUP_DIR}"

echo "Starting backup..."
docker compose -f "${COMPOSE}" down

# Ensure containers restart even on error
trap 'docker compose -f "${COMPOSE}" up -d' EXIT

# Create backup
docker compose -f "${COMPOSE}" run --rm admin tar --exclude cache -czf - /data > "${BACKUP_FILE}"

echo "Backup created: ${BACKUP_FILE} ($(du -h "${BACKUP_FILE}" | cut -f1))"