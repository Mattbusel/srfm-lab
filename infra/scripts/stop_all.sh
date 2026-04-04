#!/usr/bin/env bash
# stop_all.sh — Gracefully stop the SRFM trading lab stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(cd "${SCRIPT_DIR}/../docker" && pwd)"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${CYAN}[stop_all]${NC} $*"; }
ok()  { echo -e "${GREEN}[  OK  ]${NC} $*"; }

log "Stopping SRFM stack..."

cd "${DOCKER_DIR}"

# Send SIGTERM to allow graceful shutdown.
docker-compose stop --timeout 30

# Optionally remove containers (not volumes).
if [ "${REMOVE_CONTAINERS:-false}" = "true" ]; then
    log "Removing containers..."
    docker-compose rm -f
    ok "Containers removed"
fi

ok "All services stopped"

if [ "${SHOW_LOGS:-false}" = "true" ]; then
    log "Recent logs:"
    docker-compose logs --tail=20
fi
