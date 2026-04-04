#!/usr/bin/env bash
# start_all.sh — Start the full SRFM trading lab stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${INFRA_DIR}/docker"

# Colours.
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[start_all]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
err()  { echo -e "${RED}[ ERR  ]${NC} $*" >&2; }

# ---- Check prerequisites ----
for cmd in docker docker-compose go; do
    if ! command -v "$cmd" &>/dev/null; then
        warn "$cmd not found — some services may not start"
    fi
done

# ---- Load .env if present ----
ENV_FILE="${INFRA_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    log "Loading environment from $ENV_FILE"
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
else
    warn ".env not found at $ENV_FILE — using defaults"
    warn "Copy infra/.env.example to infra/.env and fill in your credentials"
fi

# ---- Create data directories ----
log "Creating data directories..."
mkdir -p "${INFRA_DIR}/../spacetime/cache"
mkdir -p "${INFRA_DIR}/data/parquet"
mkdir -p "${INFRA_DIR}/data/cache"

# ---- Build Go binaries (optional, Docker handles it otherwise) ----
if [ "${BUILD_LOCAL:-false}" = "true" ]; then
    log "Building gateway..."
    (cd "${INFRA_DIR}/gateway" && go build -o /tmp/srfm-gateway ./cmd/gateway) && ok "gateway built"

    log "Building monitor..."
    (cd "${INFRA_DIR}/monitor" && go build -o /tmp/srfm-monitor ./cmd/monitor) && ok "monitor built"
fi

# ---- Start Docker Compose stack ----
log "Starting Docker Compose stack..."
cd "${DOCKER_DIR}"

COMPOSE_ARGS=""
if [ "${PROFILE:-full}" = "data-only" ]; then
    COMPOSE_ARGS="--profile data-only"
fi

docker-compose up -d --build ${COMPOSE_ARGS}

# ---- Wait for services ----
log "Waiting for services to become healthy..."

wait_for() {
    local name="$1"
    local url="$2"
    local max_retries="${3:-30}"
    local retry=0
    while [ $retry -lt $max_retries ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            ok "$name is up"
            return 0
        fi
        retry=$((retry + 1))
        sleep 2
    done
    warn "$name did not become healthy after ${max_retries} retries"
    return 1
}

wait_for "gateway"    "http://localhost:8080/health"
wait_for "monitor"    "http://localhost:8081/"
wait_for "influxdb"   "http://localhost:8086/health"
wait_for "grafana"    "http://localhost:3001/api/health"
wait_for "prometheus" "http://localhost:9090/-/healthy"

# ---- Summary ----
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       SRFM Lab Stack is running          ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║${NC}  Gateway    : http://localhost:8080       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  Monitor    : http://localhost:8081       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  Arena Web  : http://localhost:3000       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  Grafana    : http://localhost:3001       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  Prometheus : http://localhost:9090       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  InfluxDB   : http://localhost:8086       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  DuckDB     : http://localhost:8082       ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
log "Run './stop_all.sh' to stop all services"
