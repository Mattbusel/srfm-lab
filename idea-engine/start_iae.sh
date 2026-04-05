#!/usr/bin/env bash
# =============================================================================
# start_iae.sh  —  Start the Idea Automation Engine in dev mode (no Docker)
#
# Usage:
#   bash idea-engine/start_iae.sh           # start all services
#   bash idea-engine/start_iae.sh --stop    # kill all IAE processes
#   bash idea-engine/start_iae.sh --status  # show running services
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(dirname "$SCRIPT_DIR")"
IAE_ROOT="$SCRIPT_DIR"
DB_PATH="$IAE_ROOT/idea_engine.db"
LOGS_DIR="$IAE_ROOT/logs"
PIDS_DIR="$IAE_ROOT/.pids"

mkdir -p "$LOGS_DIR" "$PIDS_DIR"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[IAE]${NC} $*"; }
warn()  { echo -e "${YELLOW}[IAE]${NC} $*"; }
error() { echo -e "${RED}[IAE]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
pid_file() { echo "$PIDS_DIR/$1.pid"; }

is_running() {
    local pf; pf=$(pid_file "$1")
    [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null
}

start_service() {
    local name="$1"; shift
    local logfile="$LOGS_DIR/$name.log"
    if is_running "$name"; then
        warn "$name already running (PID $(cat "$(pid_file "$name")"))"
        return
    fi
    "$@" >> "$logfile" 2>&1 &
    echo $! > "$(pid_file "$name")"
    info "Started $name (PID $!) — logs: $logfile"
}

stop_service() {
    local name="$1"
    local pf; pf=$(pid_file "$name")
    if [[ -f "$pf" ]]; then
        local pid; pid=$(cat "$pf")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            info "Stopped $name (PID $pid)"
        fi
        rm -f "$pf"
    else
        warn "$name not running"
    fi
}

# ---------------------------------------------------------------------------
# --stop
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--stop" ]]; then
    stop_service "idea-bus"
    stop_service "idea-api"
    stop_service "idea-scheduler"
    stop_service "idea-dashboard"
    info "All IAE services stopped."
    exit 0
fi

# ---------------------------------------------------------------------------
# --status
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--status" ]]; then
    for svc in idea-bus idea-api idea-scheduler idea-dashboard; do
        if is_running "$svc"; then
            echo -e "  ${GREEN}●${NC} $svc (PID $(cat "$(pid_file "$svc")"))"
        else
            echo -e "  ${RED}○${NC} $svc (not running)"
        fi
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# Run migrations first
# ---------------------------------------------------------------------------
info "Running schema migrations..."
cd "$LAB_ROOT"
python -m idea_engine.db.migrate --db "$DB_PATH"

# ---------------------------------------------------------------------------
# Start services
# ---------------------------------------------------------------------------

# 1. Event bus
if command -v go &>/dev/null && [[ -f "$IAE_ROOT/bus/main.go" ]]; then
    start_service "idea-bus" \
        bash -c "cd '$IAE_ROOT/bus' && go run . --port 8768 --db '$DB_PATH'"
else
    warn "Go not found or bus/main.go missing — skipping idea-bus"
fi

sleep 1

# 2. Idea API
if command -v go &>/dev/null && [[ -f "$IAE_ROOT/idea-api/main.go" ]]; then
    start_service "idea-api" \
        bash -c "cd '$IAE_ROOT/idea-api' && go run . --port 8767 --db '$DB_PATH' --bus-url http://localhost:8768"
else
    warn "Go not found or idea-api/main.go missing — skipping idea-api"
fi

sleep 1

# 3. Scheduler
if command -v go &>/dev/null && [[ -f "$IAE_ROOT/scheduler/main.go" ]]; then
    start_service "idea-scheduler" \
        bash -c "cd '$IAE_ROOT/scheduler' && go run . --port 8769 --db '$DB_PATH' --api-url http://localhost:8767 --bus-url http://localhost:8768 --lab-path '$LAB_ROOT'"
else
    warn "Go not found or scheduler/main.go missing — skipping idea-scheduler"
fi

# 4. Dashboard (Vite dev server)
if command -v npm &>/dev/null && [[ -f "$IAE_ROOT/idea-dashboard/package.json" ]]; then
    if [[ ! -d "$IAE_ROOT/idea-dashboard/node_modules" ]]; then
        info "Installing dashboard dependencies..."
        cd "$IAE_ROOT/idea-dashboard" && npm install --silent
    fi
    start_service "idea-dashboard" \
        bash -c "cd '$IAE_ROOT/idea-dashboard' && npm run dev"
else
    warn "npm not found or package.json missing — skipping idea-dashboard"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
info "IAE stack started:"
echo "  Idea API      →  http://localhost:8767"
echo "  Event Bus     →  http://localhost:8768"
echo "  Scheduler     →  http://localhost:8769"
echo "  Dashboard     →  http://localhost:5175"
echo ""
echo "  Logs:  $LOGS_DIR/"
echo "  DB:    $DB_PATH"
echo ""
echo "  Stop:  bash $0 --stop"
echo "  Status:bash $0 --status"
