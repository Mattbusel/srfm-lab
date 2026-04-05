#!/usr/bin/env bash
# scripts/start_all.sh — SRFM Lab process supervisor (bash)
# Usage:
#   ./scripts/start_all.sh          # start all services
#   ./scripts/start_all.sh stop     # kill all managed PIDs
#   ./scripts/start_all.sh status   # show which are running
#   ./scripts/start_all.sh restart  # stop then start

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"
ENV_FILE="$REPO_ROOT/tools/.env"
HEALTH_INTERVAL=30   # seconds between health-check pings

mkdir -p "$LOGS_DIR"

# ── Load environment ──────────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[start_all] Loaded env from $ENV_FILE"
fi

# ── Service definitions ───────────────────────────────────────────────────────
# FORMAT: NAME|WORKDIR|COMMAND|HEALTH_URL
declare -a SERVICES=(
    "market-data|$REPO_ROOT/market-data|./market-data.exe|http://localhost:8780/health"
    "coordination|$REPO_ROOT/coordination|mix run --no-halt|http://localhost:8781/health"
    "bridge|$REPO_ROOT|python bridge/heartbeat.py|http://localhost:8783/health"
    "autonomous-loop|$REPO_ROOT|python -m idea_engine.autonomous_loop.orchestrator|"
    "live-trader|$REPO_ROOT|python tools/live_trader_alpaca.py|"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
pid_file()  { echo "$LOGS_DIR/$1.pid"; }
log_file()  { echo "$LOGS_DIR/$1.log"; }

is_running() {
    local name="$1"
    local pidfile
    pidfile="$(pid_file "$name")"
    [[ -f "$pidfile" ]] || return 1
    local pid
    pid="$(cat "$pidfile")"
    kill -0 "$pid" 2>/dev/null
}

start_service() {
    local name workdir cmd health
    IFS='|' read -r name workdir cmd health <<< "$1"

    if is_running "$name"; then
        echo "[start_all] $name already running (PID $(cat "$(pid_file "$name")"))"
        return 0
    fi

    echo "[start_all] Starting $name ..."
    mkdir -p "$workdir"

    # Use bash -c so the command string is expanded properly
    bash -c "cd '$workdir' && exec $cmd" \
        >> "$(log_file "$name")" 2>&1 &
    local pid=$!
    echo "$pid" > "$(pid_file "$name")"
    echo "[start_all] $name started (PID $pid) — log: $(log_file "$name")"
}

stop_service() {
    local name="$1"
    local pidfile
    pidfile="$(pid_file "$name")"
    if [[ ! -f "$pidfile" ]]; then
        echo "[start_all] $name — no PID file, skipping"
        return 0
    fi
    local pid
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
        echo "[start_all] Stopping $name (PID $pid) ..."
        kill "$pid" && sleep 1
        kill -9 "$pid" 2>/dev/null || true
    else
        echo "[start_all] $name was not running"
    fi
    rm -f "$pidfile"
}

check_health() {
    local name health="$2"
    name="$1"
    if [[ -z "$health" ]]; then
        # No URL — just check process liveness
        if is_running "$name"; then
            echo "[health] $name  UP (no HTTP check)"
        else
            echo "[health] $name  DOWN — restarting ..."
            start_service "$3"
        fi
        return
    fi
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$health" 2>/dev/null || echo "000")
    if [[ "$http_code" == "200" ]] || [[ "$http_code" == "204" ]]; then
        echo "[health] $name  UP ($http_code)"
    else
        if is_running "$name"; then
            echo "[health] $name  DEGRADED (HTTP $http_code) — process alive, leaving it"
        else
            echo "[health] $name  DOWN (HTTP $http_code) — restarting ..."
            start_service "$3"
        fi
    fi
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd="${1:-start}"

case "$cmd" in
    start)
        for svc in "${SERVICES[@]}"; do
            start_service "$svc"
        done
        echo ""
        echo "[start_all] All services started.  Health-checking every ${HEALTH_INTERVAL}s."
        echo "[start_all] Press Ctrl-C to stop the health-check loop (services keep running)."
        echo "[start_all] To stop everything: ./scripts/start_all.sh stop"
        echo ""

        # Health-check loop (runs in foreground so the script is the monitor)
        while true; do
            sleep "$HEALTH_INTERVAL"
            echo "--- health check $(date '+%Y-%m-%d %H:%M:%S') ---"
            for svc in "${SERVICES[@]}"; do
                IFS='|' read -r n _wd _cmd h <<< "$svc"
                check_health "$n" "$h" "$svc"
            done
        done
        ;;

    stop)
        for svc in "${SERVICES[@]}"; do
            IFS='|' read -r name _ _ _ <<< "$svc"
            stop_service "$name"
        done
        echo "[start_all] All services stopped."
        ;;

    restart)
        "$0" stop
        sleep 2
        exec "$0" start
        ;;

    status)
        echo ""
        printf "%-20s %-8s %s\n" "SERVICE" "STATUS" "PID"
        printf "%-20s %-8s %s\n" "-------" "------" "---"
        for svc in "${SERVICES[@]}"; do
            IFS='|' read -r name _ _ _ <<< "$svc"
            if is_running "$name"; then
                pid="$(cat "$(pid_file "$name")")"
                printf "%-20s %-8s %s\n" "$name" "UP" "$pid"
            else
                printf "%-20s %-8s %s\n" "$name" "DOWN" "-"
            fi
        done
        echo ""
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
