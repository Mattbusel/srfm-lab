#!/bin/bash
# scripts/start_all_services.sh
# ==============================
# Start all SRFM Lab services in dependency order with health verification.
#
# Start order:
#   1. Elixir coordination layer  :8781
#   2. Go market-data service     :8780
#   3. C++ signal engine          (IPC / shared memory)
#   4. Python risk API            :8791
#   5. Go risk aggregator         :8792
#   6. Observability API          :9091
#   7. Metrics collector          :9090
#   8. Python live trader
#
# Usage:
#   ./scripts/start_all_services.sh [--dry-run] [--no-trader]
#
#   --dry-run     Print what would be started, do not actually start anything
#   --no-trader   Start all infrastructure services but hold off on the live trader

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
PID_DIR="${REPO_ROOT}/.pids"
ENV_FILE="${REPO_ROOT}/.env"

DRY_RUN=false
START_TRADER=true

for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=true ;;
        --no-trader)  START_TRADER=false ;;
    esac
done

mkdir -p "$LOG_DIR" "$PID_DIR"

# Load environment variables if present
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }
section() { echo -e "\n${BOLD}${CYAN}--- $1 ---${NC}"; }

# ---------------------------------------------------------------------------
# PID tracking
# ---------------------------------------------------------------------------

declare -A SERVICE_PIDS=()

save_pid() {
    local name="$1"
    local pid="$2"
    SERVICE_PIDS["$name"]="$pid"
    echo "$pid" > "${PID_DIR}/${name}.pid"
    info "Saved PID $pid for $name"
}

# ---------------------------------------------------------------------------
# Cleanup on Ctrl-C / EXIT
# ---------------------------------------------------------------------------

cleanup() {
    echo ""
    section "Shutting down all services"
    for name in "${!SERVICE_PIDS[@]}"; do
        local pid="${SERVICE_PIDS[$name]}"
        if kill -0 "$pid" 2>/dev/null; then
            info "Stopping $name (PID $pid)..."
            kill -TERM "$pid" 2>/dev/null || true
            # Give graceful shutdown 5s, then SIGKILL
            for _ in 1 2 3 4 5; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "${PID_DIR}/${name}.pid"
    done
    ok "All services stopped."
}

trap cleanup INT TERM EXIT

# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------

wait_for_http() {
    local name="$1"
    local url="$2"
    local timeout="${3:-30}"
    local interval=1
    local elapsed=0

    info "Waiting for $name at $url (timeout: ${timeout}s)..."
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf --max-time 2 "$url" > /dev/null 2>&1; then
            ok "$name is healthy"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    error "$name did not become healthy within ${timeout}s"
    return 1
}

wait_for_port() {
    local name="$1"
    local host="${2:-localhost}"
    local port="$3"
    local timeout="${4:-20}"
    local elapsed=0

    info "Waiting for $name on ${host}:${port}..."
    while [[ $elapsed -lt $timeout ]]; do
        if (echo >/dev/tcp/"$host"/"$port") 2>/dev/null; then
            ok "$name port $port is open"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    error "$name port $port not open after ${timeout}s"
    return 1
}

start_or_dry_run() {
    local name="$1"
    shift
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} Would start: $name -- $*"
        return 0
    fi
    "$@" &
    local pid=$!
    save_pid "$name" "$pid"
    return 0
}

# ---------------------------------------------------------------------------
# 1. Elixir coordination layer (:8781)
# ---------------------------------------------------------------------------

section "1/8: Elixir Coordination Layer"

if command -v mix &>/dev/null && [[ -f "${REPO_ROOT}/coordination/mix.exs" ]]; then
    start_or_dry_run "coordination" \
        bash -c "cd '${REPO_ROOT}/coordination' && mix run --no-halt \
                  >> '${LOG_DIR}/coordination.log' 2>&1"

    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_http "coordination" "http://localhost:8781/health" 45 || {
            error "Coordination layer failed to start. Check ${LOG_DIR}/coordination.log"
            exit 1
        }
    fi
else
    warn "Elixir/mix not available or coordination/mix.exs missing -- skipping coordination layer"
fi

# ---------------------------------------------------------------------------
# 2. Go market-data service (:8780)
# ---------------------------------------------------------------------------

section "2/8: Go Market-Data Service"

MARKET_DATA_BIN="${REPO_ROOT}/market-data/market-data"
if [[ ! -f "$MARKET_DATA_BIN" ]]; then
    MARKET_DATA_BIN="${REPO_ROOT}/target/release/market-data"
fi

if [[ -f "$MARKET_DATA_BIN" ]]; then
    start_or_dry_run "market_data" \
        "$MARKET_DATA_BIN" \
        --port 8780 \
        --log-level info \
        >> "${LOG_DIR}/market_data.log" 2>&1

    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_http "market-data" "http://localhost:8780/health" 20 || {
            error "Market-data service failed to start"
            exit 1
        }
    fi
elif command -v go &>/dev/null && [[ -d "${REPO_ROOT}/market-data" ]]; then
    info "Binary not found -- building market-data service..."
    (cd "${REPO_ROOT}/market-data" && go build -o market-data ./... >> "${LOG_DIR}/build.log" 2>&1)
    start_or_dry_run "market_data" \
        "${REPO_ROOT}/market-data/market-data" \
        >> "${LOG_DIR}/market_data.log" 2>&1

    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_http "market-data" "http://localhost:8780/health" 30 || {
            error "Market-data service failed to start after build"
            exit 1
        }
    fi
else
    warn "market-data binary not found and go not available -- skipping"
fi

# ---------------------------------------------------------------------------
# 3. C++ signal engine (IPC / shared memory)
# ---------------------------------------------------------------------------

section "3/8: C++ Signal Engine"

CPP_BIN="${REPO_ROOT}/cpp/build/Release/srfm_signal_engine"
if [[ ! -f "$CPP_BIN" ]]; then
    CPP_BIN="${REPO_ROOT}/cpp/build/srfm_signal_engine"
fi

if [[ -f "$CPP_BIN" ]]; then
    start_or_dry_run "signal_engine" \
        "$CPP_BIN" \
        --ipc-key srfm_signals \
        --ring-size 4096 \
        >> "${LOG_DIR}/signal_engine.log" 2>&1

    if [[ "$DRY_RUN" == "false" ]]; then
        # Signal engine communicates via IPC -- wait for sentinel file
        SENTINEL="/tmp/srfm_signal_engine.ready"
        timeout_s=15
        elapsed=0
        while [[ ! -f "$SENTINEL" ]] && [[ $elapsed -lt $timeout_s ]]; do
            sleep 1
            elapsed=$((elapsed + 1))
        done
        if [[ -f "$SENTINEL" ]]; then
            ok "Signal engine ready (sentinel found)"
        else
            warn "Signal engine sentinel not found after ${timeout_s}s -- may still be starting"
        fi
    fi
else
    warn "C++ signal engine binary not found at $CPP_BIN -- skipping"
    warn "Run: cmake --build cpp/build --config Release"
fi

# ---------------------------------------------------------------------------
# 4. Python risk API (:8791)
# ---------------------------------------------------------------------------

section "4/8: Python Risk API"

if command -v python &>/dev/null && [[ -f "${REPO_ROOT}/execution/risk/risk_api.py" ]]; then
    start_or_dry_run "risk_api" \
        python -m execution.risk.risk_api \
        >> "${LOG_DIR}/risk_api.log" 2>&1

    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_http "risk-api" "http://localhost:8791/risk/health" 25 || {
            error "Risk API failed to start. Check ${LOG_DIR}/risk_api.log"
            exit 1
        }
    fi
else
    warn "Python risk API not found -- skipping"
fi

# ---------------------------------------------------------------------------
# 5. Go risk aggregator (:8792)
# ---------------------------------------------------------------------------

section "5/8: Go Risk Aggregator"

RISK_AGG_BIN="${REPO_ROOT}/cmd/risk-aggregator/risk-aggregator"
if [[ -f "$RISK_AGG_BIN" ]]; then
    start_or_dry_run "risk_aggregator" \
        "$RISK_AGG_BIN" \
        --port 8792 \
        --risk-api http://localhost:8791 \
        >> "${LOG_DIR}/risk_aggregator.log" 2>&1

    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_http "risk-aggregator" "http://localhost:8792/health" 20 || {
            warn "Risk aggregator health check failed -- continuing anyway"
        }
    fi
else
    warn "Risk aggregator binary not found at $RISK_AGG_BIN -- skipping"
fi

# ---------------------------------------------------------------------------
# 6. Observability API (:9091)
# ---------------------------------------------------------------------------

section "6/8: Observability API"

if command -v python &>/dev/null && [[ -f "${REPO_ROOT}/infra/observability/api.py" ]]; then
    start_or_dry_run "observability_api" \
        python -m infra.observability.api \
        >> "${LOG_DIR}/observability_api.log" 2>&1

    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_http "observability-api" "http://localhost:9091/health" 15 || {
            warn "Observability API health check failed -- continuing"
        }
    fi
else
    warn "Observability API not found -- skipping"
fi

# ---------------------------------------------------------------------------
# 7. Metrics collector (:9090)
# ---------------------------------------------------------------------------

section "7/8: Metrics Collector"

METRICS_BIN="${REPO_ROOT}/cmd/metrics-collector/metrics-collector"
if [[ -f "$METRICS_BIN" ]]; then
    start_or_dry_run "metrics_collector" \
        "$METRICS_BIN" \
        --port 9090 \
        --scrape-interval 15s \
        >> "${LOG_DIR}/metrics_collector.log" 2>&1

    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_port "metrics-collector" localhost 9090 15 || {
            warn "Metrics collector not responding -- continuing"
        }
    fi
else
    warn "Metrics collector binary not found -- skipping"
fi

# ---------------------------------------------------------------------------
# 8. Python live trader
# ---------------------------------------------------------------------------

section "8/8: Python Live Trader"

TRADER_SCRIPT="${REPO_ROOT}/strategies/live/trader.py"
if [[ ! -f "$TRADER_SCRIPT" ]]; then
    TRADER_SCRIPT="${REPO_ROOT}/run_api.py"
fi

if [[ "$START_TRADER" == "true" ]]; then
    if [[ -z "${ALPACA_API_KEY:-}" ]] || [[ -z "${ALPACA_SECRET_KEY:-}" ]]; then
        error "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment or .env"
        error "Live trader NOT started. Set credentials and re-run."
        exit 1
    fi

    if command -v python &>/dev/null && [[ -f "$TRADER_SCRIPT" ]]; then
        start_or_dry_run "live_trader" \
            python "$TRADER_SCRIPT" \
            --coord-url http://localhost:8781 \
            --risk-api-url http://localhost:8791 \
            --market-data-url http://localhost:8780 \
            >> "${LOG_DIR}/live_trader.log" 2>&1

        ok "Live trader started. Tail logs: tail -f ${LOG_DIR}/live_trader.log"
    else
        warn "Live trader script not found at $TRADER_SCRIPT -- skipping"
    fi
else
    info "Live trader skipped (--no-trader flag)"
fi

# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------

section "Service Status"
if [[ "$DRY_RUN" == "true" ]]; then
    info "Dry-run complete. No services were started."
    exit 0
fi

echo ""
printf "%-25s %-10s %-8s\n" "Service" "PID" "Port"
printf "%-25s %-10s %-8s\n" "-------" "---" "----"

declare -A SERVICE_PORTS=(
    ["coordination"]="8781"
    ["market_data"]="8780"
    ["signal_engine"]="IPC"
    ["risk_api"]="8791"
    ["risk_aggregator"]="8792"
    ["observability_api"]="9091"
    ["metrics_collector"]="9090"
    ["live_trader"]="--"
)

for svc in coordination market_data signal_engine risk_api risk_aggregator \
           observability_api metrics_collector live_trader; do
    pid="${SERVICE_PIDS[$svc]:-N/A}"
    port="${SERVICE_PORTS[$svc]:-?}"
    if [[ "$pid" != "N/A" ]] && kill -0 "$pid" 2>/dev/null; then
        printf "${GREEN}%-25s${NC} %-10s %-8s\n" "$svc" "$pid" "$port"
    elif [[ "$pid" == "N/A" ]]; then
        printf "${YELLOW}%-25s${NC} %-10s %-8s\n" "$svc (skipped)" "N/A" "$port"
    else
        printf "${RED}%-25s${NC} %-10s %-8s\n" "$svc (dead)" "$pid" "$port"
    fi
done

echo ""
ok "All services started. Press Ctrl-C to stop everything."
echo ""

# Wait forever -- cleanup runs on trap
wait
