#!/usr/bin/env bash
# scripts/health_check.sh — One-shot status report
# Prints which services are up, their PID, and the last line of their log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"

# Service name → health URL (empty = process-only check)
declare -A HEALTH_URLS=(
    [market-data]="http://localhost:8780/health"
    [coordination]="http://localhost:8781/health"
    [bridge]="http://localhost:8783/health"
    [autonomous-loop]=""
    [live-trader]=""
)

SERVICE_NAMES=(market-data coordination bridge autonomous-loop live-trader)

echo ""
echo "SRFM Lab — Health Report  $(date '+%Y-%m-%d %H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-22s %-6s %-8s %-10s  %s\n" "SERVICE" "PID" "PROCESS" "HTTP" "LAST LOG LINE"
printf "%-22s %-6s %-8s %-10s  %s\n" "-------" "---" "-------" "----" "-------------"

for name in "${SERVICE_NAMES[@]}"; do
    pidfile="$LOGS_DIR/$name.pid"
    logfile="$LOGS_DIR/$name.log"
    url="${HEALTH_URLS[$name]:-}"

    # PID / process status
    if [[ -f "$pidfile" ]]; then
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            proc_status="UP"
        else
            proc_status="DEAD"
            pid="$pid(!)"
        fi
    else
        pid="-"
        proc_status="NO-PID"
    fi

    # HTTP health check
    if [[ -n "$url" ]]; then
        http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null || echo "ERR")
        if [[ "$http_code" == "200" ]] || [[ "$http_code" == "204" ]]; then
            http_status="OK($http_code)"
        else
            http_status="FAIL($http_code)"
        fi
    else
        http_status="(no check)"
    fi

    # Last log line
    if [[ -f "$logfile" ]]; then
        last_log="$(tail -1 "$logfile" 2>/dev/null | cut -c1-80 || echo '')"
    else
        last_log="(no log)"
    fi

    printf "%-22s %-6s %-8s %-10s  %s\n" "$name" "$pid" "$proc_status" "$http_status" "$last_log"
done

echo ""

# Supervisor status
supervisor_pid="$LOGS_DIR/supervisor.pid"
if [[ -f "$supervisor_pid" ]]; then
    spid="$(cat "$supervisor_pid")"
    if kill -0 "$spid" 2>/dev/null; then
        echo "Supervisor (port 8790): RUNNING (PID $spid)  →  http://localhost:8790/status"
    else
        echo "Supervisor: DEAD (was PID $spid)"
    fi
else
    echo "Supervisor: not started"
fi
echo ""
