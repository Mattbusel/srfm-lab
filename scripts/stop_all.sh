#!/usr/bin/env bash
# scripts/stop_all.sh — Kill all services tracked in logs/*.pid

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"

if [[ ! -d "$LOGS_DIR" ]]; then
    echo "[stop_all] No logs/ directory found — nothing to stop."
    exit 0
fi

shopt -s nullglob
PID_FILES=("$LOGS_DIR"/*.pid)

if [[ ${#PID_FILES[@]} -eq 0 ]]; then
    echo "[stop_all] No .pid files found in $LOGS_DIR — nothing to stop."
    exit 0
fi

for pidfile in "${PID_FILES[@]}"; do
    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile" 2>/dev/null || true)"

    if [[ -z "$pid" ]]; then
        echo "[stop_all] $name — empty PID file, removing"
        rm -f "$pidfile"
        continue
    fi

    if kill -0 "$pid" 2>/dev/null; then
        echo "[stop_all] Stopping $name (PID $pid) ..."
        kill "$pid"
        # Give it 2s to die gracefully, then force-kill
        for i in 1 2; do
            sleep 1
            kill -0 "$pid" 2>/dev/null || break
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "[stop_all] $name did not exit cleanly — sending SIGKILL"
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "[stop_all] $name stopped."
    else
        echo "[stop_all] $name (PID $pid) was not running."
    fi

    rm -f "$pidfile"
done

echo "[stop_all] Done."
