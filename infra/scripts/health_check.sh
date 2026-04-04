#!/usr/bin/env bash
# health_check.sh — Ping all SRFM services and report status.
set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    local name="$1"
    local url="$2"
    local expected_substring="${3:-}"

    response=$(curl -sf --max-time 5 "$url" 2>/dev/null)
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo -e "  ${RED}✗${NC} ${name}: unreachable (${url})"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if [ -n "$expected_substring" ] && ! echo "$response" | grep -q "$expected_substring"; then
        echo -e "  ${YELLOW}?${NC} ${name}: responded but missing '${expected_substring}' (${url})"
        FAIL=$((FAIL + 1))
        return 1
    fi

    echo -e "  ${GREEN}✓${NC} ${name}: OK"
    PASS=$((PASS + 1))
    return 0
}

echo ""
echo "SRFM Health Check — $(date)"
echo "────────────────────────��───────────────"

check "Gateway REST"    "http://localhost:8080/health"  '"status":"ok"'
check "Gateway WS"      "http://localhost:8080/symbols"  '"symbols"'
check "Gateway Metrics" "http://localhost:8080/metrics"  "gateway_bars_per_second"
check "Monitor Dashboard" "http://localhost:8081/"       "SRFM Monitor"
check "Monitor API"     "http://localhost:8081/api/state" '"as_of"'
check "InfluxDB"        "http://localhost:8086/health"   '"status":"pass"'
check "Grafana"         "http://localhost:3001/api/health" '"database":"ok"'
check "Prometheus"      "http://localhost:9090/-/healthy" "Prometheus Server is Healthy"
check "DuckDB Proxy"    "http://localhost:8082/health"   '"status":"ok"'
check "Arena API"       "http://localhost:8000/health"   '"status"'

echo "─────────────────────────��──────────────"
echo ""
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All ${PASS} services healthy${NC}"
    exit 0
else
    echo -e "${YELLOW}${PASS} healthy, ${RED}${FAIL} failed${NC}"
    exit 1
fi
