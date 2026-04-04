#!/usr/bin/env bash
# run_backtest_suite.sh — Batch run all backtests in the Spacetime engine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACETIME_DIR="$(cd "${SCRIPT_DIR}/../../spacetime" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../data/backtest_results/$(date +%Y%m%d_%H%M%S)"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[backtest]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
err()  { echo -e "${RED}[  ERR ]${NC} $*" >&2; }

# Configuration.
SYMBOLS="${SYMBOLS:-AAPL MSFT SPY QQQ TSLA NVDA BTCUSDT ETHUSDT}"
TIMEFRAMES="${TIMEFRAMES:-1m 5m 15m 1h}"
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-12-31}"
PARALLEL="${PARALLEL:-4}"
PYTHON="${PYTHON:-python3}"

log "Backtest Suite"
log "  Symbols   : ${SYMBOLS}"
log "  Timeframes: ${TIMEFRAMES}"
log "  Period    : ${START_DATE} → ${END_DATE}"
log "  Results   : ${RESULTS_DIR}"

mkdir -p "${RESULTS_DIR}"

# Track results.
PASS=0
FAIL=0
declare -a FAILED_RUNS

# Run a single backtest.
run_backtest() {
    local symbol="$1"
    local timeframe="$2"
    local out_file="${RESULTS_DIR}/${symbol}_${timeframe}.json"
    local log_file="${RESULTS_DIR}/${symbol}_${timeframe}.log"

    # Try to find a backtest entry point in the spacetime project.
    local bt_script=""
    for candidate in \
        "${SPACETIME_DIR}/backtest.py" \
        "${SPACETIME_DIR}/run_backtest.py" \
        "${SPACETIME_DIR}/scripts/backtest.py" \
        "${SPACETIME_DIR}/web/backtest.py"; do
        if [ -f "$candidate" ]; then
            bt_script="$candidate"
            break
        fi
    done

    if [ -z "$bt_script" ]; then
        warn "No backtest script found in ${SPACETIME_DIR}"
        return 1
    fi

    log "Running ${symbol} ${timeframe}..."

    if $PYTHON "$bt_script" \
        --symbol "$symbol" \
        --timeframe "$timeframe" \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --output "$out_file" \
        > "$log_file" 2>&1; then
        ok "${symbol} ${timeframe}: complete → ${out_file}"
        return 0
    else
        err "${symbol} ${timeframe}: FAILED (see ${log_file})"
        return 1
    fi
}

export -f run_backtest
export RESULTS_DIR SPACETIME_DIR START_DATE END_DATE PYTHON

# Run backtests in parallel using xargs.
JOBS=()
for symbol in $SYMBOLS; do
    for tf in $TIMEFRAMES; do
        JOBS+=("${symbol}|${tf}")
    done
done

log "Running ${#JOBS[@]} backtests with parallelism=${PARALLEL}..."

printf '%s\n' "${JOBS[@]}" | xargs -P "${PARALLEL}" -I{} bash -c '
    IFS="|" read -r sym tf <<< "{}"
    run_backtest "$sym" "$tf" && echo "PASS|$sym|$tf" || echo "FAIL|$sym|$tf"
' 2>/dev/null | while IFS="|" read -r status sym tf; do
    if [ "$status" = "PASS" ]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_RUNS+=("${sym}/${tf}")
    fi
done

# ---- Aggregate results ----
log "Aggregating results..."
SUMMARY="${RESULTS_DIR}/summary.json"

$PYTHON3 - << 'PYEOF' > "$SUMMARY" 2>/dev/null || warn "Could not aggregate results (Python not available)"
import json, os, glob, sys

results_dir = os.environ.get("RESULTS_DIR", ".")
files = glob.glob(os.path.join(results_dir, "*.json"))

summary = {
    "run_at": __import__("datetime").datetime.utcnow().isoformat(),
    "total": len(files),
    "results": []
}

for f in sorted(files):
    try:
        with open(f) as fh:
            data = json.load(fh)
        name = os.path.basename(f).replace(".json", "")
        summary["results"].append({
            "name": name,
            "sharpe": data.get("sharpe_ratio", data.get("sharpe", None)),
            "total_return": data.get("total_return", None),
            "max_dd": data.get("max_drawdown", None),
            "trades": data.get("num_trades", data.get("trades", None)),
        })
    except Exception as e:
        pass

summary["results"].sort(key=lambda r: r.get("sharpe") or -999, reverse=True)
print(json.dumps(summary, indent=2))
PYEOF

echo ""
echo "════════════════════════════════════════"
echo "  Backtest Suite Complete"
echo "════════════════════════════════════════"
ok "Results: ${RESULTS_DIR}"
[ -f "$SUMMARY" ] && log "Top performers:" && python3 -c "
import json
with open('${SUMMARY}') as f:
    s = json.load(f)
for r in s['results'][:5]:
    print(f\"  {r['name']}: Sharpe={r.get('sharpe','?'):.3f}, Return={r.get('total_return','?')}\")
" 2>/dev/null || true
