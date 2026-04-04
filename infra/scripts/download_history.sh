#!/usr/bin/env bash
# download_history.sh — Download historical bar data for configured symbols.
# Uses the Alpaca Data API v2 for stocks and the Binance REST API for crypto.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${INFRA_DIR}/data/parquet"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${CYAN}[download]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }

# Configuration (override via environment or .env).
ALPACA_API_KEY="${ALPACA_API_KEY:-}"
ALPACA_SECRET="${ALPACA_SECRET:-}"
ALPACA_BASE="${ALPACA_BASE:-https://data.alpaca.markets}"
SYMBOLS="${SYMBOLS:-AAPL,MSFT,SPY,QQQ,TSLA,NVDA,AMZN,GOOGL}"
CRYPTO_SYMBOLS="${CRYPTO_SYMBOLS:-BTCUSDT,ETHUSDT}"
TIMEFRAMES="${TIMEFRAMES:-1m,5m,15m,1h,1d}"
START_DATE="${START_DATE:-$(date -d '365 days ago' +%Y-%m-%d 2>/dev/null || date -v-365d +%Y-%m-%d)}"
END_DATE="${END_DATE:-$(date +%Y-%m-%d)}"

# Load .env.
if [ -f "${INFRA_DIR}/.env" ]; then
    set -o allexport
    source "${INFRA_DIR}/.env"
    set +o allexport
fi

mkdir -p "${DATA_DIR}"

# ---- Alpaca stock data ----
download_alpaca_bars() {
    local symbol="$1"
    local timeframe="$2"
    local out_dir="${DATA_DIR}/${symbol}/${timeframe}"
    mkdir -p "$out_dir"

    log "Downloading ${symbol} ${timeframe} from Alpaca..."

    # Map to Alpaca timeframe format.
    local alpaca_tf
    case "$timeframe" in
        1m)  alpaca_tf="1Min"  ;;
        5m)  alpaca_tf="5Min"  ;;
        15m) alpaca_tf="15Min" ;;
        1h)  alpaca_tf="1Hour" ;;
        1d)  alpaca_tf="1Day"  ;;
        *)   warn "Unknown timeframe $timeframe"; return ;;
    esac

    local page_token=""
    local file="${out_dir}/${START_DATE}_${END_DATE}.csv"
    local wrote_header=false

    while true; do
        local url="${ALPACA_BASE}/v2/stocks/${symbol}/bars?timeframe=${alpaca_tf}&start=${START_DATE}T00:00:00Z&end=${END_DATE}T23:59:59Z&limit=10000&feed=iex"
        if [ -n "$page_token" ]; then
            url="${url}&page_token=${page_token}"
        fi

        response=$(curl -sf \
            -H "APCA-API-KEY-ID: ${ALPACA_API_KEY}" \
            -H "APCA-API-SECRET-KEY: ${ALPACA_SECRET}" \
            "$url" 2>/dev/null) || { warn "Failed to download ${symbol} ${timeframe}"; return; }

        if [ "$wrote_header" = false ]; then
            echo "timestamp,symbol,open,high,low,close,volume,source" > "$file"
            wrote_header=true
        fi

        # Parse JSON with Python (always available in Docker).
        echo "$response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
bars = data.get('bars', [])
for b in bars:
    print(f\"{b['t']},${symbol},{b['o']},{b['h']},{b['l']},{b['c']},{b['v']},alpaca\")
" >> "$file" 2>/dev/null || true

        page_token=$(echo "$response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('next_page_token', ''))
" 2>/dev/null)

        if [ -z "$page_token" ]; then
            break
        fi
    done

    local count
    count=$(wc -l < "$file" 2>/dev/null || echo 0)
    ok "${symbol} ${timeframe}: $(( count - 1 )) bars -> ${file}"
}

# ---- Binance crypto data ----
download_binance_bars() {
    local symbol="$1"    # e.g. BTCUSDT
    local timeframe="$2"
    local out_dir="${DATA_DIR}/${symbol}/${timeframe}"
    mkdir -p "$out_dir"

    # Map timeframe to Binance interval.
    local binance_tf
    case "$timeframe" in
        1m)  binance_tf="1m"  ;;
        5m)  binance_tf="5m"  ;;
        15m) binance_tf="15m" ;;
        1h)  binance_tf="1h"  ;;
        1d)  binance_tf="1d"  ;;
        *)   warn "Unknown timeframe $timeframe"; return ;;
    esac

    log "Downloading ${symbol} ${timeframe} from Binance..."

    local file="${out_dir}/${START_DATE}_${END_DATE}.csv"
    echo "timestamp,symbol,open,high,low,close,volume,source" > "$file"

    # Convert start/end to ms.
    local start_ms end_ms
    start_ms=$(date -d "${START_DATE}" +%s 2>/dev/null || date -j -f "%Y-%m-%d" "${START_DATE}" +%s 2>/dev/null)
    start_ms=$((start_ms * 1000))
    end_ms=$(date -d "${END_DATE}" +%s 2>/dev/null || date -j -f "%Y-%m-%d" "${END_DATE}" +%s 2>/dev/null)
    end_ms=$((end_ms * 1000))

    local current_ms=$start_ms
    local batch=1000

    while [ $current_ms -lt $end_ms ]; do
        response=$(curl -sf \
            "https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${binance_tf}&startTime=${current_ms}&limit=${batch}" \
            2>/dev/null) || { warn "Binance API error for ${symbol}"; break; }

        count=$(echo "$response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for k in data:
    import datetime
    ts = datetime.datetime.utcfromtimestamp(k[0]/1000).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f'{ts},${symbol},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},binance')
print(len(data), file=sys.stderr)
" >> "$file" 2>/tmp/binance_count) || break

        fetched=$(cat /tmp/binance_count 2>/dev/null || echo 0)
        if [ "$fetched" -eq 0 ]; then
            break
        fi

        # Advance by batch * interval_ms.
        case "$timeframe" in
            1m)  interval_ms=60000 ;;
            5m)  interval_ms=300000 ;;
            15m) interval_ms=900000 ;;
            1h)  interval_ms=3600000 ;;
            1d)  interval_ms=86400000 ;;
            *)   interval_ms=60000 ;;
        esac
        current_ms=$((current_ms + batch * interval_ms))

        sleep 0.1  # Rate limit.
    done

    local line_count
    line_count=$(wc -l < "$file")
    ok "${symbol} ${timeframe}: $((line_count - 1)) bars -> ${file}"
}

# ---- Main ----
log "Historical data download"
log "  Symbols   : ${SYMBOLS}"
log "  Crypto    : ${CRYPTO_SYMBOLS}"
log "  Timeframes: ${TIMEFRAMES}"
log "  Date range: ${START_DATE} → ${END_DATE}"
log "  Output dir: ${DATA_DIR}"
echo ""

# Download equity symbols via Alpaca.
if [ -n "$ALPACA_API_KEY" ]; then
    IFS=',' read -ra SYMS <<< "$SYMBOLS"
    IFS=',' read -ra TFS  <<< "$TIMEFRAMES"
    for sym in "${SYMS[@]}"; do
        for tf in "${TFS[@]}"; do
            download_alpaca_bars "$sym" "$tf"
        done
    done
else
    warn "ALPACA_API_KEY not set — skipping equity download"
fi

# Download crypto via Binance.
IFS=',' read -ra CRYPTO <<< "$CRYPTO_SYMBOLS"
IFS=',' read -ra TFS    <<< "$TIMEFRAMES"
for sym in "${CRYPTO[@]}"; do
    for tf in "${TFS[@]}"; do
        download_binance_bars "$sym" "$tf"
    done
done

echo ""
ok "Download complete. Files in: ${DATA_DIR}"
log "Run 'tsdb import ${DATA_DIR}/**/*.csv' to load into InfluxDB"
