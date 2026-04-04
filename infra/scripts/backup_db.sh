#!/usr/bin/env bash
# backup_db.sh — Back up SQLite, parquet, InfluxDB, and cache files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPACETIME_DIR="$(cd "${INFRA_DIR}/../spacetime" && pwd)"

BACKUP_ROOT="${BACKUP_DIR:-${INFRA_DIR}/backups}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/${TIMESTAMP}"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${CYAN}[backup]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }

log "Starting backup — ${TIMESTAMP}"
log "Destination: ${BACKUP_DIR}"

mkdir -p "${BACKUP_DIR}"

# ---- SQLite databases ----
log "Backing up SQLite databases..."
find "${SPACETIME_DIR}" -name "*.db" -o -name "*.sqlite" 2>/dev/null | while read -r db; do
    rel=$(realpath --relative-to="${SPACETIME_DIR}" "$db" 2>/dev/null || echo "$(basename "$db")")
    dest="${BACKUP_DIR}/sqlite/$(dirname "$rel")"
    mkdir -p "$dest"
    cp "$db" "$dest/"
    ok "Backed up: $db"
done

# ---- Parquet/CSV data files ----
DATA_DIR="${INFRA_DIR}/data"
if [ -d "$DATA_DIR" ]; then
    log "Backing up data files..."
    tar -czf "${BACKUP_DIR}/data.tar.gz" -C "${INFRA_DIR}" data/ 2>/dev/null && ok "data/ → data.tar.gz" || warn "data/ backup failed"
fi

# ---- InfluxDB backup (via Docker) ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q srfm-influxdb; then
    log "Backing up InfluxDB..."
    mkdir -p "${BACKUP_DIR}/influxdb"
    docker exec srfm-influxdb influx backup /tmp/influx-backup \
        --org srfm \
        --token "${INFLUX_TOKEN:-srfm-dev-token}" 2>/dev/null && \
    docker cp srfm-influxdb:/tmp/influx-backup/. "${BACKUP_DIR}/influxdb/" && \
    ok "InfluxDB backed up" || warn "InfluxDB backup failed (service may be down)"
else
    warn "InfluxDB container not running — skipping"
fi

# ---- Spacetime cache ----
CACHE_DIR="${SPACETIME_DIR}/cache"
if [ -d "$CACHE_DIR" ]; then
    log "Backing up spacetime cache..."
    cp -r "$CACHE_DIR" "${BACKUP_DIR}/spacetime_cache" && ok "spacetime cache backed up"
fi

# ---- Bar cache JSON ----
CACHE_FILE="${INFRA_DIR}/data/cache.json"
if [ -f "$CACHE_FILE" ]; then
    cp "$CACHE_FILE" "${BACKUP_DIR}/bar_cache.json" && ok "bar_cache.json backed up"
fi

# ---- Compute size ----
TOTAL_SIZE=$(du -sh "${BACKUP_DIR}" 2>/dev/null | cut -f1)

echo ""
ok "Backup complete: ${BACKUP_DIR} (${TOTAL_SIZE})"

# ---- Prune old backups (keep last 30) ----
KEEP_LAST="${KEEP_BACKUPS:-30}"
log "Pruning backups (keeping last ${KEEP_LAST})..."
ls -1dt "${BACKUP_ROOT}"/*/ 2>/dev/null | tail -n +$((KEEP_LAST + 1)) | while read -r old; do
    rm -rf "$old"
    warn "Removed old backup: $old"
done
