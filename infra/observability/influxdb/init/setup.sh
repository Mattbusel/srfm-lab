#!/usr/bin/env bash
# influxdb/init/setup.sh
#
# Auto-runs inside the InfluxDB container on first boot via
# docker-entrypoint-initdb.d.  The DOCKER_INFLUXDB_INIT_* env vars handle
# the primary setup (org, bucket, admin token, retention); this script adds
# extra buckets, tasks, and verifies the setup is complete.
#
# Environment variables are inherited from docker-compose.yml.

set -euo pipefail

INFLUX_CMD="influx"
TOKEN="${DOCKER_INFLUXDB_INIT_ADMIN_TOKEN:-larsa-super-secret-token}"
ORG="${DOCKER_INFLUXDB_INIT_ORG:-srfm}"
HOST="http://localhost:8086"

log() { echo "[larsa-influx-init] $*"; }

wait_for_influx() {
    local max=30
    local i=0
    log "Waiting for InfluxDB to be ready..."
    until $INFLUX_CMD ping --host "$HOST" >/dev/null 2>&1; do
        i=$((i+1))
        if [ $i -ge $max ]; then
            log "ERROR: InfluxDB did not start within ${max} attempts"
            exit 1
        fi
        sleep 2
    done
    log "InfluxDB is ready."
}

create_bucket_if_missing() {
    local name="$1"
    local retention="${2:-90d}"
    if $INFLUX_CMD bucket list \
        --host "$HOST" \
        --token "$TOKEN" \
        --org "$ORG" \
        --name "$name" >/dev/null 2>&1; then
        log "Bucket '$name' already exists — skipping."
    else
        $INFLUX_CMD bucket create \
            --host "$HOST" \
            --token "$TOKEN" \
            --org "$ORG" \
            --name "$name" \
            --retention "$retention"
        log "Created bucket '$name' (retention=$retention)."
    fi
}

# ── Wait for InfluxDB ─────────────────────────────────────────────────────────
wait_for_influx

# ── Primary bucket is created by DOCKER_INFLUXDB_INIT_* — just verify ────────
log "Verifying primary bucket 'larsa_metrics'..."
create_bucket_if_missing "larsa_metrics"    "90d"

# ── Extra buckets ─────────────────────────────────────────────────────────────
# Raw trade events — shorter retention, high write frequency
create_bucket_if_missing "larsa_trades"     "365d"

# Regime snapshots — lower write frequency, longer retention
create_bucket_if_missing "larsa_regime"     "365d"

# ── Create a read-only API token for Grafana ──────────────────────────────────
log "Creating Grafana read-only token..."
GRAFANA_TOKEN=$($INFLUX_CMD auth create \
    --host "$HOST" \
    --token "$TOKEN" \
    --org "$ORG" \
    --description "Grafana read-only" \
    --read-buckets \
    2>/dev/null | awk 'NR==2{print $3}' || true)

if [ -n "$GRAFANA_TOKEN" ]; then
    log "Grafana token created: ${GRAFANA_TOKEN:0:16}..."
    # Write to a file so docker-compose can pick it up if needed
    echo "$GRAFANA_TOKEN" > /var/lib/influxdb2/.grafana_token
else
    log "WARN: Could not create Grafana token (may already exist). Using admin token."
fi

# ── Print summary ─────────────────────────────────────────────────────────────
log "====== InfluxDB setup complete ======"
log "  Org:     $ORG"
log "  Host:    $HOST"
log "  Buckets: larsa_metrics, larsa_trades, larsa_regime"
log "  Admin token: ${TOKEN:0:16}..."
log "  Grafana URL: http://influxdb:8086"
log "============================================"
