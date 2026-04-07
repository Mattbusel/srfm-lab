# SRFM Lab Operations Runbook

**System:** SRFM Trading Lab (IAE Architecture)  
**Maintained by:** Ops / Research  
**Last updated:** 2026-04-06

---

## Table of Contents

1. [Daily Startup Checklist](#1-daily-startup-checklist)
2. [Common Failure Scenarios](#2-common-failure-scenarios)
   - 2.1 [Alpaca circuit breaker open](#21-alpaca-circuit-breaker-open)
   - 2.2 [Parameter rollback triggered](#22-parameter-rollback-triggered)
   - 2.3 [Signal engine not writing IPC](#23-signal-engine-not-writing-ipc)
   - 2.4 [Memory warning from live trader](#24-memory-warning-from-live-trader)
   - 2.5 [Go market-data service unresponsive](#25-go-market-data-service-unresponsive)
   - 2.6 [Risk API returning stale VaR](#26-risk-api-returning-stale-var)
3. [Performance Degradation Playbook](#3-performance-degradation-playbook)
4. [Emergency Procedures](#4-emergency-procedures)
   - 4.1 [Flatten all positions immediately](#41-flatten-all-positions-immediately)
   - 4.2 [Kill switch -- full system halt](#42-kill-switch----full-system-halt)
5. [Backup and Recovery](#5-backup-and-recovery)
   - 5.1 [SQLite DB backup](#51-sqlite-db-backup)
   - 5.2 [Parameter snapshot restore](#52-parameter-snapshot-restore)
   - 5.3 [Full system restore procedure](#53-full-system-restore-procedure)
6. [Service Reference](#6-service-reference)

---

## 1. Daily Startup Checklist

Run every trading day before market open (9:15 AM ET or earlier).

### Pre-market (before 9:15 AM ET)

- [ ] **Check system health**

  ```bash
  python scripts/health_check.py --pretty
  ```

  Expected: `"status": "ok"` with all critical services green.  
  If degraded: resolve before proceeding (see Section 2).

- [ ] **Verify Alpaca account status**

  ```bash
  python scripts/health_check.py --check alpaca --pretty
  ```

  Confirm `account_status: ACTIVE` and buying power is plausible.

- [ ] **Confirm circuit breakers are closed**

  ```bash
  curl -s http://localhost:8781/health | python -m json.tool
  ```

  All circuits must show `"state": "closed"`.

- [ ] **Check parameter snapshot is current**

  ```bash
  curl -s http://localhost:8781/params/current | python -m json.tool
  ```

  Verify key parameters (BH_DECAY, SIGNAL_ENTRY_THRESHOLD) match last approved values.  
  Check the coordination journal if unsure: `tail -100 logs/coordination.log`.

- [ ] **Review overnight DB growth**

  ```bash
  sqlite3 execution/live_trades.db "SELECT COUNT(*) FROM live_trades WHERE ts > datetime('now', '-12 hours')"
  ```

  Unexpected overnight trades (> 5) warrant investigation before market open.

- [ ] **Check disk space**

  ```bash
  df -h . logs/
  ```

  Warn if either is > 80% used. Rotate logs if needed: `scripts/rotate_logs.sh`.

- [ ] **Verify signal engine is running**

  ```bash
  ls -la /tmp/srfm_signal_engine.ready  # sentinel file
  python scripts/health_check.py --check signal_engine
  ```

- [ ] **Start services if not already running**

  ```bash
  ./scripts/start_all_services.sh --no-trader
  # -- review service status table --
  # If all green, start trader:
  ./scripts/start_all_services.sh
  ```

### Market hours (9:30 AM -- 4:00 PM ET)

- [ ] Monitor Grafana dashboard: `http://localhost:3000`
- [ ] Spot-check live P&L against expected range every hour
- [ ] If any alert fires in logs/alerts/, investigate immediately

### Post-market (after 4:00 PM ET)

- [ ] Run end-of-day DB backup: `./scripts/backup_db.sh`
- [ ] Review session Sharpe and max drawdown in Grafana
- [ ] Export trade log: `python scripts/export_trades.py --date today`
- [ ] Check if any parameter proposal auto-rolled-back (search logs):

  ```bash
  grep "rollback_triggered" logs/coordination.log | tail -10
  ```

---

## 2. Common Failure Scenarios

### 2.1 Alpaca Circuit Breaker Open

**Symptom:** Orders are not being submitted. Coordination layer health shows `"alpaca": "open"`.  
Live trader logs contain: `"circuit_breaker=open for alpaca, blocking order"`.

**Diagnosis:**

1. Check when the breaker opened:

   ```bash
   grep "circuit.*alpaca.*open" logs/coordination.log | tail -5
   ```

2. Check what triggered it -- usually too many consecutive failed API calls:

   ```bash
   grep "alpaca.*error\|alpaca.*timeout\|alpaca.*429" logs/live_trader.log | tail -20
   ```

3. Verify Alpaca is actually reachable:

   ```bash
   curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
        -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
        https://paper-api.alpaca.markets/v2/account
   ```

**Remediation:**

- If Alpaca is reachable and the error was transient (rate limit, brief outage):

  ```bash
  curl -X POST http://localhost:8781/circuit/alpaca/reset
  ```

- If Alpaca is returning 403 (auth failure): re-check `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`.

- If Alpaca is down: do NOT reset the breaker. Wait for service restoration, then reset and monitor.

- After reset, verify the breaker stays closed for at least 5 minutes before leaving it unattended.

**Prevention:** Circuit breaker thresholds are set in `coordination/config/circuit_breakers.exs`.  
Default: opens after 5 consecutive failures within 60s. Tune if you see spurious trips on slow API days.

---

### 2.2 Parameter Rollback Triggered

**Symptom:** Coordination log contains `"rollback_triggered"`. Strategy may be trading with stale parameters.

**Diagnosis:**

1. Identify the rollback event:

   ```bash
   grep "rollback_triggered\|poor_performance" logs/coordination.log | tail -10
   ```

2. Check what the 4-hour rolling Sharpe was when it triggered:

   ```bash
   curl -s http://localhost:8781/metrics/sharpe
   ```

3. Compare current vs. snapshot parameters:

   ```bash
   curl -s http://localhost:8781/params/current
   # Compare against the last approved params in config/param_snapshots/
   ls -lt config/param_snapshots/
   ```

4. Look for the root cause -- was the poor Sharpe driven by a bad data feed, fat-finger size, or a genuine regime change?

   ```bash
   python scripts/generate_factor_report.py --lookback 4h
   ```

**Remediation:**

- If the rollback was correct (genuine poor performance): leave the restored params in place and investigate strategy performance before re-proposing.

- If the rollback was a false positive (bad data feed, data gap):

  ```bash
  # Re-propose the parameters that were rolled back
  curl -X POST http://localhost:8781/params/propose \
       -H "Content-Type: application/json" \
       -d '{"params": {"BH_DECAY": 0.95, "SIGNAL_ENTRY_THRESHOLD": 0.65}}'
  ```

- Adjust the rollback threshold if consistently firing on acceptable drawdowns:
  Edit `coordination/lib/srfm_coordination/parameter_coordinator.ex`, field `@rollback_sharpe_threshold`.

---

### 2.3 Signal Engine Not Writing IPC

**Symptom:** `health_check.py --check signal_engine` returns `down` or `degraded`.  
Live trader logs: `"IPC read timeout"` or `"ring buffer stale"`.

**Diagnosis:**

1. Check if the process is running:

   ```bash
   pgrep -a srfm_signal_engine
   cat .pids/signal_engine.pid 2>/dev/null
   ```

2. Check for crash dumps or OOM:

   ```bash
   tail -50 logs/signal_engine.log
   dmesg | grep -i "oom\|killed" | tail -5
   ```

3. Check IPC objects:

   ```bash
   ls -la /dev/shm/srfm_signals 2>/dev/null || echo "ring buffer file missing"
   ls -la /tmp/srfm_signal_engine.ready 2>/dev/null || echo "sentinel missing"
   ```

4. Try running the engine manually in foreground:

   ```bash
   ./cpp/build/Release/srfm_signal_engine --ipc-key srfm_signals --ring-size 4096 --verbose
   ```

**Remediation:**

- If crashed (no process found): restart it

  ```bash
  rm -f /tmp/srfm_signal_engine.ready
  ./cpp/build/Release/srfm_signal_engine \
      --ipc-key srfm_signals --ring-size 4096 \
      >> logs/signal_engine.log 2>&1 &
  # Wait for sentinel
  sleep 3
  python scripts/health_check.py --check signal_engine
  ```

- If OOM killed: check memory usage with `free -h`. Consider reducing ring buffer size:
  `--ring-size 2048`. Also check for memory leaks in recent C++ builds.

- If the binary is missing: rebuild with `cmake --build cpp/build --config Release`.

- If the engine starts but dies immediately: check `logs/signal_engine.log` for missing market data feed.
  The engine requires the Go market-data service on :8780. Start market-data first.

---

### 2.4 Memory Warning from Live Trader

**Symptom:** Log message: `"WARNING: RSS memory exceeding threshold"` or Python OOM.

**Diagnosis:**

1. Check current memory usage:

   ```bash
   # PID from pid file
   TRADER_PID=$(cat .pids/live_trader.pid 2>/dev/null)
   ps -o pid,rss,vsz,comm -p "$TRADER_PID" 2>/dev/null
   ```

2. Identify which buffer is growing -- common culprits:

   - `_price_buffer` in BHEngine accumulating unboundedly
   - Historical return series for VaR computation not capped
   - Julia analytics retaining large arrays

   ```bash
   python scripts/memory_profile.py --pid "$TRADER_PID"
   ```

3. Check the HIST_WINDOW setting in `execution/risk/live_var.py` (default 252 days).
   If running for > 1 year without restart, the rolling window may have grown.

**Remediation:**

- Graceful restart of the live trader (positions are NOT closed, only the process restarts):

  ```bash
  kill -HUP $(cat .pids/live_trader.pid)
  # Trader should catch SIGHUP and restart cleanly
  ```

- If SIGHUP does not work, use a full restart:

  ```bash
  # 1. Pause new entries via coordination layer
  curl -X POST http://localhost:8781/params/propose \
       -H "Content-Type: application/json" \
       -d '{"params": {"SIGNAL_ENTRY_THRESHOLD": 0.99}}'  # effectively disables entries
  # 2. Wait for current bar to complete (max 1 minute)
  sleep 60
  # 3. Restart trader
  kill $(cat .pids/live_trader.pid)
  python run_api.py >> logs/live_trader.log 2>&1 &
  # 4. Restore normal threshold
  curl -X POST http://localhost:8781/params/propose \
       -H "Content-Type: application/json" \
       -d '{"params": {"SIGNAL_ENTRY_THRESHOLD": 0.65}}'
  ```

---

### 2.5 Go Market-Data Service Unresponsive

**Symptom:** `health_check.py --check market_data` returns `down`. Live trader logs: `"market data feed timeout"`.

**Diagnosis:**

```bash
curl -sf http://localhost:8780/health || echo "service down"
tail -30 logs/market_data.log
pgrep -a market-data
```

**Remediation:**

```bash
# Kill old process
pkill -f market-data || true
sleep 2

# Restart
./market-data/market-data --port 8780 --log-level info >> logs/market_data.log 2>&1 &

# Verify
sleep 3
curl -sf http://localhost:8780/health && echo "market-data OK"
```

---

### 2.6 Risk API Returning Stale VaR

**Symptom:** `/risk/portfolio` returns a VaR with `computed_at` timestamp > 10 minutes old.

**Diagnosis:**

```bash
curl -s http://localhost:8791/risk/portfolio | python -m json.tool | grep computed_at
tail -30 logs/risk_api.log
```

**Remediation:**

- Force a VaR recalculation:

  ```bash
  curl -X POST http://localhost:8791/risk/recalculate
  ```

- If the risk API is stuck, restart it:

  ```bash
  kill $(cat .pids/risk_api.pid)
  python -m execution.risk.risk_api >> logs/risk_api.log 2>&1 &
  ```

---

## 3. Performance Degradation Playbook

Use this when rolling Sharpe falls significantly below historical baseline without an obvious external cause.

### Step 1 -- Confirm the degradation is real

```bash
# Check last 5 trading days
python scripts/run_strategy.py --mode backtest --lookback 5d --report
```

Compare current Sharpe to the 30-day rolling baseline in `results/`.

### Step 2 -- Check data quality

Bad data (stale prices, bad fills) is the most common cause of phantom P&L degradation.

```bash
# Scan for outlier prices in recent trades
sqlite3 execution/live_trades.db \
  "SELECT symbol, fill_price, ts FROM live_trades WHERE ts > datetime('now', '-3 days') ORDER BY ts DESC LIMIT 50"

# Check for fill quality issues (slippage > 0.5%)
python scripts/tca_report.py --lookback 3d
```

### Step 3 -- Check regime alignment

The LARSA strategy has known weak performance in high-VIX sideways regimes.

```bash
python scripts/run_regime_analysis.py --output-json | python -m json.tool
```

If regime is `SIDEWAYS` or `HIGH_VOLATILITY` and has been for > 2 days:

- Consider reducing `POSITION_SIZE_BASE` by 50% temporarily:

  ```bash
  curl -X POST http://localhost:8781/params/propose \
       -H "Content-Type: application/json" \
       -d '{"params": {"POSITION_SIZE_BASE": 0.01}}'
  ```

### Step 4 -- Check BH physics parameters

A CF_BULL_THRESH that is too low causes over-trading (false signals). Too high causes under-trading.

```bash
# Run signal quality analysis
python research/signal_analytics/signal_quality.py --lookback 5d
```

Healthy signal: entry accuracy > 52%, signal rate 0.5--3 per day.

If signal rate is > 5/day: raise CF_BULL_THRESH by 10%:

```bash
curl -X POST http://localhost:8781/params/propose \
     -H "Content-Type: application/json" \
     -d '{"params": {"CF_BULL_THRESH": 0.0011}}'  # from 0.001 (+10%)
```

### Step 5 -- Check execution quality

High slippage or partial fills can flip a positive signal into a loss.

```bash
python execution/tca/tca_analysis.py --lookback 5d --format table
```

Acceptable: mean slippage < 0.05%. Alert threshold: > 0.15%.

### Step 6 -- Stress-test current parameters

Run the current parameters against the last 20 trading days of actual data:

```bash
python scripts/stress_test.py --params-from-coord --lookback 20d
```

If Sharpe < 0.5 in stress test, rollback parameters and investigate offline.

---

## 4. Emergency Procedures

### 4.1 Flatten All Positions Immediately

**Use when:** runaway loss, system malfunction, or pre-planned risk event (e.g., Fed announcement).

**Option A -- Via Alpaca directly (fastest, most reliable):**

```bash
# Set API keys (must be in shell or .env)
source .env

# Close all positions via Alpaca REST API
curl -X DELETE \
     -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     "https://paper-api.alpaca.markets/v2/positions"
```

Confirm positions closed:

```bash
curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     "https://paper-api.alpaca.markets/v2/positions" | python -m json.tool
```

Expected response: `[]` (empty array).

**Option B -- Via coordination layer (graceful, logs the event):**

```bash
curl -X POST http://localhost:8781/admin/flatten_all \
     -H "Content-Type: application/json" \
     -d '{"reason": "manual_flatten", "operator": "your_name"}'
```

**Option C -- Kill the trader to stop new entries, then flatten manually:**

```bash
# 1. Stop live trader (prevents new entries)
kill $(cat .pids/live_trader.pid) 2>/dev/null || pkill -f "run_api.py" || true

# 2. Use Alpaca API to close positions (Option A above)
```

**After flattening:**

1. Verify zero positions on Alpaca dashboard or via API
2. Record the incident in `JOURNAL.md`
3. Do not restart trader until root cause is identified

---

### 4.2 Kill Switch -- Full System Halt

**Use when:** complete system failure, suspected compromise, or exchange/broker outage.

```bash
# Flatten positions first (see 4.1)
# Then kill all services
./scripts/stop_all_services.sh

# Verify nothing is running on our ports
for port in 8780 8781 8791 8792 9090 9091; do
    lsof -ti :$port | xargs kill -9 2>/dev/null || true
done

# Clean up PID files
rm -f .pids/*.pid

echo "All SRFM Lab services stopped."
```

---

## 5. Backup and Recovery

### 5.1 SQLite DB Backup

The live trades database is the most critical data asset. Back it up before any deployment or risky operation.

**Manual backup:**

```bash
DB_PATH="execution/live_trades.db"
BACKUP_DIR="backups/db"
mkdir -p "$BACKUP_DIR"
BACKUP_FILE="${BACKUP_DIR}/live_trades_$(date +%Y%m%d_%H%M%S).db"

# Use SQLite's online backup (safe with WAL mode, no shutdown required)
sqlite3 "$DB_PATH" ".backup '${BACKUP_FILE}'"

echo "Backup written to: $BACKUP_FILE"
ls -lh "$BACKUP_FILE"
```

**Automated daily backup (add to crontab):**

```
# Backup at 5:00 PM ET (after market close)
0 17 * * 1-5 cd /path/to/srfm-lab && ./scripts/backup_db.sh >> logs/backup.log 2>&1
```

**Verify backup integrity:**

```bash
sqlite3 "$BACKUP_FILE" "PRAGMA integrity_check"
sqlite3 "$BACKUP_FILE" "SELECT COUNT(*) FROM live_trades"
```

**Backup retention:** Keep daily backups for 30 days, weekly for 6 months.

```bash
# Prune backups older than 30 days
find backups/db/ -name "*.db" -mtime +30 -delete
```

---

### 5.2 Parameter Snapshot Restore

Parameter snapshots are stored in two places:

1. **In-memory in the coordination layer** (ephemeral -- lost on restart)
2. **On disk in `config/param_snapshots/`** (persistent)

**Save current parameters to disk:**

```bash
mkdir -p config/param_snapshots
SNAP_FILE="config/param_snapshots/params_$(date +%Y%m%d_%H%M%S).json"
curl -s http://localhost:8781/params/current > "$SNAP_FILE"
echo "Snapshot saved: $SNAP_FILE"
```

**Restore parameters from a snapshot file:**

```bash
SNAP_FILE="config/param_snapshots/params_20240115_091523.json"

# Extract the params dict from the snapshot
PARAMS=$(python -c "import json,sys; d=json.load(open('$SNAP_FILE')); print(json.dumps({'params': d['params']}))")

curl -X POST http://localhost:8781/params/propose \
     -H "Content-Type: application/json" \
     -d "$PARAMS"
```

**Verify the restore:**

```bash
curl -s http://localhost:8781/params/current | python -m json.tool
diff <(python -m json.tool "$SNAP_FILE") <(curl -s http://localhost:8781/params/current | python -m json.tool)
```

**List available snapshots:**

```bash
ls -lt config/param_snapshots/ | head -20
```

---

### 5.3 Full System Restore Procedure

Use after a catastrophic failure (host crash, disk corruption) that requires rebuilding from scratch.

**Prerequisites:**

- Git repo cloned at target path
- `.env` file with all credentials
- Python 3.11+, Rust toolchain, Go 1.21+, Elixir/OTP installed

**Steps:**

1. **Restore dependencies and build all components:**

   ```bash
   ./scripts/deploy.sh --no-restart --env staging
   ```

2. **Restore the SQLite database from most recent backup:**

   ```bash
   cp backups/db/live_trades_LATEST.db execution/live_trades.db
   sqlite3 execution/live_trades.db "PRAGMA integrity_check"
   ```

3. **Start infrastructure services (no trader yet):**

   ```bash
   ./scripts/start_all_services.sh --no-trader
   ```

4. **Wait for all services to be healthy:**

   ```bash
   python scripts/health_check.py --pretty
   # Repeat until status = "ok"
   ```

5. **Restore the last known good parameter snapshot:**

   ```bash
   SNAP=$(ls -t config/param_snapshots/*.json | head -1)
   echo "Restoring from: $SNAP"
   PARAMS=$(python -c "import json; d=json.load(open('$SNAP')); print(json.dumps({'params': d['params']}))")
   curl -X POST http://localhost:8781/params/propose \
        -H "Content-Type: application/json" \
        -d "$PARAMS"
   ```

6. **Run smoke tests:**

   ```bash
   python scripts/health_check.py --pretty
   ./scripts/run_all_tests.sh --fast
   ```

7. **Start the live trader (only after smoke tests pass):**

   ```bash
   # Re-run start script -- trader will start automatically
   ./scripts/start_all_services.sh
   ```

8. **Verify positions match broker:**

   ```bash
   # Check what Alpaca says vs. what our DB says
   python scripts/reconcile_positions.py
   ```

   Resolve any discrepancies before leaving the system unattended.

---

## 6. Service Reference

| Service | Port | Binary / Entry Point | Config | Log |
|---|---|---|---|---|
| Elixir coordination | 8781 | `mix run` in `coordination/` | `coordination/config/` | `logs/coordination.log` |
| Go market-data | 8780 | `market-data/market-data` | env vars | `logs/market_data.log` |
| C++ signal engine | IPC | `cpp/build/Release/srfm_signal_engine` | CLI flags | `logs/signal_engine.log` |
| Python risk API | 8791 | `python -m execution.risk.risk_api` | `execution/risk/` | `logs/risk_api.log` |
| Go risk aggregator | 8792 | `cmd/risk-aggregator/risk-aggregator` | env vars | `logs/risk_aggregator.log` |
| Observability API | 9091 | `python -m infra.observability.api` | `infra/` | `logs/observability_api.log` |
| Metrics collector | 9090 | `cmd/metrics-collector/metrics-collector` | env vars | `logs/metrics_collector.log` |
| Python live trader | -- | `python run_api.py` | `.env` + coord layer | `logs/live_trader.log` |

**Key health endpoints:**

| Service | Health URL |
|---|---|
| Coordination | `GET http://localhost:8781/health` |
| Market-data | `GET http://localhost:8780/health` |
| Risk API | `GET http://localhost:8791/risk/health` |
| Risk aggregator | `GET http://localhost:8792/health` |
| Observability | `GET http://localhost:9091/health` |

**Useful one-liner to check all at once:**

```bash
for url in \
  "http://localhost:8781/health" \
  "http://localhost:8780/health" \
  "http://localhost:8791/risk/health" \
  "http://localhost:8792/health" \
  "http://localhost:9091/health"; do
  status=$(curl -so /dev/null -w "%{http_code}" --max-time 2 "$url" 2>/dev/null || echo "000")
  echo "$status  $url"
done
```
