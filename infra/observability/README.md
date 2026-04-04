# LARSA Observability Stack

Full metrics pipeline for the LARSA v16 live trader: metrics emission → InfluxDB → Grafana.

## Architecture

```
live_trader_alpaca.py
        │
        │  collector.update(state_dict)  [every rebalance cycle]
        ▼
metrics_server.py  ──────────────────────────────────────────────────────┐
  • Prometheus HTTP endpoint  :9090/metrics  (prometheus_client)          │
  • Background InfluxDB writer every 15 s    (influxdb-client)            │
  • Reads TradeLogger for rolling perf stats                               │
        │                                                                   │
        │  [HTTP scrape every 15s]                                          │
        ▼                                          [writes every 15s]      │
  Prometheus :9091  ◄────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
                                                    InfluxDB :8086
                                                    bucket: larsa_metrics
                                                           │
                                                           ▼
                                                    Grafana :3000
                                                    (3 dashboards auto-provisioned)
```

## Quick Start

### 1. Start the Docker stack

```bash
cd infra/observability
docker-compose up -d
```

Wait ~30 s for InfluxDB to initialise, then Grafana will be ready.

### 2. Install Python dependencies

```bash
pip install prometheus_client influxdb-client pandas numpy
```

### 3. Start the metrics server

Run standalone (generates synthetic demo metrics — useful for testing the stack without the live trader):

```bash
python infra/observability/metrics_server.py
```

Or, integrate into the live trader (see **Integration** below).

### 4. Access Grafana

- URL: http://localhost:3000
- Username: `admin`
- Password: `larsa-admin-pass`

Three dashboards are auto-provisioned in the **LARSA** folder:

| Dashboard | UID | Description |
|---|---|---|
| LARSA Live Trading | `larsa-live-trading` | Equity, positions, delta scores, PID, trade log |
| LARSA Regime Monitor | `larsa-regime-monitor` | BH mass, mean reversion, regime classification |
| LARSA Performance Analytics | `larsa-performance` | Equity curve, win rates, P&L distributions |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LARSA_PROM_PORT` | `9090` | Prometheus HTTP server port |
| `LARSA_INFLUX_URL` | `http://localhost:8086` | InfluxDB URL |
| `LARSA_INFLUX_TOKEN` | `larsa-super-secret-token` | InfluxDB API token |
| `LARSA_INFLUX_ORG` | `srfm` | InfluxDB organisation |
| `LARSA_INFLUX_BUCKET` | `larsa_metrics` | InfluxDB bucket |
| `LARSA_INFLUX_INTERVAL` | `15` | Seconds between InfluxDB writes |

## Integration with Live Trader

Add to `tools/live_trader_alpaca.py`:

```python
# At the top of the file, after imports:
from infra.observability.metrics_server import MetricsCollector, start_metrics_server
from infra.observability.trade_logger import TradeLogger

# After engine/orders are initialised (before the asyncio.gather):
_metrics: MetricsCollector = None
_trade_log: TradeLogger = None

async def main():
    global _metrics, _trade_log

    # Start observability
    _metrics   = await start_metrics_server()
    _trade_log = _metrics._trade_logger

    # ... rest of main() unchanged ...
```

Then in `_trigger_rebalance`, after `_orders.rebalance(...)`:

```python
# Build state dict from engine state
state = {
    "equity":    equity,
    "drawdown":  (peak_equity - equity) / (peak_equity + 1e-9),
    "position_frac":  {s: _engine.last_frac[s] for s in INSTRUMENTS},
    "delta_score":    delta_score,   # from compute_targets
    "bh_mass": {
        s: {
            "daily":  _engine.d_bh[s].mass,
            "hourly": _engine.h_bh[s].mass,
            "m15":    _engine.m15_bh[s].mass,
        }
        for s in INSTRUMENTS
    },
    "bh_active": {
        s: {
            "daily":  _engine.d_bh[s].active,
            "hourly": _engine.h_bh[s].active,
            "m15":    _engine.m15_bh[s].active,
        }
        for s in INSTRUMENTS
    },
    "tf_score": {
        s: (4 if _engine.d_bh[s].active else 0)
         + (2 if _engine.h_bh[s].active else 0)
         + (1 if _engine.m15_bh[s].active else 0)
        for s in INSTRUMENTS
    },
    "atr": {
        s: _engine.m1_atr[s].atr or _engine.h_atr[s].atr or 0.0
        for s in INSTRUMENTS
    },
    "pid_stale_threshold": STALE_15M_MOVE,
    "pid_max_frac": DELTA_MAX_FRAC,
}
_metrics.update(state)
```

For trade logging, call in `OrderManager.rebalance()` after each fill:

```python
_metrics.record_trade(
    symbol=sym, side="buy" if delta_dollars > 0 else "sell",
    qty=qty, price=price, pnl=0.0,   # pnl known on close
    entry_price=_engine.entry_price.get(sym),
    bars_held=_engine.bars_held.get(sym, 0),
    equity_after=equity,
)
```

## Files

```
infra/observability/
├── metrics_server.py              # Prometheus HTTP + InfluxDB writer
├── trade_logger.py                # SQLite trade log + rolling analytics
├── docker-compose.yml             # InfluxDB + Grafana + Prometheus
├── prometheus/
│   └── prometheus.yml             # Scrape config
├── influxdb/
│   └── init/
│       └── setup.sh               # Bucket creation + token setup
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── influxdb.yaml      # InfluxDB + Prometheus datasources
│       └── dashboards/
│           └── dashboards.yaml    # Dashboard file provider config
└── dashboards/
    ├── live_trading.json          # Main trading dashboard (7 rows)
    ├── regime_monitor.json        # BH mass + mean reversion
    └── performance.json           # Equity curve + distributions
```

## InfluxDB Measurements

| Measurement | Tags | Fields |
|---|---|---|
| `larsa_account` | — | `equity`, `win_rate`, `rolling_sharpe`, `rolling_pnl`, `drawdown`, `pid_stale_threshold`, `pid_max_frac`, `trade_count` |
| `larsa_symbol` | `symbol` | `position_frac`, `position_pnl`, `delta_score`, `tf_score`, `atr`, `garch_vol`, `ou_zscore`, `bh_mass_daily`, `bh_mass_hourly`, `bh_mass_15m`, `bh_active_daily`, `bh_active_hourly`, `bh_active_15m` |
| `larsa_trade` | `symbol`, `side` | `qty`, `price`, `entry_price`, `pnl`, `bars_held`, `trade_duration_s` |

## Resetting the Stack

```bash
docker-compose down -v   # removes all volumes — deletes all data
docker-compose up -d
```

## Troubleshooting

**Grafana shows "No data"**: Confirm the metrics_server is running and reachable at `localhost:9090/metrics`. Check `LARSA_INFLUX_URL` points to `localhost:8086` (not `influxdb:8086`) when running outside Docker.

**InfluxDB token errors**: The admin token is `larsa-super-secret-token`. Override via `LARSA_INFLUX_TOKEN` env var.

**Prometheus can't reach metrics_server**: The docker-compose uses `host.docker.internal` to reach the host. On Linux, ensure `extra_hosts: host.docker.internal:host-gateway` is supported (Docker ≥ 20.10).
