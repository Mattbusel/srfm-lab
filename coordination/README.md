# SRFM Coordination Service

Elixir/OTP fault-tolerant coordination layer for the SRFM trading lab.
Keeps IAE microservices running 24/7 via supervision trees, circuit breaking,
health monitoring, and distributed parameter coordination.

HTTP API: `http://localhost:8781`

---

## Supervision Tree

```
SrfmCoordination.Supervisor  (one_for_one, max_restarts: 10/60s)
├── ServiceRegistry           Registry — PID + metadata for all IAE services
├── ServiceSupervisor         DynamicSupervisor — external process launchers
├── HealthMonitor             GenServer — polls /health every 30s, drives restarts
├── CircuitBreakerSupervisor  Supervisor
│   ├── CircuitBreaker(:alpaca)
│   ├── CircuitBreaker(:binance)
│   ├── CircuitBreaker(:coinmetrics)
│   ├── CircuitBreaker(:fear_greed)
│   └── CircuitBreaker(:alternative_me)
├── EventBus                  GenServer — in-process pub/sub, ETS history
├── MetricsCollector          GenServer — Prometheus scraper + aggregator
├── ParameterCoordinator      GenServer — validated parameter fan-out
├── AlertManager              GenServer — dedup + route + snooze alerts
└── HTTPServer                Plug.Cowboy on port 8781
```

---

## Quick Start

```bash
cd coordination
mix deps.get
mix run --no-halt          # dev
MIX_ENV=prod mix run --no-halt
```

Run tests:

```bash
mix test
```

---

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health summary (overall score, per-service status) |
| GET | `/health/services` | Per-service detail with recent check history |
| GET | `/services` | All registered services |
| POST | `/services/:name/restart` | Trigger restart of a managed service |
| GET | `/metrics` | Aggregated metrics snapshot |
| GET | `/events?topic=alert&limit=50` | Recent events (all topics or filtered) |
| POST | `/parameters` | Apply parameter delta `{delta: {...}, author: "..."}` |
| GET | `/parameters` | All current parameters |
| GET | `/circuit-breakers` | State of all circuit breakers |
| POST | `/circuit-breakers/:name/reset` | Manually reset a circuit to CLOSED |
| POST | `/halt` | Emergency halt — sets global halt flag, emits EMERGENCY alert |
| GET | `/halt` | Check halt status |

### Example: Apply Parameter Delta

```bash
curl -X POST http://localhost:8781/parameters \
  -H 'content-type: application/json' \
  -d '{"delta": {"alpha": 0.05, "vol_target": 0.12}, "author": "IAE_hypothesis_42"}'
```

### Example: System Health

```bash
curl http://localhost:8781/health
```

```json
{
  "overall": "healthy",
  "score": 100.0,
  "uptime_seconds": 3600,
  "event_count_24h": 842,
  "service_counts": {"healthy": 5, "degraded": 0, "down": 0, "unknown": 0},
  "services": [...],
  "circuit_breakers": [...],
  "checked_at": "2025-01-01T12:00:00Z"
}
```

---

## Key Behaviors

### Health Monitor

- Polls every 30 seconds (10s in dev)
- 3 consecutive failures → DEGRADED + `:service_degraded` event
- 5 consecutive failures → DOWN + automatic restart via ServiceSupervisor
- Last 1000 checks per service stored in ETS `:srfm_health_history`

### Circuit Breaker

```
CLOSED ──(5 failures/60s)──► OPEN ──(60s cooldown)──► HALF_OPEN
  ▲                                                        │
  └──────────────(probe succeeds)─────────────────────────┘
                   (probe fails) ──► OPEN (reset cooldown)
```

Usage in code:

```elixir
SrfmCoordination.CircuitBreaker.call(:alpaca, fn ->
  HTTPoison.get("https://api.alpaca.markets/v2/account")
end)
# => {:ok, result} | {:error, :circuit_open} | {:error, reason}
```

### Parameter Coordinator

1. Validate delta via RiskGuard (`POST /riskguard/validate`)
2. Write to ETS ParameterStore with version + timestamp
3. Fan-out `POST /parameters/update` to all healthy/degraded services
4. If >20% fail to ACK within 30s → rollback to previous snapshot
5. On success → emit `:parameter_changed` event

### Alert Manager

| Level | Routing |
|-------|---------|
| `:info` | Logger.info only |
| `:warning` | Logger.warning only |
| `:critical` | Logger.error + `alerts.log` + stderr |
| `:emergency` | Logger.error + `alerts.log` + stderr |

Same alert key is deduplicated for 15 minutes. Use `snooze/2` for maintenance windows.

---

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `http_port` | 8781 | Coordination API port |
| `health_poll_interval_ms` | 30_000 | Health check cadence |
| `health_degraded_threshold` | 3 | Consecutive failures before DEGRADED |
| `health_down_threshold` | 5 | Consecutive failures before DOWN + restart |
| `circuit_failure_threshold` | 5 | Failures before circuit opens |
| `circuit_cooldown_ms` | 60_000 | Time OPEN before probe |
| `param_ack_timeout_ms` | 30_000 | ACK window for parameter updates |
| `param_rollback_threshold` | 0.20 | Max failure rate before rollback |
| `riskguard_url` | localhost:8790 | RiskGuard validation endpoint |

---

## Module Reference

| Module | Role |
|--------|------|
| `Application` | OTP entry point, builds supervision tree |
| `ServiceRegistry` | ETS-backed registry for service metadata |
| `ServiceSupervisor` | DynamicSupervisor + ServiceWorker (Port owner) |
| `HealthMonitor` | Async HTTP health polling, status state machine |
| `CircuitBreaker` | Per-API circuit state machine |
| `EventBus` | Pub/sub, ETS event history, monitor-based cleanup |
| `MetricsCollector` | Prometheus scraper, ETS storage, periodic flush |
| `ParameterCoordinator` | Validated fan-out with rollback |
| `AlertManager` | Dedup, routing, snooze, rule evaluation |
| `HTTP.Router` | Plug router for REST API |
| `HTTP.HealthController` | Health endpoint response builders |
