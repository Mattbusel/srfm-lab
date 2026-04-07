# Coordination Layer (Elixir/OTP)

A fault-tolerant supervision and parameter coordination system built on Erlang/OTP via
Elixir. It manages process lifecycles for all IAE microservices, validates parameter
updates before they reach the live trader, and provides a circuit breaker layer between
the IAE feedback loop and external broker APIs.

Status: **live in production**, running at `:8781`.

---

## Purpose

The IAE loop produces parameter updates every 4-6 hours. Before any update reaches
the live trader, it must pass three gates:

1. **Schema validation**: parameters are within their allowed ranges
2. **Risk validation**: the proposed change does not exceed safe delta thresholds
3. **Rollback guard**: if post-update performance degrades, the previous parameters
   are restored automatically

Additionally, the coordination layer supervises all IAE microservices (API, event bus,
scheduler, webhook). If any service crashes, OTP restarts it within seconds. The live
trader never sees a service outage directly.

This layer is written in Elixir because the fault-tolerance guarantees of OTP (process
isolation, supervisor trees, let-it-crash philosophy) are a natural fit for a system
that must stay running continuously and recover from partial failures automatically.

---

## Supervision Tree

```
Application (one_for_one)
  ServiceRegistry       -- ETS-backed service registry
  HealthMonitor         -- polls /health on all services every 30s
  EventBus              -- in-process pub/sub with ETS history
  CircuitBreakerSup
    CircuitBreaker[alpaca]     -- Alpaca API circuit breaker
    CircuitBreaker[binance]    -- Binance API circuit breaker
    CircuitBreaker[polygon]    -- Polygon.io data circuit breaker
  ParameterCoordinator  -- validates and fans out parameter updates
  HttpServer            -- Plug REST API at :8781
```

The supervisor uses `one_for_one` strategy with a maximum of 10 restarts per 60
seconds. If any child exceeds this rate, the supervisor itself restarts, which
triggers a cascade through the monitoring alerts.

---

## ServiceRegistry

An ETS (Erlang Term Storage) table that maps service names to their PIDs and metadata:

```elixir
# lib/service_registry.ex
defmodule ServiceRegistry do
  def register(name, pid, metadata \\ %{}) do
    :ets.insert(:service_registry, {name, pid, :os.system_time(:millisecond), metadata})
  end

  def lookup(name), do: :ets.lookup(:service_registry, name)

  def all_services(), do: :ets.tab2list(:service_registry)
end
```

All IAE services register on startup. The `HealthMonitor` reads from this registry to
know which endpoints to poll.

---

## HealthMonitor

Polls the `/health` endpoint of every registered service every 30 seconds:

```elixir
# lib/health_monitor.ex
def handle_info(:check, state) do
  Enum.each(ServiceRegistry.all_services(), fn {name, _pid, _ts, %{health_url: url}} ->
    case HTTPoison.get(url, [], timeout: 5_000) do
      {:ok, %{status_code: 200}} ->
        EventBus.publish(:service_healthy, %{name: name})
      _ ->
        EventBus.publish(:service_unhealthy, %{name: name})
        maybe_restart(name)
    end
  end)
  {:noreply, state, @check_interval_ms}
end
```

Services that fail 3 consecutive health checks are restarted via `System.cmd/3`. The
restart is logged to the audit trail and published as a `service_restarted` event.

---

## CircuitBreaker

Per-API circuit breaker with three states:

```
CLOSED
  Requests pass through normally.
  Failure counter increments on timeout / 5xx response.
  After 5 failures within 60 seconds: -> OPEN

OPEN
  All requests immediately return {:error, :circuit_open}.
  After 60 seconds cooldown: -> HALF_OPEN

HALF_OPEN
  One probe request allowed.
  If probe succeeds: -> CLOSED (failure counter reset)
  If probe fails: -> OPEN (cooldown resets)
```

Implementation:

```elixir
# lib/circuit_breaker.ex
def call(breaker_name, fun) do
  case get_state(breaker_name) do
    :closed   -> execute_and_track(breaker_name, fun)
    :open     -> {:error, :circuit_open}
    :half_open -> probe(breaker_name, fun)
  end
end
```

The circuit breakers protect against cascade failures: if Alpaca's API is degraded,
the Alpaca circuit opens, and order submissions automatically fail fast rather than
blocking the live trader's event loop on timeouts.

---

## ParameterCoordinator

The most critical component. Receives proposed parameter updates from the IAE, validates
them, and fans them out to all consumers:

```elixir
# lib/parameter_coordinator.ex
def handle_cast({:propose_update, new_params, source}, state) do
  with :ok <- validate_schema(new_params),
       :ok <- RiskGuard.validate_delta(state.current_params, new_params),
       :ok <- check_rollback_window(state) do

    apply_params(new_params)
    EventBus.publish(:params_updated, %{params: new_params, source: source})
    {:noreply, %{state | current_params: new_params, prev_params: state.current_params}}

  else
    {:error, reason} ->
      Logger.warning("Parameter update rejected: #{reason}")
      EventBus.publish(:params_rejected, %{reason: reason, source: source})
      {:noreply, state}
  end
end
```

**Schema validation**: every parameter must be within its allowed range (defined in
`config/param_schema.json`). CF values must be positive. MIN_HOLD_BARS must be an
integer in [1, 48]. Blocked hours must be a subset of [0, 23].

**RiskGuard delta check** (`:8790`): the proposed change cannot shift any single
parameter by more than 25% of its current value in one update. Large genome jumps are
broken into incremental steps applied over multiple cycles.

**Rollback logic**: if the live trader's 4-hour rolling Sharpe drops below -0.5 in the
2 hours following a parameter update, the coordinator automatically restores
`prev_params` and publishes a `rollback_triggered` event.

---

## EventBus

In-process pub/sub with persistent event history:

```elixir
# lib/event_bus.ex
def publish(topic, payload) do
  event = %{topic: topic, payload: payload, ts: :os.system_time(:nanosecond)}
  :ets.insert(:event_history, {topic, event})
  Registry.dispatch(:event_bus_registry, topic, fn entries ->
    Enum.each(entries, fn {pid, _} -> send(pid, {:event, event}) end)
  end)
end

def subscribe(topic) do
  Registry.register(:event_bus_registry, topic, nil)
end
```

The ETS history retains the last 1000 events per topic. Subscribers can replay missed
events on reconnection, which is important for the webhook listener and the parameter
bridge.

Published topics:
- `params_updated` -- successful parameter update
- `params_rejected` -- validation failure
- `rollback_triggered` -- automatic rollback
- `service_healthy` / `service_unhealthy` -- health monitor output
- `service_restarted` -- supervisor restart
- `circuit_open` / `circuit_closed` -- circuit breaker transitions
- `pattern_confirmed` -- IAE confirmed a new pattern
- `backtest_complete` -- IAE backtest job finished

---

## REST API

All endpoints at `:8781`.

| Method | Path | Description |
|---|---|---|
| POST | `/params/propose` | Submit parameter update for validation |
| GET | `/params/current` | Current live parameters |
| GET | `/params/history` | Last 10 parameter updates with source |
| POST | `/params/rollback` | Manual rollback to previous parameters |
| GET | `/services` | All registered services with health status |
| POST | `/services/:name/restart` | Manual service restart |
| GET | `/circuit/:name` | Circuit breaker state for named API |
| POST | `/circuit/:name/reset` | Manually reset circuit to CLOSED |
| GET | `/events?topic=params_updated&n=50` | Recent events by topic |
| GET | `/health` | Coordination layer health |

---

## Integration Points

| System | Direction | Method |
|---|---|---|
| `idea-engine/cmd/bus/main.go` | outbound | POST `/params/propose` when genome evolution completes |
| `bridge/live_param_bridge.py` | inbound | reads `/params/current`, polls for updates |
| `scripts/supervisor.py` | inbound | registers at `/services` on startup |
| `cmd/alerter/` | inbound | subscribes to `service_unhealthy` events |
| `infra/observability/` | inbound | scrapes circuit breaker and event counts |
| `execution/routing/` | outbound | `CircuitBreaker[alpaca]` gates order submission |

---

## Running

```bash
cd coordination
mix deps.get
mix run --no-halt

# Or in release mode
MIX_ENV=prod mix release
_build/prod/rel/coordination/bin/coordination start
```

Health check:
```bash
curl http://localhost:8781/health
# {"status":"ok","services":{"api":"healthy","bus":"healthy","scheduler":"healthy"}}
```

---

## Fault Tolerance Guarantees

1. **Process isolation**: each service runs in a separate OS process. A crash in the
   Go IAE API does not affect the Elixir coordinator.

2. **Automatic restart**: the `HealthMonitor` restarts dead services within 30-90
   seconds. The live trader continues trading during the restart window.

3. **Circuit protection**: if Alpaca is degraded, the circuit opens within 5 failed
   requests. Subsequent order attempts fail fast (< 1ms) rather than timing out.

4. **Parameter safety**: no parameter update bypasses schema validation and the
   RiskGuard delta check. The genome engine cannot push a catastrophic parameter
   change to the live trader in a single step.

5. **Rollback guarantee**: if a parameter update demonstrably hurts 2-hour performance,
   it is reversed automatically without human intervention.
