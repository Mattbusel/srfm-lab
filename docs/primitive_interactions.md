# Primitive Interaction Reference

Every primitive in the SRFM system, what it reads, what it writes, and which other
primitives it interacts with directly. This is the canonical dependency map.

---

## Primitive Inventory

### Live Trading Primitives

| Primitive | File | Language | Category |
|---|---|---|---|
| `LiveTrader` | `tools/live_trader_alpaca.py` | Python | Core |
| `BHState` | `tools/live_trader_alpaca.py` | Python | Core |
| `GARCHTracker` | `tools/live_trader_alpaca.py` | Python | Core |
| `OUDetector` | `tools/live_trader_alpaca.py` | Python | Core |
| `ATRTracker` | `tools/live_trader_alpaca.py` | Python | Core |
| `BullScale` | `tools/live_trader_alpaca.py` | Python | Core |
| `OptionOverlay` | `tools/live_trader_alpaca.py` | Python | Core |
| `QuatNavPy` | `bridge/quat_nav_bridge.py` | Python | Observability |
| `NavStateWriter` | `bridge/quat_nav_bridge.py` | Python | Observability |
| `LiveParamBridge` | `bridge/live_param_bridge.py` | Python | Config |
| `PerformanceTracker` | `bridge/performance_tracker.py` | Python | Monitoring |
| `TradeMonitor` | `bridge/trade_monitor.py` | Python | Monitoring |

### Execution Primitives

| Primitive | File | Language | Category |
|---|---|---|---|
| `OrderBook` | `execution/orderbook/orderbook.py` | Python | Market Data |
| `AlpacaL2Feed` | `execution/orderbook/alpaca_l2_feed.py` | Python | Market Data |
| `BinanceL2Feed` | `execution/orderbook/binance_l2_feed.py` | Python | Market Data |
| `BookManager` | `execution/orderbook/book_manager.py` | Python | Market Data |
| `FeedMonitor` | `execution/orderbook/feed_monitor.py` | Python | Market Data |
| `SmartRouter` | `execution/routing/smart_router.py` | Python | Routing |
| `OrderManager` | `execution/oms/order_manager.py` | Python | OMS |
| `RiskGuard` | `execution/oms/risk_guard.py` | Python | Risk |
| `CircuitBreaker` | `execution/monitoring/circuit_breaker.py` | Python | Safety |
| `AuditLog` | `execution/audit/audit_log.py` | Python | Audit |
| `TCAAnalyzer` | `execution/tca/tca_analyzer.py` | Python | Analysis |

### BH Physics Primitives

| Primitive | File | Language | Category |
|---|---|---|---|
| `MinkowskiClassifier` | `lib/srfm_core.py` | Python | Physics |
| `BlackHoleDetector` | `lib/srfm_core.py` | Python | Physics |
| `GeodesicAnalyzer` | `lib/srfm_core.py` | Python | Physics |
| `GravitationalLens` | `lib/srfm_core.py` | Python | Physics |
| `MarketRegime` | `lib/regime.py` | Python | Physics |
| `agent_d3qn` | `lib/agents.py` | Python | Signal |
| `agent_ddqn` | `lib/agents.py` | Python | Signal |
| `agent_td3qn` | `lib/agents.py` | Python | Signal |
| `BHState` (C++) | `cpp/signal-engine/src/bh_physics/` | C++ | Physics |
| `QuatNav` (C++) | `cpp/signal-engine/src/quaternion/` | C++ | Nav |
| `BHState` (Rust) | `crates/larsa-core/` | Rust | Physics |

### IAE Primitives

| Primitive | File | Language | Category |
|---|---|---|---|
| `TimeOfDayMiner` | `idea-engine/ingestion/miners/` | Python | Mining |
| `RegimeClusterMiner` | `idea-engine/ingestion/miners/` | Python | Mining |
| `BHPhysicsMiner` | `idea-engine/ingestion/miners/` | Python | Mining |
| `DrawdownMiner` | `idea-engine/ingestion/miners/` | Python | Mining |
| `BootstrapFilter` | `idea-engine/ingestion/statistical_filters/` | Python | Filter |
| `GenomeEngine` | `crates/idea-genome-engine/` | Rust | Evolution |
| `CounterfactualEngine` | `crates/counterfactual-engine/` | Rust | Validation |
| `RegimeOracle` | `idea-engine/regime-oracle/` | Python/Rust | Classification |
| `OnChainEngine` | `idea-engine/onchain/` | Python | Alternative Data |
| `MLSignalModule` | `tools/backtest_wave4.py` | Python | Signal |
| `EventCalendarFilter` | `tools/backtest_wave4.py` | Python | Filter |
| `NetworkSignalTracker` | `tools/backtest_wave4.py` | Python | Signal |
| `ParameterCoordinator` | `coordination/lib/` | Elixir | Coordination |
| `CircuitBreaker` (Elixir) | `coordination/lib/` | Elixir | Safety |

### Statistical Primitives

| Primitive | File | Language | Category |
|---|---|---|---|
| `PortfolioEngine` | `crates/portfolio-engine/` | Rust | Portfolio |
| `RiskEngine` | `crates/risk-engine/` | Rust | Risk |
| `ExecutionOptimizer` | `crates/execution-optimizer/` | Rust | Execution |
| `FractalEngine` | `crates/fractal-engine/` | Rust | Analysis |
| `MonteCarloEngine` | `crates/monte-carlo-engine/` | Rust | Simulation |
| `BHPhysics.jl` | `julia/src/BHPhysics.jl` | Julia | Physics |
| `SystemicRisk.jl` | `julia/src/SystemicRisk.jl` | Julia | Risk |
| `volatility_surface.jl` | `idea-engine/stats-service/julia/` | Julia | Vol |
| `alpha_research.jl` | `idea-engine/stats-service/julia/` | Julia | Alpha |
| `regime_models.R` | `r/R/` | R | Regime |
| `systemic_risk.R` | `r/R/` | R | Risk |

---

## Interaction Matrix

### LiveTrader Interactions

`LiveTrader` is the root of the live trading dependency graph. On every 15-minute bar:

```
LiveTrader._on_15m_bar(bar)
  reads: Alpaca WebSocket (via market-data service :8780)
  reads: config/live_params.json (via LiveParamBridge, hot-reload polling)
  reads: config/signal_overrides.json (hot-reload polling)

  calls: BHState.update(bar)           -- mass accumulation
    uses: MinkowskiClassifier          -- ds^2 classification
    uses: GeodesicAnalyzer             -- 20-bar regression block

  calls: GARCHTracker.update(bar)      -- GARCH(1,1) vol forecast

  calls: OUDetector.update(bar)        -- OU mean-reversion state
    uses: MinkowskiClassifier (prev)   -- proper time weighting

  calls: ATRTracker.update(bar)        -- ATR for stop/size calculation

  calls: BullScale.update(bar, bh_state) -- BTC lead scaler

  calls: agent_d3qn(features)         -- trend + momentum signal
  calls: agent_ddqn(features)         -- alignment composite signal
  calls: agent_td3qn(features)        -- mean-reversion signal
    each uses: GravitationalLens      -- mass-weighted signal boost
    each uses: MarketRegime           -- regime-dependent weights

  calls: QuatNavPy.update(bar, bh_mass, bh_active)  -- nav observability
    output goes to: NavStateWriter -> execution/live_trades.db (nav_state table)

  calls: SmartRouter.route(signal, size)   -- spread-tier order routing
    queries: OrderBook.get_spread()         -- current bid-ask via BookManager
    calls: AlpacaL2Feed OR BinanceL2Feed    -- live order submission

  calls: OrderManager.submit(order)   -- OMS lifecycle tracking
    calls: RiskGuard.pre_trade_check() -- notional, Greeks, VAR validation
    calls: AuditLog.record()           -- immutable event recording

  writes: execution/live_trades.db    -- fills, positions, P&L (WAL mode)
```

### BHState Interactions

`BHState` is the per-instrument runtime wrapper in the live trader:

```
BHState wraps: MinkowskiClassifier, BlackHoleDetector, GeodesicAnalyzer
  input:  close price, volume, timestamp (from bar)
  output: mass, active, proper_time, ds2, hawking_temp

  consumed by: LiveTrader (signal gating)
  consumed by: QuatNavPy (bh_mass, bh_active for Lorentz boost)
  consumed by: GravitationalLens (mu = tanh(mass / MU_SCALE))
  consumed by: BHPhysicsMiner (IAE pattern extraction)
  consumed by: agent_* (gravitational lensing factor)
  consumed by: GARCHTracker (GARCH updates are mass-weighted)
  consumed by: PortfolioEngine (BH active flag drives correlation tier)
  mirrored by: QuatNav (C++) in cpp/signal-engine
  mirrored by: BHState (Rust) in crates/larsa-core
```

### QuatNavPy Interactions

```
QuatNavPy.update(close, volume, timestamp_ns, bh_mass, bh_was_active, bh_active)
  reads from: BHState (mass, active flags) -- same bar, after BH update
  reads from: bar data (close, volume, timestamp)

  computes:
    bar quaternion q_bar from (dt, price, vol, MI) in 4-space
    delta_q = q_bar * q_prev^{-1}
    Q_current = normalize(delta_q * Q_current)
    Lorentz boost when bh_was_active != bh_active
    geodesic_deviation from SLERP extrapolation of q_prev2, q_prev vs q_bar

  output goes to:
    NavStateWriter -> execution/live_trades.db (nav_state table)
    NOT used in: entry/exit logic (observability only in LARSA v17)

  future consumers:
    position sizing (angular_velocity sizing multiplier)
    entry conviction gate (geodesic_deviation filter)
    regime fingerprinting (Q_current clustering)
```

### GARCHTracker Interactions

```
GARCHTracker.update(bar)
  input:  log return from bar
  output: conditional vol h_t, long-run vol target

  consumed by: LiveTrader position sizing (garch_scale = target_vol / forecast_vol)
  consumed by: MLSignalModule (GARCH vol as feature in logistic regression)
  consumed by: MonteCarloEngine (GARCH-filtered path generation)
  consumed by: PortfolioEngine (GARCH vol in covariance shrinkage)
```

### SmartRouter Interactions

```
SmartRouter.route(signal, symbol, size)
  reads: BookManager.get_spread(symbol)
    reads: AlpacaL2Feed (primary) or BinanceL2Feed (fallback)
    fallback managed by: BookManager (30s failover timer)

  routing logic:
    spread <= 50 bps  -> Alpaca order submission
    spread 50-100 bps -> Binance order submission
    spread > 100 bps  -> block trade (defer or split via ExecutionOptimizer)

  calls: CircuitBreaker[alpaca].call(submit_fn)
    if circuit OPEN: -> immediate error, no submission

  result goes to: OrderManager.on_submitted(order_id)
  logged by: AuditLog.record()
  monitored by: TCAAnalyzer (slippage vs arrival price)
```

### IAE Loop Interactions

```
IAE loop (every 4-6 hours):

Step 1: Ingestion
  execution/live_trades.db -> ingestion/pipeline.py
    -> TimeOfDayMiner(trades) -> idea_engine.db
    -> RegimeClusterMiner(trades) -> idea_engine.db
    -> BHPhysicsMiner(trades) -> idea_engine.db  -- reads BHState output
    -> DrawdownMiner(trades) -> idea_engine.db
    -> BootstrapFilter(patterns) -> confirmed_patterns in idea_engine.db

Step 2: Genome Evolution
  confirmed_patterns -> GenomeEngine (crates/idea-genome-engine)
    runs 100 genomes x 100 generations via:
      -> tick-backtest (Rust, parallel rayon) for fitness evaluation
      -> larsa-core (Rust) for BH physics
    output: best_genome.json -> config/genome_result.json

Step 3: Sensitivity Validation
  best_genome.json -> CounterfactualEngine (crates/counterfactual-engine)
    -> LHS sampling around genome
    -> tick-backtest for each sample
    -> SobolAnalyzer: sensitivity indices per parameter
    output: robustness_report.json

Step 4: Parameter Proposal
  best_genome.json -> idea-engine/cmd/api -> POST /params/propose -> ParameterCoordinator
    validates against: param_schema.json
    validates against: RiskGuard delta limits
    if accepted: writes config/live_params.json
      -> LiveParamBridge picks up via file polling (30s interval)
        -> LiveTrader hot-reloads parameters without restart
    if rejected: event published, previous params retained

Step 5: Rollback Monitoring
  PerformanceTracker monitors: execution/live_trades.db (rolling 4h Sharpe)
  if Sharpe < -0.5 post-update:
    -> ParameterCoordinator.rollback()
    -> config/live_params.json restored to prev_params
    -> LiveParamBridge picks up rollback
```

### PortfolioEngine Interactions

```
PortfolioEngine (crates/portfolio-engine)
  inputs:
    returns matrix (21 instruments x 252 bars) from data-pipeline
    BH active flags per instrument (from larsa-core backtest)
    GARCH vol forecasts per instrument

  method selection:
    n_instruments <= 10: standard inverse-vol weighting
    10 < n_instruments <= 30: Ledoit-Wolf shrinkage
    n_instruments > 30: HRP (avoids matrix inversion instability)

  outputs:
    weight vector per instrument
    portfolio-level expected Sharpe and max DD
    correlation-adjusted position limits

  consumed by: LiveTrader position sizing (DELTA_MAX_FRAC enforcement)
  consumed by: RiskGuard (portfolio-level Greeks and VaR limits)
  consumed by: GenomeEngine (portfolio Sharpe as secondary objective)
```

### BookManager Interactions

```
BookManager
  owns: AlpacaL2Feed (primary)
  owns: BinanceL2Feed (fallback)
  owns: FeedMonitor (health polling every 60s)

  state machine:
    ALPACA_PRIMARY: default state
    BINANCE_PRIMARY: activated after 30s without Alpaca update
    switching back: hysteresis, 60s after Alpaca reconnects

  consumed by: SmartRouter (spread check before every order)
  consumed by: LiveTrader (intra-bar spread monitoring)
  consumed by: FeedMonitor -> orderbook_metrics.jsonl (60s spread samples)
  consumed by: market-data/monitoring/metrics.go -> Prometheus
```

---

## Cross-Language Boundaries

The system crosses language boundaries in five places:

| Caller | Callee | Interface | Data |
|---|---|---|---|
| Python (live trader) | Go (market-data) | HTTP REST + WebSocket | OHLCV bars, L2 spread |
| Python (IAE pipeline) | Rust (genome/tick-backtest) | subprocess + JSON files | trade CSV, genome JSON |
| Python (execution) | Zig/C (orderbook) | ctypes .so/.dll call | spread bps, order IDs |
| Python (bridge) | Elixir (coordination) | HTTP REST | param deltas, health |
| Go (IAE bus) | Python (live_param_bridge) | file polling + REST | live_params.json |

There is no direct RPC between Python and Rust at runtime. All Rust computation is
batch (offline or near-offline) invoked via subprocess. The live trading hot path is
entirely Python with ctypes calls for the spread check.

---

## Data Stores

| Store | Location | Written by | Read by |
|---|---|---|---|
| `execution/live_trades.db` | SQLite WAL | LiveTrader, NavStateWriter, AuditLog | IAE miners, PerformanceTracker, TradeMonitor, spacetime API |
| `config/live_params.json` | JSON file | ParameterCoordinator, GenomeEngine | LiveTrader (via LiveParamBridge, 30s poll) |
| `config/signal_overrides.json` | JSON file | Manual / IAE | LiveTrader (hot-reload) |
| `idea_engine.db` | SQLite WAL | IAE miners, filters | GenomeEngine, IAE API, dashboards |
| `market_data.db` | SQLite WAL | market-data service | LiveTrader (bootstrap), spacetime API |
| `orderbook_metrics.jsonl` | JSONL file | FeedMonitor | TCAAnalyzer, research analysis |
| `results/` | CSV + JSON | tick-backtest, MC engine | Research, IAE comparison |

---

## Primitive Flags

| Flag | Meaning |
|---|---|
| **LIVE** | Runs in production on every bar |
| **PRIMITIVE** | Core building block used by multiple systems |
| **OBSERVABILITY** | Writes data, not wired into trading decisions |
| **EXPERIMENTAL** | Built and tested, not deployed in LARSA v17 |
| **IAE** | Part of the autonomous research feedback loop |
| **INFRA** | Infrastructure, not a trading primitive |

| Primitive | Flag |
|---|---|
| `BHState` | LIVE, PRIMITIVE |
| `MinkowskiClassifier` | LIVE, PRIMITIVE |
| `GARCHTracker` | LIVE, PRIMITIVE |
| `OUDetector` | LIVE, PRIMITIVE |
| `QuatNavPy` | LIVE, OBSERVABILITY |
| `NavStateWriter` | LIVE, OBSERVABILITY |
| `LiveParamBridge` | LIVE, INFRA |
| `SmartRouter` | LIVE, PRIMITIVE |
| `BookManager` | LIVE, PRIMITIVE |
| `OrderManager` | LIVE, PRIMITIVE |
| `RiskGuard` | LIVE, PRIMITIVE |
| `CircuitBreaker` | LIVE, PRIMITIVE |
| `AuditLog` | LIVE, INFRA |
| `ParameterCoordinator` | LIVE, IAE |
| `GenomeEngine` | IAE, PRIMITIVE |
| `CounterfactualEngine` | IAE, PRIMITIVE |
| `TimeOfDayMiner` | IAE, PRIMITIVE |
| `BHPhysicsMiner` | IAE, PRIMITIVE |
| `BootstrapFilter` | IAE, PRIMITIVE |
| `MLSignalModule` | EXPERIMENTAL |
| `rl-exit-optimizer` | EXPERIMENTAL |
| `EventCalendarFilter` | EXPERIMENTAL |
| `NetworkSignalTracker` | EXPERIMENTAL |
| `QuatNav` (C++) | PRIMITIVE (validation/future) |
| `tick-backtest` | PRIMITIVE (offline) |
| `MonteCarloEngine` | PRIMITIVE (offline) |
| `PortfolioEngine` | PRIMITIVE (offline/IAE) |
| `FractalEngine` | PRIMITIVE (offline/IAE) |
