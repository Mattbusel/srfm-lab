# SRFM Trading Lab

A full-stack quantitative trading research platform built on **Special Relativistic Financial Mechanics (SRFM)** -- from raw tick data to live paper trading and autonomous idea discovery, across 9 languages and **1,039,342 lines of code**.

> Mad scientist workshop. Everything automated, everything measurable, rapid iteration at scale.

---

## Navigation

| Section | Description |
|---|---|
| [What is This?](#what-is-this) | Core innovation and philosophy |
| [Architecture](#architecture) | System diagram and data flow |
| [Quick Start](#quick-start) | Get running in 5 minutes |
| [Deep Documentation](#deep-documentation) | Every subsystem has a dedicated doc |
| [Tools and Primitives](#tools-and-primitives) | Every executable tool flagged |
| [Stack](#stack) | All 9 languages, LOC counts, and roles |
| [File Structure](#file-structure) | Complete directory map |
| [BH Physics Reference](#bh-physics-reference) | Signal math |
| [Key Parameters](#key-parameters) | All tunable constants |
| [IAE Research Output](#iae-live-research-output) | Live findings |
| [Service Endpoints](#service-endpoints) | All running ports |

---

## Deep Documentation

Pick a subsystem to deep dive into. Every doc covers architecture, key primitives, code examples, and integration points.

### Core Physics and Signal

| Doc | What it covers |
|---|---|
| [BH Physics Engine](docs/bh_physics.md) | Minkowski metric, mass accumulation, Hawking temperature, delta scoring, worked example |
| [Quaternion Navigation](docs/quaternion_nav.md) | 4-space bar representation, rotation tracking, geodesic deviation, angular velocity, Lorentz boosts |
| [C++ Signal Engine](docs/signal_engine_cpp.md) | SignalOutput struct, InstrumentState, SIMD indicators, ring buffer, Kalman filter, tick indicators, ZMQ publisher |

### Strategy and Intelligence

| Doc | What it covers |
|---|---|
| [IAE Architecture](docs/iae_architecture.md) | Genome evolution, hypothesis engine, causal discovery, regime oracle, feedback loop |
| [Genome Evolution](docs/genome_evolution.md) | NSGA-II Rust crate, crossover/mutation/selection strategies, constraint handling, lineage tracking |
| [RL Exit Optimizer](docs/rl_exit_optimizer.md) | Q-table policy (3125 states), Double DQN trainer, PER experience replay, reward shaping |
| [Online Learning](docs/online_learning.md) | FTRL-Proximal, Passive-Aggressive II, Hedge algorithm, Adam/AdaGrad/RMSProp, bandit explorer |
| [Optimization](docs/optimization.md) | Bayesian optimizer (GP+EI), NSGA-II hyperparameter search, regime-conditional Optuna, Sobol sensitivity |
| [Wave 4 Backtest](docs/wave4_backtest.md) | EventCalendarFilter, Granger lead signal, ML signal module, 4-variant comparison |

### Execution

| Doc | What it covers |
|---|---|
| [Execution Stack](docs/execution_stack.md) | L2 orderbook, smart router, spread-tier routing, supervisor, Docker deployment |
| [Order Management](docs/order_management.md) | TWAP/VWAP/Iceberg engines, algo scheduler, order state machine, TCA benchmarks |
| [Broker Adapters](docs/broker_adapters.md) | Alpaca/Binance/Paper adapters, circuit breaker integration, failover chain, adding new brokers |
| [Coordination Layer](docs/coordination_layer.md) | Elixir/OTP supervision, circuit breakers, parameter validation, rollback, event bus |

### Research and Validation

| Doc | What it covers |
|---|---|
| [Research Validation](docs/research_validation.md) | CPCV purged K-fold, Deflated Sharpe Ratio, causal inference, market efficiency tests |
| [Market Microstructure](docs/market_microstructure.md) | VPIN, OFI, Kyle's Lambda, L3 orderbook, Zig order flow, adversarial orderbook testing |
| [Statistical Tooling](docs/statistical_tooling.md) | All Julia and R modules -- copulas, SVI, Kalman, HJB PDE, SABR, HMM, WFA |
| [Monte Carlo Engine](docs/monte_carlo.md) | GBM, Merton jump-diffusion, Heston, Longstaff-Schwartz American pricing, variance reduction |

### Infrastructure

| Doc | What it covers |
|---|---|
| [Market Data Service](docs/market_data_service.md) | Dual-feed L2 aggregation, 15m bar assembly, WebSocket hub, failover, Prometheus metrics |
| [Native Layer](docs/native_layer.md) | Zig ITCH 5.0 decoder, lock-free L2 book, AVX2 L3 book, SIMD matrix, bar compression, tick processor |
| [Rust Crates Reference](docs/rust_crates.md) | All 27 Rust crates: genome, MC, portfolio, risk, execution, fractal, FIX, online-learning, tick-backtest |
| [Stack Overview](docs/stack_overview.md) | Every language, what it does, how to run it, integration diagram |
| [Primitive Interactions](docs/primitive_interactions.md) | Full dependency map: every primitive, what it reads, writes, and calls |

### Guides

| Doc | What it covers |
|---|---|
| [Quick Start](docs/guides/quick_start.md) | Prerequisites, first backtest, first live trade |
| [Running Backtests](docs/guides/running_backtests.md) | All backtest modes, parameters, output interpretation |
| [Live Trading](docs/guides/live_trading.md) | Paper and live setup, Alpaca keys, risk limits |
| [Strategy Builder](docs/guides/strategy_builder.md) | Adding signals, instruments, and custom strategies |
| [Interpreting Results](docs/guides/interpreting_results.md) | Sharpe, DSR, MC percentiles, IAE pattern scores |

---

## What is This?

The core innovation is the **Black Hole (BH) Physics Strategy** -- a signal model derived from special-relativistic mechanics applied to price data. Price bars are classified as *timelike* or *spacelike* using a Minkowski spacetime metric (`ds^2 = c^2*dt^2 - dx^2`). Mass accumulates on ordered (causal) bars, and a gravitational well forms when mass crosses the **BH_FORM=1.92** threshold -- the black hole formation event that gates entries.

On top of this sits the **Idea Automation Engine (IAE)** -- an autonomous research system that runs genetic genome evolution (NSGA-II), causal discovery, regime classification, walk-forward validation, and academic paper mining continuously, feeding confirmed patterns back into live strategy parameters.

The system runs continuously in production on Alpaca paper trading, evolving its own parameters every 4-6 hours.

-> **[Full BH Physics deep dive](docs/bh_physics.md)**
-> **[Full IAE architecture deep dive](docs/iae_architecture.md)**

---

## Architecture

```
+------------------------------------------------------------------------------+
|                          SRFM Trading Lab                                    |
|                                                                              |
|  +---------------------+    +------------------------------------------+    |
|  |   Live Trader        |    |         Idea Automation Engine (IAE)     |    |
|  |  live_trader_        |    |                                          |    |
|  |  alpaca.py          |<---+  Genome Evolution (NSGA-II Rust)         |    |
|  |                     |    |  Hypothesis Generator (Bayesian)         |    |
|  |  BH Physics Engine  |    |  Regime Oracle (6 modes)                 |    |
|  |  GARCH vol forecast |    |  Causal Discovery (Granger + PC)         |    |
|  |  OU mean reversion  |    |  Walk-Forward (CPCV + DSR)               |    |
|  |  Mayer dampener     |    |  Signal Library (105+ signals, IC decay) |    |
|  |  BTC lead signal    |    |  Academic Miner (arXiv + SSRN)           |    |
|  |  Dynamic CORR       |    |                                          |    |
|  |  RL Exit Policy     |    |  +-------------------------------------+ |    |
|  |  Regime Ensemble    |    |  |  Event Bus (:8768)  Go API (:8767) | |    |
|  +------+-------------+    |  |  Scheduler (:8769)  Webhook (:8770)| |    |
|         |                   |  +-------------------------------------+ |    |
|         | fills             |  React Dashboard (:5175)                 |    |
|         v                   +------------------------------------------+    |
|  +-------------+                                                             |
|  | SQLite       |   +----------------------------------------------------+  |
|  | trade log   |   |              Execution Stack                       |  |
|  | (WAL mode)  |   |  L2 Orderbook (Alpaca WS + Binance fallback)       |  |
|  +------+------+   |  BookManager (30s failover)  FeedMonitor           |  |
|         |           |  SmartRouter (spread-tier: <=50/50-100/>100 bps)  |  |
|         |           |  TWAP/VWAP/Iceberg algos  AlgoScheduler           |  |
|         |           |  Broker Adapters (Alpaca/Binance/Paper)            |  |
|         |           |  CircuitBreaker + AdapterManager failover          |  |
|         |           +----------------------------------------------------+  |
|         |                                                                    |
|         v                                                                    |
|  +--------------------+   +--------------------------------------------+   |
|  |  live_monitor/      |   |     crypto_backtest_mc.py + wave4          |   |
|  |  CLI + dashboard   |   |  3-TF BH + GARCH + OU + MC (10K paths)    |   |
|  +--------------------+   |  EventCalendar + Granger + ML signal       |   |
|                            +--------------------------------------------+   |
|                                                                              |
|  +------------------------------------------------------------------+       |
|  |  Coordination Layer (Elixir/OTP :8781)                           |       |
|  |  ParameterCoordinator  CircuitBreaker  HealthMonitor  EventBus  |       |
|  |  RiskGuard delta check  Rollback guard  Schema validation        |       |
|  +------------------------------------------------------------------+       |
|                                                                              |
|  Julia (~123K LOC)   R (~60K LOC)   Rust (~141K LOC)   C/C++ (~19K LOC)  |
|  stats-service: copulas, HJB PDE, SVI/SABR, SARIMA, Kalman, AMM, CoVaR  |
+------------------------------------------------------------------------------+
```

**Data flow:** Alpaca streams -> BH Engine -> position sizing -> SmartRouter -> L2 spread check -> orders -> SQLite log -> IAE ingestion -> genome evolution -> parameter feedback -> live strategy.

-> **[Execution stack deep dive](docs/execution_stack.md)**

---

## Quick Start

### Prerequisites

```bash
pip install alpaca-py pandas numpy scipy statsmodels matplotlib
# Rust (genome engine, Monte Carlo, 27 crates)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Go 1.22+ (IAE microservices, market data)
# Node 18+ (React dashboards)
# Julia 1.9+ (statistical tooling)
# R 4.2+ (HMM, WFA, regime models)
# Zig 0.12+ (native layer: ITCH decoder, lock-free book)
```

### Backtest

```bash
# Full crypto BH backtest + 10,000-path Monte Carlo
python tools/crypto_backtest_mc.py

# With Wave 4 enhancements (EventCalendar + Granger lead + ML signal)
python tools/backtest_wave4.py

# Custom parameters
python tools/crypto_backtest_mc.py \
  --start-date 2022-01-01 --end-date 2025-01-01 \
  --mc-paths 5000 --symbols BTC,ETH,SOL \
  --bh-form 1.92 --corr 0.25 --garch-target-vol 0.90 \
  --output-dir tools/backtest_output --verbose
```

### Live Paper Trading

```bash
# All services at once
bash scripts/start_all.sh start

# Or individual trader
python tools/live_trader_alpaca.py --paper --log-level INFO

# Dry run (log orders but don't submit)
python tools/live_trader_alpaca.py --dry-run
```

### Full Analysis Pipeline

```bash
# Macro regime + on-chain + alt data + fear/greed + IAE idea miner
python run_full_analysis.py

# IAE ideas only (from backtest data)
python run_iae_analysis.py
```

### IAE Stack

```bash
python -m idea_engine.db.migrate              # Initialize schema
python -m idea_engine.ingestion.pipeline      # Run all miners
cd idea-engine && go run cmd/api/main.go      # API :8767
go run cmd/bus/main.go                        # Event bus :8768
go run cmd/scheduler/main.go                  # Scheduler :8769
cd idea-engine/idea-dashboard && npm run dev  # Dashboard :5175
```

---

## Tools and Primitives

All executable tools, engines, and core primitives flagged by language and role.

Full dependency map showing which primitives call which: **[Primitive Interactions](docs/primitive_interactions.md)**

### Primitive Signal Flow

```
Bar arrives from market-data :8780
  |
  v
BHState.update(bar)                    -- Minkowski ds^2, mass accumulation
  |-- MinkowskiClassifier              -- TIMELIKE / SPACELIKE classification
  |-- GeodesicAnalyzer                 -- 20-bar regression quality
  |-- GravitationalLens                -- mu = tanh(mass / scale)
  |
  +-> GARCHTracker.update(bar)         -- conditional vol h_t
  +-> OUDetector.update(bar)           -- mean-reversion theta, mu, sigma
  +-> ATRTracker.update(bar)           -- stop/size distance
  +-> BullScale.update(bar, bh)        -- BTC lead scaler
  +-> HurstExponent.update(bar)        -- H>0.58 trending, H<0.42 mean-reverting
  |
  +-> RegimeEnsemble                   -- 6-detector weighted majority vote
  |   |-- HMMRegime (Baum-Welch EM + Viterbi)
  |   |-- GARCHRegime (vol-based)
  |   |-- HurstRegime (scaling law)
  |   |-- OnlineRegimeSGD (adaptive)
  |   +-- TransitionPredictor (Markov matrix + Laplace smoothing)
  |
  +-> agent_d3qn(features)             -- trend + momentum signal
  +-> agent_ddqn(features)             -- alignment composite
  +-> agent_td3qn(features)            -- mean-reversion contrarian
  |   each weighted by RegimeEnsemble (BULL/BEAR/SIDEWAYS/HIGH_VOL/TRENDING/MEAN_REV)
  |
  +-> RLExitPolicy.should_exit()       -- Q-table lookup (3125 states, ~100ns)
  |   state: [pnl_bps, bars_held, bh_mass, garch_vol, spread_bps] binned 5x5x5x5x5
  |
  +-> QuatNavPy.update(bar, bh)        -- nav observability (read-only)
      NavStateWriter -> live_trades.db (nav_state table)
  |
  v
SmartRouter.route(signal, size)
  BookManager.get_spread()             -- live L2 spread check
    AlpacaL2Feed (primary)
    BinanceL2Feed (fallback, 30s failover)
  CircuitBreaker[alpaca/binance]       -- fast-fail if API degraded
  |-- spread <= 50 bps: market order
  |-- spread 50-100 bps: TWAP over 3 bars
  +-- spread > 100 bps: VWAP or block
  |
  v
BrokerAdapter (Alpaca / Binance / Paper)
  AlpacaAdapter: token bucket 200 req/min, exponential backoff
  BinanceAdapter: HMAC-SHA256 signing, Spot/Futures routing
  AdapterManager: failover chain, health-based routing
  |
  v
OrderManager.submit()
  RiskGuard.pre_trade_check()          -- notional, VaR, Greeks
  ComplianceLogger.log_submission()    -- hash-chain tamper-evident audit
  AuditLog.record()                    -- immutable event store
  |
  v
execution/live_trades.db (WAL)
  |
  v (every 4-6h)
IAE ingestion -> GenomeEngine (Rust NSGA-II)
  tick-backtest (Rust rayon)           -- parallel fitness evaluation
  CounterfactualEngine                 -- Sobol sensitivity validation
  ParameterCoordinator (Elixir)        -- schema + delta + rollback validation
  LiveParamBridge (30s poll)           -- hot-reload to LiveTrader
```

### Python -- Core Trading

| Tool / Primitive | Path | Role | Flag |
|---|---|---|---|
| `LiveTrader` | `tools/live_trader_alpaca.py` | Main live trading class | **LIVE TRADER** |
| `BHState` | `tools/live_trader_alpaca.py` | BH mass accumulation struct | **PRIMITIVE** |
| `QuatNavPy` | `bridge/quat_nav_bridge.py` | Quaternion navigation state machine | **PRIMITIVE** |
| `NavStateWriter` | `bridge/quat_nav_bridge.py` | Writes nav signals to live_trades.db | **PRIMITIVE** |
| `GARCHTracker` | `tools/live_trader_alpaca.py` | GARCH(1,1) vol forecaster | **PRIMITIVE** |
| `OUDetector` | `tools/live_trader_alpaca.py` | OU mean-reversion detector | **PRIMITIVE** |
| `ATRTracker` | `tools/live_trader_alpaca.py` | ATR position sizing | **PRIMITIVE** |
| `BullScale` | `tools/live_trader_alpaca.py` | BTC lead signal scaler | **PRIMITIVE** |
| `RLExitPolicy` | `lib/rl_exit_policy.py` | Q-table exit policy (3125 states) | **PRIMITIVE** |
| `RegimeEnsemble` | `lib/regime.py` | 6-detector weighted majority vote | **PRIMITIVE** |
| `HurstExponent` | `lib/srfm_core.py` | R/S analysis trending vs mean-reverting | **PRIMITIVE** |
| `SignalCombiner` | `lib/signal_combiner.py` | IC-weighted, Rank, Ensemble, conflict detection | **PRIMITIVE** |
| `PortfolioConstructor` | `lib/portfolio_constructor.py` | RiskBudget, TurnoverConstraint, Sector limits | **PRIMITIVE** |
| `PnLCalculator` | `lib/pnl_calculator.py` | FIFO cost basis, daily aggregation, attribution | **PRIMITIVE** |
| `AlphaDecayTracker` | `lib/alpha_decay_tracker.py` | Exponential IC decay, retire candidates | **PRIMITIVE** |
| `ExecutionState` | `lib/execution_state.py` | Thread-safe state with SQLite WAL, crash recovery | **PRIMITIVE** |
| `MarketCalendar` | `lib/market_calendar.py` | NYSE hours, half-days, FOMC/CPI/NFP/OpEx | **PRIMITIVE** |
| `crypto_backtest_mc.py` | `tools/` | BH backtest + 10K-path Monte Carlo | **BACKTEST ENGINE** |
| `backtest_wave4.py` | `tools/` | Wave 4: EventCalendar + Granger + ML signal | **BACKTEST ENGINE** |
| `EventCalendarFilter` | `tools/backtest_wave4.py` | FOMC/unlock event filter (0.5x +-2h) | **PRIMITIVE** |
| `MLSignalModule` | `tools/backtest_wave4.py` | Logistic SGD signal (5-lag returns + GARCH) | **PRIMITIVE** |
| `NetworkSignalTracker` | `tools/backtest_wave4.py` | BTC Granger lead (1.2x when |corr|>0.3) | **PRIMITIVE** |
| `run_full_analysis.py` | `/` | Macro + on-chain + alt data + IAE pipeline | **ANALYSIS RUNNER** |
| `run_iae_analysis.py` | `/` | IAE idea miner (63K trades) | **ANALYSIS RUNNER** |

### Python -- IAE Pipeline

| Tool / Primitive | Path | Role | Flag |
|---|---|---|---|
| `ingestion/pipeline.py` | `idea-engine/` | 4-stage mine -> filter -> store | **PIPELINE** |
| `TimeOfDayMiner` | `idea-engine/ingestion/miners/` | Hour-of-day P&L statistical test | **MINER** |
| `RegimeClusterMiner` | `idea-engine/ingestion/miners/` | Cluster-based regime patterns | **MINER** |
| `BHPhysicsMiner` | `idea-engine/ingestion/miners/` | BH state pattern extraction | **MINER** |
| `DrawdownMiner` | `idea-engine/ingestion/miners/` | Drawdown event patterns | **MINER** |
| `BootstrapFilter` | `idea-engine/ingestion/statistical_filters/` | BH FDR correction | **FILTER** |
| `OnChainEngine` | `idea-engine/onchain/` | BTC on-chain composite signal | **SIGNAL ENGINE** |
| `RegimeClassifier` | `idea-engine/macro-factor/` | 6-class macro regime oracle | **CLASSIFIER** |
| `GenomeAnalyzer` | `idea-engine/analysis/` | Breakthrough events, gradient estimator | **ANALYZER** |
| `IAEPerformanceTracker` | `idea-engine/analysis/` | Cycle-level performance tracking | **MONITOR** |
| `ParameterExplorer` | `idea-engine/analysis/` | Landscape mapping, sensitivity | **ANALYZER** |
| `LiveFeedbackAnalyzer` | `idea-engine/analysis/` | Live trading gradient bridge | **ANALYZER** |

### Python -- Execution Infrastructure

| Tool / Primitive | Path | Role | Flag |
|---|---|---|---|
| `OrderBook` | `execution/orderbook/orderbook.py` | Thread-safe L2 book with VWAP fill | **PRIMITIVE** |
| `AlpacaL2Feed` | `execution/orderbook/alpaca_l2_feed.py` | Alpaca WSS L2 feed | **FEED** |
| `BinanceL2Feed` | `execution/orderbook/binance_l2_feed.py` | Binance @depth10 fallback feed | **FEED** |
| `BookManager` | `execution/orderbook/book_manager.py` | Dual-feed + 30s failover | **INFRASTRUCTURE** |
| `SmartRouter` | `execution/routing/smart_router.py` | Spread-tier order routing | **ROUTER** |
| `AlpacaAdapter` | `execution/broker_adapters/alpaca_adapter.py` | Token bucket rate limiter, retry | **ADAPTER** |
| `BinanceAdapter` | `execution/broker_adapters/binance_adapter.py` | HMAC signing, Spot/Futures routing | **ADAPTER** |
| `PaperAdapter` | `execution/broker_adapters/paper_adapter.py` | FIFO P&L, partial fill simulation | **ADAPTER** |
| `AdapterManager` | `execution/broker_adapters/adapter_manager.py` | Multi-adapter failover chain | **ROUTER** |
| `TWAPEngine` | `execution/order_management/twap_engine.py` | Time-sliced TWAP execution | **ALGO** |
| `VWAPEngine` | `execution/order_management/twap_engine.py` | U-shaped intraday VWAP execution | **ALGO** |
| `IcebergEngine` | `execution/order_management/algo_scheduler.py` | Hidden qty with visible slice | **ALGO** |
| `AlgoScheduler` | `execution/order_management/algo_scheduler.py` | Unified algo priority scheduler | **INFRASTRUCTURE** |
| `OrderBookTracker` | `execution/order_management/order_book_tracker.py` | SQLite WAL tracker, crash recovery | **INFRASTRUCTURE** |
| `ComplianceLogger` | `execution/audit/compliance_logger.py` | Hash-chain tamper-evident audit log | **AUDIT** |
| `FillProcessor` | `execution/oms/fill_processor.py` | Fill normalization and P&L attribution | **OMS** |
| `OrderRouter` | `execution/oms/order_router.py` | Internal order routing state machine | **OMS** |
| `supervisor.py` | `scripts/` | HTTP :8790 process supervisor | **INFRASTRUCTURE** |

### Python -- Config and Risk

| Tool / Primitive | Path | Role | Flag |
|---|---|---|---|
| `ConfigManager` | `config/config_manager.py` | Dot-notation config, file watcher, subscribers | **CONFIG** |
| `InstrumentsManager` | `config/instruments_manager.py` | 32-instrument universe, BH physics fields | **CONFIG** |
| `EventCalendarManager` | `config/event_calendar_manager.py` | FOMC/CPI/NFP/OpEx blackout dates | **CONFIG** |
| `RiskConfig` | `config/risk_config.py` | Risk limits, position compliance, BH-aware sizing | **RISK** |

### Python -- Research and Validation

| Tool / Primitive | Path | Role | Flag |
|---|---|---|---|
| `CausalInference` | `research/validation/causal_inference.py` | Granger, PSM, DiD, 2SLS | **VALIDATION** |
| `OutOfSampleValidator` | `research/validation/out_of_sample_validator.py` | CPCV, DSR, BH FDR | **VALIDATION** |
| `MarketEfficiencyTests` | `research/validation/market_efficiency_tests.py` | VR test, runs test, GPH long memory | **VALIDATION** |
| `PerformancePersistence` | `research/validation/performance_persistence.py` | Contingency table, IR stability | **VALIDATION** |
| `EfficientFrontierLab` | `research/portfolio_lab/efficient_frontier_lab.py` | SLSQP, bootstrap confidence bands | **RESEARCH** |
| `FactorExposureAnalyzer` | `research/portfolio_lab/factor_exposure_analyzer.py` | OLS attribution, 7-factor model | **RESEARCH** |
| `ReturnAttributionLab` | `research/portfolio_lab/return_attribution_lab.py` | BHB Brinson, factor attribution | **RESEARCH** |
| `AgentBasedModel` | `research/simulation/agent_based_model.py` | 5 agent types, OU true-value walk | **SIMULATION** |
| `MicrostructureSimulator` | `research/simulation/microstructure_simulator.py` | LOB simulator, tick data generator | **SIMULATION** |
| `WhaleTracker` | `research/onchain_advanced/whale_tracker.py` | Per-asset thresholds, z-score significance | **ON-CHAIN** |
| `NetworkValue` | `research/onchain_advanced/network_value.py` | NVT, Metcalfe, MVRV, S2F | **ON-CHAIN** |

### Python -- Optimization

| Tool / Primitive | Path | Role | Flag |
|---|---|---|---|
| `BayesianOptimizer` | `optimization/bayesian_optimizer.py` | Matern 5/2 GP + EI acquisition | **OPTIMIZER** |
| `HyperparameterSearch` | `optimization/hyperparameter_search.py` | NSGA-II with hypervolume, Sobol sampling | **OPTIMIZER** |
| `RegimeParameterOptimizer` | `optimization/regime_parameter_optimizer.py` | Per-regime Optuna studies, SQLite store | **OPTIMIZER** |
| `ParameterLandscape` | `spacetime/engine/parameter_landscape.py` | DuckDB-cached landscape, gradient, robustness | **ANALYZER** |
| `ScenarioEngine` | `spacetime/engine/scenario_engine.py` | 12 scenarios, parallel ThreadPoolExecutor | **SIMULATION** |

### Rust -- Performance Primitives

| Tool / Primitive | Crate | Role | Flag |
|---|---|---|---|
| `genome-engine` | `idea-engine/rust/` | NSGA-II multi-objective genome evolution | **OPTIMIZER** |
| `idea-genome-engine` | `crates/idea-genome-engine/` | Crossover/mutation/selection/constraint strategies | **OPTIMIZER** |
| `counterfactual-oracle` | `idea-engine/rust/` | Counterfactual scenario runner | **ORACLE** |
| `tick-backtest` | `crates/tick-backtest/` | Tick-level BH backtest (rayon parallel) | **BACKTEST ENGINE** |
| `larsa-core` | `crates/larsa-core/` | Core BH engine (Rust port, PyO3) | **PRIMITIVE** |
| `larsa-wasm` | `crates/larsa-wasm/` | BH trajectory, heatmap, efficient frontier (WASM) | **WASM** |
| `portfolio-engine` | `crates/portfolio-engine/` | Ledoit-Wolf, HRP, Black-Litterman | **OPTIMIZER** |
| `risk-engine` | `crates/risk-engine/` | VaR/CVaR, Greeks, stress scenarios | **RISK** |
| `monte-carlo-engine` | `crates/monte-carlo-engine/` | GBM, Merton, Heston, Longstaff-Schwartz | **SIMULATION** |
| `online-learning` | `crates/online-learning/` | FTRL, PA-II, Hedge, Adam, bandits | **ML** |
| `rl-exit-optimizer` | `crates/rl-exit-optimizer/` | Double DQN, PER, reward shaping | **RL** |
| `regime-analytics` | `crates/regime-analytics/` | HMM (Baum-Welch), transition model, conditional perf | **ANALYTICS** |
| `smart-order-router` | `crates/smart-order-router/` | PoV strategy, dark pool router, liquidity aggregator | **EXECUTION** |
| `microstructure-engine` | `crates/microstructure-engine/` | VPIN, OFI, effective spread, regime signals | **MICROSTRUCTURE** |
| `orderbook-sim` | `crates/orderbook-sim/` | Synthetic orderbook, adversarial testing | **SIMULATION** |
| `fix-engine` | `crates/fix-engine/` | FIX 4.2 session, execution report parser, order manager | **PROTOCOL** |
| `execution-optimizer` | `crates/execution-optimizer/` | Almgren-Chriss trajectory, adaptive urgency | **EXECUTION** |
| `data-pipeline` | `crates/data-pipeline/` | Quality checks, timeseries ops, HDR histogram metrics | **DATA** |
| `factor-analytics` | `crates/factor-analytics/` | IC, factor decay, Fama-MacBeth | **ANALYTICS** |
| `alpha-decay` | `crates/alpha-decay/` | IC half-life, signal retirement scoring | **ANALYTICS** |
| `fractal-analysis` | `crates/fractal-analysis/` | Hurst exponent, DFA, multifractal | **ANALYTICS** |
| `order-flow-engine` | `crates/order-flow-engine/` | Order flow prediction, liquidity provision (Avellaneda-Stoikov) | **MICROSTRUCTURE** |

-> **[Full Rust crates reference](docs/rust_crates.md)**

### Julia -- Statistical Primitives

> Full module reference: **[Statistical Tooling docs](docs/statistical_tooling.md)**

| Module | Location | Key Capability | Flag |
|---|---|---|---|
| `BHPhysics.jl` | `julia/src/` | BH engine, walk-forward, cross-sectional | **CORE ENGINE** |
| `Stochastic.jl` | `julia/src/` | GARCH, Heston, Hawkes, OU, Merton JD | **STOCHASTIC** |
| `Bayesian.jl` | `julia/src/` | MCMC, Bayesian CF estimation | **INFERENCE** |
| `SystemicRisk.jl` | `julia/src/` | CoVaR, MES, SRISK, Eisenberg-Noe, DebtRank | **RISK** |
| `NumericalMethods.jl` | `julia/src/` | PDE solvers, Halton/Sobol MC, quadrature | **NUMERICAL** |
| `TimeSeriesAdvanced.jl` | `julia/src/` | SARIMA, Kalman/RTS, DFM, Granger, VECM | **TIME SERIES** |
| `CryptoDefi.jl` | `julia/src/` | AMM pricing, Uniswap v3, IL, MEV | **CRYPTO** |
| `ExecutionAnalytics.jl` | `julia/src/` | VWAP/TWAP, Almgren-Chriss, TCA | **EXECUTION** |
| `volatility_surface.jl` | `idea-engine/stats-service/julia/` | SVI, SABR, Dupire local vol, variance swaps | **VOLATILITY** |
| `alpha_research.jl` | `idea-engine/stats-service/julia/` | IC/ICIR pipeline, factor decay, quintile bt | **ALPHA** |
| `stochastic_control.jl` | `idea-engine/stats-service/julia/` | HJB PDE, optimal control | **CONTROL** |
| `copula_models.jl` | `idea-engine/stats-service/julia/` | Gaussian/Clayton/Gumbel copulas, tail dep | **DEPENDENCE** |
| `machine_learning_advanced.jl` | `idea-engine/stats-service/julia/` | GBM, neural net from scratch, CV, SHAP | **ML** |
| `reinforcement_learning.jl` | `idea-engine/stats-service/julia/` | Q-learning, policy gradient, DQN | **RL** |
| `numerical_methods.jl` | `idea-engine/stats-service/julia/` | Halton/Sobol, variance reduction, PDE | **NUMERICAL** |

*40 Julia source modules + 34 notebooks. See [Statistical Tooling](docs/statistical_tooling.md) for full list.*

### R -- Statistical Analysis

> Full module reference: **[Statistical Tooling docs](docs/statistical_tooling.md)**

| Module | Location | Key Capability | Flag |
|---|---|---|---|
| `bh_analysis.R` | `r/R/` | BH state reconstruction, regime classification | **CORE** |
| `regime_models.R` | `r/R/` | HMM, Markov switching, GARCH-DCC | **REGIME** |
| `volatility_models.R` | `idea-engine/stats-service/r/` | GARCH, EGARCH, realized vol, HAR | **VOLATILITY** |
| `spectral_analysis.R` | `idea-engine/stats-service/r/` | FFT, wavelet decomposition, periodogram | **SPECTRAL** |
| `copula_analysis.R` | `idea-engine/stats-service/r/` | Copula fitting, tail dependence | **DEPENDENCE** |
| `bayesian_portfolio.R` | `idea-engine/stats-service/r/` | Bayesian Black-Litterman, shrinkage | **PORTFOLIO** |
| `systemic_risk.R` | `r/R/` | CoVaR, MES, SRISK, DebtRank contagion | **RISK** |
| `advanced_ml.R` | `idea-engine/stats-service/r/` | SVM, GP regression, MLP+Adam, attention | **ML** |
| `time_series_advanced.R` | `r/R/` | TBATS, DFM via EM, DCC-GARCH, BEKK | **TIME SERIES** |
| `stress_testing.R` | `r/R/` | COVID/LUNA/FTX/Apr-2026 historical scenarios | **RISK** |
| `signal_research.R` | `r/R/` | IC/ICIR, Fama-MacBeth, factor decay half-life | **ALPHA** |
| `walk_forward_analysis.R` | `idea-engine/stats-service/r/` | CPCV walk-forward, Sobol optimization | **VALIDATION** |

*29 R modules across 3 directories + 12 research scripts. See [Statistical Tooling](docs/statistical_tooling.md) for full list.*

### C/C++ -- Low-Latency Primitives

> Deep dive: **[C++ Signal Engine](docs/signal_engine_cpp.md)** | **[Native Layer](docs/native_layer.md)**

| Component | Path | Role | Flag |
|---|---|---|---|
| `BHState` (C++) | `cpp/signal-engine/src/bh_physics/` | Sub-millisecond BH mass accumulation | **PRIMITIVE** |
| `QuatNav` (C++) | `cpp/signal-engine/src/quaternion/` | C++ quaternion nav | **PRIMITIVE** |
| `GARCHState` (C++) | `cpp/signal-engine/src/bh_physics/` | GARCH(1,1) vol forecaster | **PRIMITIVE** |
| `KalmanFilter` (C++) | `cpp/signal-engine/src/indicators/` | 1D/2D/Adaptive/Pair Kalman filters, C ABI | **PRIMITIVE** |
| `TickIndicators` (C++) | `cpp/signal-engine/src/indicators/` | TickImbalance, VPIN, MicropriceCalculator, OFI | **PRIMITIVE** |
| `MultiTimeframeSignal` (C++) | `cpp/signal-engine/src/composite/` | 4H/1H/15M aggregation (0.50/0.30/0.20), 4H override | **PRIMITIVE** |
| `LorentzBoost` (C++) | `cpp/signal-engine/src/bh_physics/` | Minkowski boost, worldline integrator, proper time | **PRIMITIVE** |
| `ZmqPublisher` (C++) | `cpp/signal-engine/src/io/` | In-process bus, batch publisher, signal router | **PRIMITIVE** |
| Fast indicators (20 signals) | `cpp/signal-engine/src/indicators/` | SIMD-accelerated RSI/MACD/BB/ATR/VWAP | **PRIMITIVE** |
| `SignalOutput` struct | `cpp/signal-engine/include/srfm/types.hpp` | 320-byte output frame (5 cache lines) | **PRIMITIVE** |
| `RingBuffer` | `cpp/signal-engine/include/srfm/ring_buffer.hpp` | Lock-free SPSC bar event queue | **PRIMITIVE** |
| L3 orderbook (C) | `native/orderbook/` | AVX2 price ladder, VWAP fill estimation | **PRIMITIVE** |
| SIMD matrix (C) | `native/matrix/` | AVX2 matmul for portfolio covariance | **PRIMITIVE** |

### Zig -- Ultra Low Latency

> Deep dive: **[Native Layer](docs/native_layer.md)**

| Component | Path | Role | Flag |
|---|---|---|---|
| ITCH 5.0 decoder | `native/zig/itch/` | NASDAQ ITCH 5.0 binary protocol parser, 4 GB/s | **DECODER** |
| Lock-free L2 book | `native/zig/orderbook/` | ~180ns add/cancel, ~15ns best bid/ask | **PRIMITIVE** |
| Bar compression | `native/zig/src/compression.zig` | Delta+RLE encoding, ~10x compression, C ABI | **PRIMITIVE** |
| Tick processor | `native/zig/src/tick_processor.zig` | SPSC ring buffer, bar aggregator, VWAP | **PRIMITIVE** |
| SIMD indicators | `native/zig/src/simd_indicators.zig` | EMA/SMA/RSI/ATR/Bollinger/MACD vectorized | **PRIMITIVE** |
| Order flow | `native/zig/src/order_flow.zig` | FootprintBar, CumulativeDelta, BuyPressure, VPIN | **PRIMITIVE** |
| Lock-free ring buffer (C) | `native/ringbuffer/` | 180M ops/sec SPSC, cache-line padded | **PRIMITIVE** |

### Go -- Market Data and Microservices

> Deep dive: **[Market Data Service](docs/market_data_service.md)**

| Service | Port | Path | Role | Flag |
|---|---|---|---|---|
| Market Data | `:8780` | `market-data/` | L2 aggregation, 15m bar assembly, WebSocket fan-out | **LIVE** |
| IAE API | `:8767` | `idea-engine/cmd/api/` | Hypotheses, patterns, backtest jobs | **API** |
| IAE Event Bus | `:8768` | `idea-engine/cmd/bus/` | Pub/sub: pattern_confirmed, backtest_complete | **MESSAGE BUS** |
| IAE Scheduler | `:8769` | `idea-engine/cmd/scheduler/` | Cron: mine (1h/4h/daily), tune (4h) | **SCHEDULER** |
| IAE Webhook | `:8770` | `idea-engine/cmd/webhook/` | Alpaca fill ingest + external alerts | **WEBHOOK** |
| Metrics Server | internal | `idea-engine/cmd/metrics-server/` | Ring buffers, Prometheus /metrics, ingest endpoints | **METRICS** |
| Genome Inspector | CLI | `idea-engine/cmd/genome-inspector/` | ANSI CLI: list/best/compare/history/stats | **CLI** |
| Correlation Tracker | internal | `market-data/pkg/analytics/` | Rolling cross-asset correlations | **ANALYTICS** |
| Regime Classifier | internal | `market-data/pkg/analytics/` | Market regime from bar stream | **ANALYTICS** |
| Bar Compressor | internal | `market-data/pkg/storage/` | Tiered cache: L1 ring + L2 SQLite, 252d retention | **STORAGE** |
| Research API | `:8766` | `infra/research-api/` | Research data query API | **API** |
| Alerter | internal | `cmd/alerter/` | Slack/email/PagerDuty alert routing | **INFRA** |
| TUI | terminal | `cmd/srfm-tui/` | Terminal live trading dashboard | **UI** |

Key Go primitives:
- `BookManager` (market-data): dual-feed failover state machine, 30s Alpaca timeout
- `BarAggregator` (market-data): tick-to-OHLCV, wall-clock-anchored 15m windows
- `WebSocketHub` (market-data): fan-out to all subscribers, 100-message slow-consumer limit
- `AlertEngine` (cmd/alerter): dedup, snooze, maintenance window, multi-channel routing
- `GenomeStore` (idea-engine/pkg/persistence): lineage BFS, generational pruning
- `FitnessAggregator` (idea-engine/pkg/evaluation): 3-period weighted Sharpe + Pareto rank
- `TieredCache` (market-data/pkg/cache): L1 ring + L2 SQLite, 252-day retention

### Elixir/OTP -- Coordination Layer

> Deep dive: **[Coordination Layer](docs/coordination_layer.md)**

| Component | Port | Role | Flag |
|---|---|---|---|
| `ParameterCoordinator` | `:8781` | Validates + fans out IAE parameter updates | **LIVE** |
| `CircuitBreaker` | `:8781` | Per-API fault isolation (Alpaca, Binance, Polygon) | **LIVE** |
| `HealthMonitor` | `:8781` | 30s health polls, automatic service restarts | **LIVE** |
| `EventBus` | `:8781` | In-process pub/sub with 1000-event ETS history | **LIVE** |
| `ServiceRegistry` | `:8781` | ETS-backed service PID + metadata registry | **LIVE** |
| `GenomeBridge` | `:8781` | HTTP bridge to IAE genome engine | **LIVE** |
| `PerformanceLedger` | `:8781` | ETS + SQLite cycle performance tracking | **LIVE** |
| `ConfigBroadcast` | `:8781` | Ack-tracked config fanout with retry | **LIVE** |
| `AlertManager` | `:8781` | Dedup, rate limit, multi-channel alerts | **LIVE** |
| `SessionManager` | `:8781` | Session lifecycle state machine | **LIVE** |
| `MetricsAggregator` | `:8781` | Prometheus text-format export | **LIVE** |

### TypeScript/React -- Dashboards

| Component | Port | Path | Role | Flag |
|---|---|---|---|---|
| IAE Dashboard | `:5175` | `idea-engine/idea-dashboard/` | Genome evolution, patterns, backtests | **UI** |
| IAE Evolution | internal | `terminal/src/pages/IAEEvolution.tsx` | Parallel coords, heatmap, pedigree SVG | **UI** |
| Signal Evolution | internal | `terminal/src/pages/SignalEvolution.tsx` | Signal tree, gene contribution, novelty scatter | **UI** |
| BH Physics Panel | internal | `terminal/src/components/bh/BHPhysicsPanel.tsx` | Quaternion sphere, spacetime flow field | **UI** |
| Walk Forward | internal | `spacetime/web/src/pages/WalkForward.tsx` | IS/OOS grid, fold timeline | **UI** |
| Factor Analysis | internal | `spacetime/web/src/pages/FactorAnalysis.tsx` | 7-factor attribution, Fama-MacBeth | **UI** |
| Spacetime Web | `:5173` | `spacetime/web/` | BH backtester UI, parameter sweep | **UI** |
| Research Dashboard | `:5174` | `research/dashboard/` | Risk, signal research, on-chain | **UI** |

### Infrastructure

| Component | Description | Flag |
|---|---|---|
| `docker-compose.yml` | 5-service deployment (trader/monitor/supervisor/orderbook/bridge) | **DEPLOY** |
| `Dockerfile.python` | 2-stage build, non-root srfm user | **DEPLOY** |
| `scripts/supervisor.py` | HTTP :8790 process supervisor, exponential backoff restart | **INFRASTRUCTURE** |
| `scripts/start_all.sh` | Health-check loop, auto-restart, stop/restart/status | **INFRASTRUCTURE** |
| `scripts/daily_startup.py` | 9-step orchestrated daily startup | **INFRASTRUCTURE** |
| `scripts/daily_shutdown.py` | 10-step EOD shutdown with P&L capture | **INFRASTRUCTURE** |
| `scripts/emergency_stop.py` | Halt + flatten all positions + PD alert | **INFRASTRUCTURE** |
| `scripts/param_update_manual.py` | Diff + confirm + audit parameter changes | **INFRASTRUCTURE** |
| `config/signal_overrides.json` | Hot-reloaded per-symbol + global multipliers | **CONFIG** |
| `config/instruments.yaml` | 32 instruments with CF calibration | **CONFIG** |
| `warehouse/warehouse_manager.py` | DuckDB analytics warehouse, migration runner, TCA queries | **DATA** |
| `.github/workflows/` | CI/CD for Python/Rust/Go/TS/Julia | **CI/CD** |

---

## Stack

| Language | LOC | Key Systems | Docs |
|---|---|---|---|
| Python | ~528K | Live trader (LARSA v18), backtesting, IAE pipeline, ML pipeline, options analytics, risk API, regime ensemble, execution algos, broker adapters, config management, research validation, optimization | [Execution Stack](docs/execution_stack.md) |
| Julia | ~123K | Advanced options (Heston/SABR/Merton/Dupire), live risk (VaR/CVaR/stress), ML signals (GP/Kalman/HMM), vectorized backtesting (CPCV/DSR), numerical methods (PDE/Sobol/quadrature) | [Statistical Tooling](docs/statistical_tooling.md) |
| Rust | ~141K | 27 crates: genome engine, Monte Carlo, portfolio, risk, online-learning (FTRL/Hedge/bandits), RL exit optimizer, regime-analytics, smart-order-router, microstructure, FIX engine, WASM analytics | [Rust Crates Reference](docs/rust_crates.md) |
| R | ~60K | HMM, regime models, WFA, factor analysis, options risk, microstructure, stress testing, copulas, spectral | [Statistical Tooling](docs/statistical_tooling.md) |
| TypeScript/React | ~58K | IAE dashboard, signal evolution, BH physics panel, walk-forward, factor analysis, spacetime UI, research dashboards | [Stack Overview](docs/stack_overview.md) |
| Go | ~84K | IAE microservices (API/bus/scheduler/webhook/metrics), market data (L2 agg/bar assembly/WebSocket), genome inspector, alerter, TUI | [Market Data Service](docs/market_data_service.md) |
| C/C++ | ~19K | Signal engine (Kalman, tick indicators, multi-timeframe, Lorentz boost, ZMQ), L3 orderbook (AVX2), matrix ops | [C++ Signal Engine](docs/signal_engine_cpp.md) |
| Zig | ~10K | ITCH 5.0 decoder, lock-free L2 book, bar compression, tick processor, SIMD indicators, order flow | [Native Layer](docs/native_layer.md) |
| Elixir/OTP | ~12K | Coordination: OTP supervision, circuit breakers, param validation, rollback, genome bridge, alert manager | [Coordination Layer](docs/coordination_layer.md) |
| SQL | ~7K | SQLite (16 migrations, WAL), DuckDB analytics, BH UDFs, warehouse views, TCA queries | [Stack Overview](docs/stack_overview.md) |
| **Total** | **~1,039,342** | **2,384 source files** | |

---

## File Structure

```
srfm-lab/
|
+-- tools/                               # Core trading tools
|   +-- live_trader_alpaca.py            # LIVE TRADER: BH Physics + GARCH + OU + IAE params
|   +-- crypto_backtest_mc.py            # BACKTEST: BH + 10K-path Monte Carlo
|   +-- backtest_wave4.py                # WAVE 4: EventCalendar + Granger + ML signal
|   +-- walk_forward_engine.py           # Walk-forward analysis
|   +-- factor_analysis.py               # Fama-MacBeth, IC/ICIR, factor decay
|   +-- larsa_v18_backtest.py            # LARSA v18 full backtest
|   +-- stress_testing.py                # 20 stress scenarios
|   +-- backtest_output/                 # SQLite DBs, CSV trade logs, PNG charts
|
+-- idea-engine/                         # IAE (Idea Automation Engine)
|   +-- db/                              # Schema migrations (SQLite WAL)
|   +-- ingestion/                       # 4-stage pipeline: load -> mine -> filter -> store
|   +-- analysis/                        # GenomeAnalyzer, PerformanceTracker, ParameterExplorer
|   +-- autonomous-loop/                 # LoopController, PerformanceEvaluator, CycleReporter (Go)
|   +-- strategy-lab/                    # ChampionManager, ExperimentTracker, Versioner (Go)
|   +-- genome/                          # Rust NSGA-II genome evolution
|   +-- hypothesis/                      # Bayesian hypothesis generator
|   +-- causal/                          # Granger + PC algorithm causal discovery
|   +-- walk_forward/                    # CPCV walk-forward engine
|   +-- regime/                          # 6-regime oracle
|   +-- signals/                         # 105+ signal library with IC tracking
|   +-- cmd/                             # Go: API :8767, bus :8768, scheduler :8769, webhook :8770
|   +-- cmd/genome-inspector/            # ANSI CLI: list/best/compare/history/stats
|   +-- cmd/metrics-server/              # Ring buffers, Prometheus /metrics
|   +-- stats-service/julia/             # 40 Julia modules
|   +-- stats-service/r/                 # 29 R modules
|   +-- idea-dashboard/                  # React/TS + Vite + Recharts + D3 (:5175)
|
+-- execution/                           # Execution infrastructure
|   +-- orderbook/                       # L2 book, Alpaca/Binance feeds, BookManager, FeedMonitor
|   +-- routing/smart_router.py          # Spread-tier routing (<=50/50-100/>100 bps)
|   +-- broker_adapters/                 # Alpaca, Binance, Paper, AdapterManager
|   +-- order_management/               # TWAP/VWAP/Iceberg engines, AlgoScheduler, OrderBookTracker
|   +-- oms/                             # FillProcessor, OrderRouter, StateMachine
|   +-- audit/                           # ComplianceLogger (hash-chain tamper-evident)
|   +-- risk/                            # VaR, attribution, correlation monitor, FastAPI :8791
|
+-- crates/                              # 27 Rust crates
|   +-- idea-genome-engine/             # Crossover, mutation, selection, constraint strategies
|   +-- monte-carlo-engine/             # GBM, Merton, Heston, Longstaff-Schwartz, VaR
|   +-- online-learning/                # FTRL, PA-II, Hedge, Adam, bandits
|   +-- rl-exit-optimizer/              # Double DQN, PER, reward shaping
|   +-- regime-analytics/               # HMM, transition model, conditional performance
|   +-- smart-order-router/             # PoV, dark pool, liquidity aggregator
|   +-- microstructure-engine/          # VPIN, OFI, effective spread, regime signals
|   +-- orderbook-sim/                  # Synthetic orderbook, adversarial testing
|   +-- fix-engine/                     # FIX 4.2 session, execution reports, order manager
|   +-- execution-optimizer/            # Almgren-Chriss trajectory, adaptive urgency
|   +-- tick-backtest/                  # Tick replay, intraday patterns, bar-from-ticks
|   +-- data-pipeline/                  # Quality checks, timeseries ops, HDR histogram
|   +-- larsa-core/                     # Core BH engine (PyO3)
|   +-- larsa-wasm/                     # BH trajectory + portfolio analytics (WASM)
|   +-- portfolio-engine/               # Ledoit-Wolf, HRP, Black-Litterman
|   +-- risk-engine/                    # VaR/CVaR, Greeks, stress scenarios
|   +-- order-flow-engine/              # Order flow prediction, Avellaneda-Stoikov MM
|   +-- [10 more crates]
|
+-- cpp/signal-engine/                   # C++ signal engine (~16K LOC)
|   +-- src/bh_physics/                  # BHState, GARCHState, OUDetector, LorentzBoost
|   +-- src/indicators/                  # KalmanFilter (4 variants), TickIndicators
|   +-- src/composite/                   # MultiTimeframeSignal (4H override rule)
|   +-- src/io/                          # ZmqPublisher, InProcessBus, BatchPublisher
|   +-- src/quaternion/                  # QuatNav
|
+-- native/                              # Zig and C ultra-low-latency
|   +-- zig/itch/                        # ITCH 5.0 decoder (4 GB/s)
|   +-- zig/orderbook/                   # Lock-free L2 book (~180ns)
|   +-- zig/src/compression.zig          # Delta+RLE bar compression
|   +-- zig/src/tick_processor.zig       # SPSC ring buffer, bar aggregator
|   +-- zig/src/simd_indicators.zig      # Vectorized EMA/RSI/ATR/MACD
|   +-- zig/src/order_flow.zig           # FootprintBar, VPIN, CumulativeDelta
|   +-- orderbook/                       # C L3 book (AVX2, individual order IDs)
|   +-- matrix/                          # C AVX2 matmul (covariance)
|   +-- ringbuffer/                      # C SPSC ring (180M ops/sec)
|
+-- coordination/                        # Elixir/OTP coordination layer (:8781)
|   +-- lib/srfm_coordination/           # 11 GenServer modules
|
+-- config/                              # Config and risk management
|   +-- config_manager.py               # Dot-notation, file watcher, subscribers
|   +-- instruments_manager.py          # 32-instrument universe
|   +-- event_calendar_manager.py       # FOMC/CPI/NFP/OpEx blackout dates
|   +-- risk_config.py                  # Risk limits, BH-aware position sizing
|   +-- instruments.yaml                # BH physics calibration per instrument
|
+-- warehouse/                           # DuckDB analytics warehouse
|   +-- migrations/                      # 020 migrations (schema versioning)
|   +-- schema/                          # Views: IAE analytics, execution quality
|   +-- queries/                         # Named query files with parameter substitution
|   +-- warehouse_manager.py            # Migration runner, upsert, analytics shortcuts
|
+-- research/                            # Research tooling
|   +-- validation/                      # CPCV, DSR, causal inference, market efficiency
|   +-- portfolio_lab/                   # Efficient frontier, factor exposure, attribution
|   +-- simulation/                      # Agent-based model, microstructure simulator
|   +-- onchain_advanced/               # Whale tracker, miner metrics, stablecoin flows
|   +-- signal_analytics/               # 105-signal library, IC/ICIR, alpha decay
|   +-- walk_forward/                    # CPCV + Sobol/Bayesian param opt
|
+-- optimization/                        # Parameter optimization
|   +-- bayesian_optimizer.py           # GP + EI acquisition
|   +-- hyperparameter_search.py        # NSGA-II with hypervolume
|   +-- regime_parameter_optimizer.py   # Per-regime Optuna studies
|
+-- ml/                                  # Machine learning pipeline
|   +-- training/cross_validator.py     # CPCV + DSR training CV
|   +-- nlp_alpha/                       # NLP alpha signals
|
+-- julia/src/                           # 40 Julia modules (~123K LOC)
+-- r/R/                                 # 21 R modules (~60K LOC)
+-- bridge/                              # On-chain bridge (MVRV, VPIN, Kyle's Lambda)
+-- spacetime/                           # Spacetime Arena (BH backtester + web UI)
+-- lib/                                 # Core Python primitives
+-- infra/                               # Observability, gRPC, event bus
+-- db/                                  # SQLite schema (16 migrations)
+-- terminal/                            # Terminal UI TypeScript components
+-- dashboard/                           # React dashboards (risk, signal, on-chain)
+-- docs/                                # All deep-dive documentation
+-- scripts/                             # Operational scripts (startup/shutdown/emergency)
+-- run_full_analysis.py                 # Macro + on-chain + alt data + IAE pipeline
+-- docker-compose.yml                   # 5-service deployment
+-- Makefile                             # 60+ targets
+-- .github/workflows/                   # CI/CD: Python/Rust/Go/TS/Julia
```

---

## Service Endpoints

| Service | Command | Port | Flag |
|---|---|---|---|
| Live Trader | `python tools/live_trader_alpaca.py` | -- | **LIVE** |
| Process Supervisor | `python scripts/supervisor.py` | `:8790` | **INFRA** |
| IAE API | `cd idea-engine && go run cmd/api/main.go` | `:8767` | **API** |
| IAE Event Bus | `go run cmd/bus/main.go` | `:8768` | **MESSAGE BUS** |
| IAE Scheduler | `go run cmd/scheduler/main.go` | `:8769` | **SCHEDULER** |
| IAE Webhook | `go run cmd/webhook/main.go` | `:8770` | **WEBHOOK** |
| IAE Metrics | `go run cmd/metrics-server/main.go` | `:8771` | **METRICS** |
| IAE Dashboard | `cd idea-engine/idea-dashboard && npm run dev` | `:5175` | **UI** |
| Coordination Layer | `cd coordination && mix run --no-halt` | `:8781` | **COORDINATION** |
| Research API | `cd infra/research-api && go run main.go` | `:8766` | **API** |
| Spacetime API | `python run_api.py` | `:8765` | **API** |
| Spacetime Web | `cd spacetime/web && npm run dev` | `:5173` | **UI** |
| Research Dashboard | `cd research/dashboard && npm run dev` | `:5174` | **UI** |
| Risk Aggregator | `python -m execution.risk.api` | `:8791` | **RISK** |
| Live Monitor (CLI) | `python -m research.live_monitor.cli monitor run` | -- | **MONITOR** |

---

## Development Commands

### Backtesting

```bash
python tools/crypto_backtest_mc.py --verbose
python tools/backtest_wave4.py                            # Wave 4 with ML + Granger
cargo run -p tick-backtest -- sweep --data-dir data/ --sym BTC --n-trials 1000
python -m research.walk_forward.cli wf optimize \
  --trades tools/backtest_output/crypto_trades.csv \
  --method sobol --n-iter 200
python -m research.regime_lab.cli regime stress \
  --trades tools/backtest_output/crypto_trades.csv
```

### Live Trading

```bash
bash scripts/start_all.sh start          # All services
bash scripts/start_all.sh status         # Health check
bash scripts/start_all.sh stop
python tools/live_trader_alpaca.py --dry-run --log-level DEBUG
python scripts/daily_startup.py          # Full 9-step startup orchestration
```

### IAE

```bash
python -m idea_engine.db.migrate
python -m idea_engine.ingestion.pipeline --verbose
cd idea-engine/rust && cargo build --release
cd idea-engine/rust && ./target/release/genome-engine --generations 100 --pop-size 200
cd idea-engine && go run cmd/genome-inspector/main.go list --top 10
```

### Native Layer

```bash
# Zig components
cd native/zig && zig build -Doptimize=ReleaseFast
cd native/zig && zig test src/simd_indicators.zig

# C components
cd native/matrix && make
cd native/orderbook && make

# Or all at once
make native
```

### Rust Crates

```bash
cargo test --workspace
cargo build --workspace --release
cargo run -p rl-exit-optimizer -- --train --episodes 10000
cargo run -p genome-inspector -- best --top 5
```

### Julia Statistical Tooling

```bash
julia julia/src/SystemicRisk.jl
julia julia/src/CryptoDefi.jl
julia idea-engine/stats-service/julia/volatility_surface.jl
julia idea-engine/stats-service/julia/alpha_research.jl
```

### R Statistical Analysis

```bash
Rscript r/R/regime_models.R
Rscript r/R/systemic_risk.R
Rscript idea-engine/stats-service/r/volatility_surface.R
```

### Build and Test

```bash
pytest tests/ -v
cargo test --workspace
cd idea-engine && go test ./...
cd idea-engine/idea-dashboard && npm test
```

---

## BH Physics Reference

-> **[Full deep dive with worked example](docs/bh_physics.md)**

| Component | Formula | Interpretation |
|---|---|---|
| MinkowskiClassifier | `ds^2 = c^2*dt^2 - dx^2` | TIMELIKE (ds^2>0) = ordered, causal; SPACELIKE = anomalous |
| BH Formation | `mass >= BH_FORM (1.92)` | Gravitational well forms; EMA asymptotes to 2.0 |
| Mass Accrual | `mass = 0.97*mass + 0.03*min(2, 1+ctl*0.1)` | Consecutive timelike bars build conviction |
| Mass Decay | `mass *= BH_DECAY (0.924)` | Noise bars bleed mass away |
| Hawking Monitor | `T_H = 1/(8*pi*M)` | Cold well = stable signal; hot well = reduce size |
| Delta Score | `tf_score * mass * ATR` | Expected dollar move -> allocation signal |
| OU Overlay | `dX = theta*(mu-X)*dt + sigma*dW` | Mean reversion on flat BH; 8% equity |
| Mayer Dampener | `scale = min(1, 2*MA200/price)` | Reduces size when price is extended |
| BTC Lead | `alt_score *= (1 + btc_active * 0.3)` | BTC activation boosts correlated alts |
| Dynamic CORR | `0.25 base -> 0.60 when 30d pair-corr > 0.60` | Stress regime portfolio risk reduction |
| Hurst Exponent | `H > 0.58 trending, H < 0.42 mean-reverting` | R/S analysis over HURST_WINDOW=100 bars |
| GARCH(1,1) | `h_t = omega + alpha*eps^2_{t-1} + beta*h_{t-1}` | Conditional variance, targets GARCH_TARGET_VOL |

---

## Key Parameters

| Parameter | Default | IAE Tuned | Effect |
|---|---|---|---|
| `BH_FORM` | 1.92 | -- | Mass threshold for BH activation |
| `BH_DECAY` | 0.924 | -- | Mass bleed rate on noise bars |
| `BH_COLLAPSE` | 0.992 | -- | Collapse multiplier on exit |
| `CF` (per instrument) | 0.001-0.025 | -- | Minkowski speed of light |
| `CORR` | dynamic | dynamic | Cross-asset correlation (0.25/0.60) |
| `GARCH_TARGET_VOL` | 1.20 | **0.90** | Target annualized volatility |
| `OU_FRAC` | 0.08 | -- | OU mean-reversion allocation |
| `MIN_HOLD` | 4 | **8** | Minimum bars before exit |
| `BLOCKED_ENTRY_HOURS` | {} | **{1,13,14,15,17,18}** | UTC hours blocked for entries |
| `BOOST_HOURS` | {} | **{3,9,16,19}** | UTC hours with 1.25x size boost |
| `WINNER_PROTECTION_PCT` | 0.001 | **0.005** | Threshold to let winners run |
| `OU_DISABLED_SYMBOLS` | {} | **{AVAX,DOT,LINK}** | Momentum symbols, skip OU |
| `DELTA_MAX_FRAC` | 0.40 | -- | Max single-instrument allocation |
| `MIN_TRADE_FRAC` | 0.03 | -- | Minimum equity shift to rebalance |
| `NAV_OMEGA_SCALE_K` | 0.5 | -- | Quaternion angular velocity scale |
| `NAV_GEO_ENTRY_GATE` | 3.0 | -- | Geodesic quality threshold |
| `HURST_WINDOW` | 100 | -- | Bars for R/S Hurst estimation |
| `BH_MASS_THRESH` | 1.92 | -- | Alias for BH_FORM in Rust/C++ |
| `RL_EXIT_EPSILON` | 0.05 | -- | Epsilon-greedy exploration in DQN trainer |

*IAE Tuned = parameter updated by IAE analysis of 63,993 backtest trades.*

---

## IAE Live Research Output

The IAE ingested 63,993 backtest trades (Jan 2024 - Apr 2026) and produced 10 actionable ideas. All 9 high-confidence ideas are now live in `tools/live_trader_alpaca.py`.

```
#1 [91%] EXIT RULE: Raise min_hold_bars 4 -> 8
   1-bar holds: avg P&L=-169, WR=35.5% (16,939 trades = 26.5% of all trades)
   5-12 bar holds: avg P&L=+111, WR=46.4%. Eliminating fast exits is the
   single highest-leverage change.

#2 [88%] ENTRY TIMING: Block entries at hours 1, 13, 14, 15, 18 UTC
   These hours: avg P&L=-131/trade vs -10 baseline. WR drops to 37%.
   Hour 1 UTC worst (-179/trade, 33.2% WR) -- thin Asian/European overlap.

#3 [85%] CROSS-ASSET: BTC as signal only, reduce BTC direct trade
   BTC is the worst P&L instrument (-156K) but the lead signal for alts.
   Cut BTC cf_scale to 0.5, boost alt allocation 1.4x when BTC-lead fires.

#4 [82%] INSTRUMENT FILTER: Remove GRT + SOL, shrink AVAX/DOT/LINK
   5 symbols = -390K combined loss. GRT (37.4% WR), SOL (36.4% WR).

#5 [80%] POSITION SIZING: GARCH target_vol 120% -> 90%
   Overtrading in high-vol regimes. Tightening GARCH cuts ~25% of trades.

#6 [78%] EXIT RULE: Winner protection 0.1% -> 0.5%
   48+ bar trades avg +610/trade (only 419 trades). Cutting winners too early.
```

Backtest comparison after applying all 6 ideas:

| Metric | Baseline | After IAE |
|---|---|---|
| Trades | 63,993 | 59,326 (-7%) |
| Win rate | 41.4% | 43.0% (+1.6pp) |
| MC median 12m | ~$678K | $1.72M |
| MC blowup rate | -- | 0% |

-> **[IAE architecture deep dive](docs/iae_architecture.md)**
-> **[Genome evolution deep dive](docs/genome_evolution.md)**

---

## Performance Notes

Key findings (2021-2026, 19 crypto pairs):
- **LARSA v1 (ES futures):** +274% over backtest window
- **2024 standalone:** +26%, driven by BTC and SOL regime
- **Full period CAGR:** -11% (crypto bear market dominated)
- **Monte Carlo (10,000 paths):** Median outcome captures distribution of sequential trade ordering; blowup rate: 0% after IAE tuning
- **Wave 4 additions:** EventCalendarFilter + Granger lead + ML signal show further improvement in OOS Sharpe

The backtest engine runs identical BH physics to live trading -- GARCH vol scaling, OU overlay, Mayer dampening -- no lookahead, no future data.

-> **[Wave 4 backtest deep dive](docs/wave4_backtest.md)**
-> **[Monte Carlo engine deep dive](docs/monte_carlo.md)**

---

## Latency Reference

| Component | Operation | Latency |
|---|---|---|
| Zig L2 book | Best bid/ask read | ~15ns |
| C SPSC ring buffer | Push + pop round trip | ~5.5ns |
| ITCH 5.0 decoder | Full message parse | ~40ns |
| Zig L2 book | Add/cancel | ~180-190ns |
| RL exit policy | Q-table state lookup | ~100ns |
| C L3 book | VWAP walk (21 levels) | ~850ns |
| SIMD matmul (C) | 21x21 double matrix | ~2.1us |
| C++ KalmanFilter1D | Single update step | ~50ns |
| C++ MultiTimeframe | 3-timeframe aggregate | ~200ns |
| Python function call | Overhead | ~50-100ns |
| SQLite WAL read | Single row | ~10-50us |
| Rust Monte Carlo | 10K GBM paths (252 steps) | ~8ms |

-> **[Full native layer reference](docs/native_layer.md)**
