# SRFM Trading Lab

A full-stack quantitative trading research platform built on **Special Relativistic Financial Mechanics (SRFM)** — from raw tick data to live paper trading and autonomous idea discovery, across 9 languages and **560K+ lines of code**.

> Mad scientist workshop. Everything automated, everything measurable, rapid iteration at scale.

---

## Navigation

| Section | Description |
|---|---|
| [What is This?](#what-is-this) | Core innovation and philosophy |
| [Architecture](#architecture) | System diagram and data flow |
| [Quick Start](#quick-start) | Get running in 5 minutes |
| [Tools & Primitives](#tools--primitives) | Every executable tool flagged |
| [Stack](#stack) | All 9 languages and their roles |
| [File Structure](#file-structure) | Complete directory map |
| [BH Physics Reference](#bh-physics-reference) | Signal math |
| [Key Parameters](#key-parameters) | All tunable constants |
| [IAE Research Output](#iae-live-research-output) | Live findings |

### Deep Documentation

| Doc | Covers |
|---|---|
| [BH Physics Engine](docs/bh_physics.md) | Minkowski metric, mass accumulation, Hawking temperature, delta scoring, full worked example |
| [IAE Architecture](docs/iae_architecture.md) | Genome evolution, hypothesis engine, causal discovery, regime oracle, feedback loop |
| [Quaternion Navigation](docs/quaternion_nav.md) | 4-space bar representation, rotation tracking, geodesic deviation, angular velocity, Lorentz boosts |
| [Execution Stack](docs/execution_stack.md) | L2 orderbook, smart router, spread-tier routing, supervisor, Docker deployment |
| [Wave 4 Backtest](docs/wave4_backtest.md) | EventCalendarFilter, Granger lead signal, ML signal module, 4-variant comparison |
| [Statistical Tooling](docs/statistical_tooling.md) | All Julia and R modules - purpose, key functions, usage |
| [Stack Overview](docs/stack_overview.md) | Every language, what it does, how to run it, integration diagram |

---

## What is This?

The core innovation is the **Black Hole (BH) Physics Strategy** — a signal model derived from special-relativistic mechanics applied to price data. Price bars are classified as *timelike* or *spacelike* using a Minkowski spacetime metric (`ds² = c²dt² − dx²`). Mass accumulates on ordered (causal) bars, and a gravitational well forms when mass crosses the **BH_FORM=1.92** threshold — the black hole formation event that gates entries.

On top of this sits the **Idea Automation Engine (IAE)** — an autonomous research system that runs genetic genome evolution (NSGA-II), causal discovery, regime classification, walk-forward validation, and academic paper mining continuously, feeding confirmed patterns back into live strategy parameters.

→ **[Full BH Physics deep dive](docs/bh_physics.md)**
→ **[Full IAE architecture deep dive](docs/iae_architecture.md)**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SRFM Trading Lab                                    │
│                                                                              │
│  ┌─────────────────────┐    ┌──────────────────────────────────────────┐    │
│  │   Live Trader        │    │         Idea Automation Engine (IAE)     │    │
│  │  live_trader_        │    │                                          │    │
│  │  alpaca.py          │◄───┤  Genome Evolution (NSGA-II Rust)         │    │
│  │                     │    │  Hypothesis Generator (Bayesian)         │    │
│  │  BH Physics Engine  │    │  Regime Oracle (6 modes)                 │    │
│  │  GARCH vol forecast │    │  Causal Discovery (Granger + PC)         │    │
│  │  OU mean reversion  │    │  Walk-Forward (CPCV)                     │    │
│  │  Mayer dampener     │    │  Signal Library (60+ signals, IC decay)  │    │
│  │  BTC lead signal    │    │  Academic Miner (arXiv + SSRN)           │    │
│  │  Dynamic CORR       │    │                                          │    │
│  └──────┬──────────────┘    │  ┌─────────────────────────────────────┐│    │
│         │                   │  │  Event Bus (:8768)  Go API (:8767)  ││    │
│         │ fills             │  │  Scheduler (:8769)  Webhook (:8770) ││    │
│         ▼                   │  └─────────────────────────────────────┘│    │
│  ┌──────────────┐           │  React Dashboard (:5175)                 │    │
│  │ SQLite       │           └──────────────────────────────────────────┘    │
│  │ trade log    │                                                            │
│  │ (WAL mode)   │   ┌────────────────────────────────────────────────────┐  │
│  └──────┬───────┘   │              Execution Stack                       │  │
│         │           │  L2 Orderbook (Alpaca WS + Binance fallback)       │  │
│         │           │  BookManager (30s failover)  FeedMonitor           │  │
│         │           │  SmartRouter (spread-tier: ≤50bps/50-100bps/>100)  │  │
│         │           │  Supervisor (:8790)  Docker-compose (5 services)   │  │
│         │           └────────────────────────────────────────────────────┘  │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────┐   ┌────────────────────────────────────────────┐  │
│  │  live_monitor/        │   │     crypto_backtest_mc.py + wave4          │  │
│  │  CLI + web dashboard │   │  3-TF BH + GARCH + OU + MC (10K paths)    │  │
│  └──────────────────────┘   │  EventCalendar + Granger + ML signal       │  │
│                              └────────────────────────────────────────────┘  │
│                                                                              │
│  Julia (~100K LOC)   R (~45K LOC)   Rust (~24K LOC)   C/C++ (~15K LOC)    │
│  stats-service: copulas, HJB PDE, SVI/SABR, SARIMA, Kalman, AMM, CoVaR   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Data flow:** Alpaca streams → BH Engine → position sizing → SmartRouter → L2 spread check → orders → SQLite log → IAE ingestion → genome evolution → parameter feedback → live strategy.

→ **[Execution stack deep dive](docs/execution_stack.md)**

---

## Quick Start

### Prerequisites

```bash
pip install alpaca-py pandas numpy scipy statsmodels matplotlib
# Rust (genome engine, Monte Carlo)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Go 1.22+ (API, bus, scheduler, webhook)
# Node 18+ (React dashboards)
# Julia 1.9+ (statistical tooling)
# R 4.2+ (HMM, WFA, regime models)
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

## Tools & Primitives

All executable tools, engines, and core primitives flagged by language and role.

### 🐍 Python — Core Trading

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
| `crypto_backtest_mc.py` | `tools/` | BH backtest + 10K-path Monte Carlo | **BACKTEST ENGINE** |
| `backtest_wave4.py` | `tools/` | Wave 4: EventCalendar + Granger + ML signal | **BACKTEST ENGINE** |
| `EventCalendarFilter` | `tools/backtest_wave4.py` | FOMC/unlock event filter (0.5x ±2h) | **PRIMITIVE** |
| `MLSignalModule` | `tools/backtest_wave4.py` | Logistic SGD signal (5-lag returns + GARCH) | **PRIMITIVE** |
| `NetworkSignalTracker` | `tools/backtest_wave4.py` | BTC Granger lead (1.2x when \|corr\|>0.3) | **PRIMITIVE** |
| `run_full_analysis.py` | `/` | Macro + on-chain + alt data + IAE pipeline | **ANALYSIS RUNNER** |
| `run_iae_analysis.py` | `/` | IAE idea miner (63K trades) | **ANALYSIS RUNNER** |

### 🐍 Python — IAE Pipeline

| Tool / Primitive | Path | Role | Flag |
|---|---|---|---|
| `load_backtest()` | `idea-engine/ingestion/loaders/` | Load trade data into IAE | **LOADER** |
| `ingestion/pipeline.py` | `idea-engine/` | 4-stage mine → filter → store | **PIPELINE** |
| `TimeOfDayMiner` | `idea-engine/ingestion/miners/` | Hour-of-day P&L statistical test | **MINER** |
| `RegimeClusterMiner` | `idea-engine/ingestion/miners/` | Cluster-based regime patterns | **MINER** |
| `BHPhysicsMiner` | `idea-engine/ingestion/miners/` | BH state pattern extraction | **MINER** |
| `DrawdownMiner` | `idea-engine/ingestion/miners/` | Drawdown event patterns | **MINER** |
| `BootstrapFilter` | `idea-engine/ingestion/statistical_filters/` | BH FDR correction | **FILTER** |
| `OnChainEngine` | `idea-engine/onchain/` | BTC on-chain composite signal | **SIGNAL ENGINE** |
| `RegimeClassifier` | `idea-engine/macro-factor/` | 6-class macro regime oracle | **CLASSIFIER** |
| `FearGreedClient` | `idea-engine/sentiment-engine/` | Fear & Greed index | **DATA CLIENT** |
| `FuturesOIFetcher` | `idea-engine/alternative-data/` | Open interest fetcher | **DATA CLIENT** |
| `FundingRateFetcher` | `idea-engine/alternative-data/` | Funding rate fetcher | **DATA CLIENT** |

### 🐍 Python — Execution Infrastructure

| Tool / Primitive | Path | Role | Flag |
|---|---|---|---|
| `OrderBook` | `execution/orderbook/orderbook.py` | Thread-safe L2 book with VWAP fill | **PRIMITIVE** |
| `AlpacaL2Feed` | `execution/orderbook/alpaca_l2_feed.py` | Alpaca WSS L2 feed | **FEED** |
| `BinanceL2Feed` | `execution/orderbook/binance_l2_feed.py` | Binance @depth10 fallback feed | **FEED** |
| `BookManager` | `execution/orderbook/book_manager.py` | Dual-feed + 30s failover | **INFRASTRUCTURE** |
| `FeedMonitor` | `execution/orderbook/feed_monitor.py` | 60s spread sampling → JSONL | **MONITOR** |
| `SmartRouter` | `execution/routing/smart_router.py` | Spread-tier order routing | **ROUTER** |
| `supervisor.py` | `scripts/` | HTTP :8790 process supervisor | **INFRASTRUCTURE** |
| `start_all.sh` | `scripts/` | 5-service startup with health checks | **INFRASTRUCTURE** |
| `trade_logger.py` | `infra/observability/` | SQLite WAL trade logger | **LOGGER** |

### 🦀 Rust — Performance Primitives

| Tool / Primitive | Crate | Role | Flag |
|---|---|---|---|
| `genome-engine` | `idea-engine/rust/` | NSGA-II multi-objective genome evolution | **OPTIMIZER** |
| `counterfactual-oracle` | `idea-engine/rust/` | Counterfactual scenario runner | **ORACLE** |
| `tick-backtest` | `crates/tick-backtest/` | Tick-level BH backtest (rayon parallel) | **BACKTEST ENGINE** |
| `larsa-core` | `crates/larsa-core/` | Core BH engine (Rust port) | **PRIMITIVE** |
| `portfolio-engine` | `crates/portfolio-engine/` | Ledoit-Wolf, HRP, Black-Litterman | **OPTIMIZER** |
| `risk-engine` | `crates/risk-engine/` | VaR/CVaR, Greeks, stress scenarios | **RISK** |

### 🐹 Go — Microservices

| Service | Port | Role | Flag |
|---|---|---|---|
| `cmd/api/main.go` | `:8767` | REST API: hypotheses, genomes, signals | **API** |
| `cmd/bus/main.go` | `:8768` | Internal pub/sub event bus | **MESSAGE BUS** |
| `cmd/scheduler/main.go` | `:8769` | Cron task orchestration | **SCHEDULER** |
| `cmd/webhook/main.go` | `:8770` | Alpaca webhooks + external alerts | **WEBHOOK** |
| `infra/research-api/main.go` | `:8766` | Research data API | **API** |

### 🔷 Julia — Statistical Primitives

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
| `CryptoMechanics.jl` | `julia/src/` | Basis, funding arb, cross-exchange spread | **CRYPTO** |
| `ExecutionAnalytics.jl` | `julia/src/` | VWAP/TWAP, Almgren-Chriss, TCA | **EXECUTION** |
| `volatility_surface.jl` | `idea-engine/stats-service/julia/` | SVI, SABR, Dupire local vol, variance swaps | **VOLATILITY** |
| `alpha_research.jl` | `idea-engine/stats-service/julia/` | IC/ICIR pipeline, factor decay, quintile bt | **ALPHA** |
| `stochastic_control.jl` | `idea-engine/stats-service/julia/` | HJB PDE, optimal control | **CONTROL** |
| `copula_models.jl` | `idea-engine/stats-service/julia/` | Gaussian/Clayton/Gumbel copulas, tail dep | **DEPENDENCE** |
| `information_theory.jl` | `idea-engine/stats-service/julia/` | Entropy, mutual info, transfer entropy | **INFORMATION** |
| `machine_learning_advanced.jl` | `idea-engine/stats-service/julia/` | GBM, neural net from scratch, CV, SHAP | **ML** |
| `reinforcement_learning.jl` | `idea-engine/stats-service/julia/` | Q-learning, policy gradient, DQN | **RL** |
| `jump_processes.jl` | `idea-engine/stats-service/julia/` | Lévy processes, compound Poisson, VG model | **STOCHASTIC** |

*40 Julia source modules + 34 notebooks. See [Statistical Tooling](docs/statistical_tooling.md) for full list.*

### 📊 R — Statistical Analysis

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
| `defi_analytics.R` | `idea-engine/stats-service/r/` | AMM pricing, IL, V3 liquidity, MEV | **CRYPTO** |
| `portfolio_attribution.R` | `idea-engine/stats-service/r/` | BHB, Brinson-Fachler, factor attribution | **ATTRIBUTION** |
| `walk_forward_analysis.R` | `idea-engine/stats-service/r/` | CPCV walk-forward, Sobol optimization | **VALIDATION** |
| `numerical_methods.R` | `idea-engine/stats-service/r/` | MC variance reduction, PDE solvers, Adam | **NUMERICAL** |

*29 R modules across 3 directories + 12 research scripts. See [Statistical Tooling](docs/statistical_tooling.md) for full list.*

### ⚡ C/C++ — Low-Latency Primitives

| Component | Path | Role | Flag |
|---|---|---|---|
| Fast indicators (20 signals) | `lib/` | SIMD-accelerated signal computation | **PRIMITIVE** |
| L3 orderbook | `lib/` | AVX2 price ladder, sub-microsecond updates | **PRIMITIVE** |
| Matrix ops | `lib/` | SIMD matrix multiply for portfolio math | **PRIMITIVE** |

### 🦎 Zig — Ultra Low Latency

| Component | Path | Role | Flag |
|---|---|---|---|
| ITCH 5.0 decoder | `zig/` | NASDAQ ITCH protocol parser | **DECODER** |
| Low-latency orderbook | `zig/` | Lock-free L2 book | **PRIMITIVE** |

### 🐳 Infrastructure

| Component | Description | Flag |
|---|---|---|
| `docker-compose.yml` | 5-service deployment (trader/monitor/supervisor/orderbook/bridge) | **DEPLOY** |
| `Dockerfile.python` | 2-stage build, non-root srfm user | **DEPLOY** |
| `scripts/supervisor.py` | HTTP :8790 process supervisor, exponential backoff restart | **INFRASTRUCTURE** |
| `scripts/start_all.sh` | Health-check loop, auto-restart, stop/restart/status | **INFRASTRUCTURE** |
| `config/signal_overrides.json` | Hot-reloaded per-symbol + global multipliers | **CONFIG** |
| `config/instruments.yaml` | 30+ instruments with CF calibration | **CONFIG** |
| `.github/workflows/` | CI/CD for Python/Rust/Go/TS/Julia | **CI/CD** |

---

## Stack

| Language | LOC | Key Systems | Docs |
|---|---|---|---|
| Python | ~235K | Live trader, backtesting, IAE pipeline, ML, research | [Execution Stack](docs/execution_stack.md) |
| Julia | ~100K | Statistical tooling: copulas, SVI/SABR, Kalman, AMM, CoVaR, RL | [Statistical Tooling](docs/statistical_tooling.md) |
| TypeScript/React | ~42K | IAE dashboard (:5175), research dashboard (:5174), trading terminal | [Stack Overview](docs/stack_overview.md) |
| Go | ~38K | IAE microservices (API/bus/scheduler/webhook), research API | [IAE Architecture](docs/iae_architecture.md) |
| R | ~45K | HMM, regime models, WFA, factor analysis, stress testing | [Statistical Tooling](docs/statistical_tooling.md) |
| Rust | ~24K | Genome engine (NSGA-II), Monte Carlo, tick backtest, portfolio/risk | [IAE Architecture](docs/iae_architecture.md) |
| C/C++ | ~15K | 20 fast indicators (SIMD), L3 orderbook (AVX2), matrix ops | [Stack Overview](docs/stack_overview.md) |
| Zig | ~8K | ITCH 5.0 decoder, lock-free orderbook | [Stack Overview](docs/stack_overview.md) |
| SQL | ~5K | SQLite (16 migrations, WAL), DuckDB analytics, BH UDFs | [Stack Overview](docs/stack_overview.md) |
| **Total** | **~560K** | | |

---

## File Structure

```
srfm-lab/
│
├── tools/                               # ★ Core trading tools
│   ├── live_trader_alpaca.py            # ★ LIVE TRADER: BH Physics + GARCH + OU + IAE params
│   ├── crypto_backtest_mc.py            # ★ BACKTEST: BH + 10K-path Monte Carlo
│   ├── backtest_wave4.py                # ★ WAVE 4: EventCalendar + Granger + ML signal
│   ├── walk_forward_engine.py           # Walk-forward analysis
│   ├── factor_analysis.py               # Fama-MacBeth, IC/ICIR, factor decay
│   └── backtest_output/                 # SQLite DBs, CSV trade logs, PNG charts
│
├── idea-engine/                         # ★ IAE (Idea Automation Engine)
│   ├── db/                              # Schema migrations (SQLite WAL)
│   ├── ingestion/                       # 4-stage pipeline: load → mine → filter → store
│   │   ├── loaders/                     # Backtest, live, walk-forward loaders
│   │   ├── miners/                      # TimeOfDay, RegimeCluster, BHPhysics, Drawdown
│   │   └── statistical_filters/         # Bootstrap filter (BH FDR correction)
│   ├── genome/                          # Rust NSGA-II genome evolution
│   ├── hypothesis/                      # Bayesian hypothesis generator
│   ├── causal/                          # Granger + PC algorithm causal discovery
│   ├── walk_forward/                    # CPCV walk-forward engine
│   ├── regime/                          # 6-regime oracle
│   ├── signals/                         # 60+ signal library with IC tracking
│   ├── macro-factor/                    # VIX, DXY, yield curve, equity momentum
│   ├── onchain/                         # BTC on-chain: MVRV, SOPR, exchange reserves
│   ├── alternative-data/               # Futures OI, funding rates, liquidations
│   ├── sentiment-engine/               # Fear & Greed, NLP scrapers
│   ├── shadow/                          # Shadow strategy runner
│   ├── counterfactual/                  # Rust counterfactual oracle
│   ├── academic/                        # arXiv + SSRN miner
│   ├── serendipity/                     # Domain analogy + mutation engine
│   ├── stats-service/
│   │   ├── julia/                       # 40 Julia modules (copulas, SVI, Kalman, AMM...)
│   │   └── r/                           # 29 R modules (HMM, WFA, regime, GARCH-DCC...)
│   ├── cmd/                             # Go: API :8767, bus :8768, scheduler :8769, webhook :8770
│   ├── idea-dashboard/                  # React/TS + Vite + Recharts + D3 (:5175)
│   └── idea_engine.db                   # SQLite: patterns, hypotheses, experiments, genomes
│
├── execution/                           # ★ Execution infrastructure
│   ├── orderbook/
│   │   ├── orderbook.py                 # Thread-safe L2 book, VWAP-to-fill
│   │   ├── alpaca_l2_feed.py            # Alpaca WSS (msgs 'o'+'q'), backoff
│   │   ├── binance_l2_feed.py           # @depth10@100ms fallback
│   │   ├── book_manager.py              # Dual-feed + 30s failover
│   │   └── feed_monitor.py             # 60s sampling → orderbook_metrics.jsonl
│   └── routing/
│       └── smart_router.py              # Spread-tier routing (≤50/50-100/>100 bps)
│
├── scripts/
│   ├── supervisor.py                    # HTTP :8790 supervisor, exponential backoff
│   └── start_all.sh                     # 5-service startup with auto-restart
│
├── julia/
│   ├── src/                             # 40 Julia modules (full stat tooling)
│   │   ├── BHPhysics.jl                 # BH engine
│   │   ├── Stochastic.jl                # GARCH, Heston, Hawkes, OU, Merton JD
│   │   ├── SystemicRisk.jl              # CoVaR, MES, SRISK, Eisenberg-Noe
│   │   ├── NumericalMethods.jl          # PDE solvers, Sobol MC, quadrature
│   │   ├── CryptoDefi.jl               # AMM, Uniswap v3, impermanent loss
│   │   └── ... (35 more modules)
│   └── notebooks/                       # 34 research notebooks (01-34)
│
├── r/
│   ├── R/                               # 21 R modules
│   │   ├── bh_analysis.R               # BH state reconstruction
│   │   ├── regime_models.R             # HMM, Markov switching, GARCH-DCC
│   │   ├── systemic_risk.R             # CoVaR, SRISK, DebtRank
│   │   └── ... (18 more modules)
│   └── research/                        # 12 research scripts
│
├── infra/
│   ├── observability/trade_logger.py    # SQLite WAL trade logger
│   ├── research-api/                    # Go research API (:8766)
│   ├── gateway/                         # Market data gateway (17 indicators)
│   ├── grpc/                            # gRPC microservices
│   └── event-bus/                       # Redis pub/sub
│
├── research/
│   ├── live_monitor/                    # ★ Terminal CLI + web dashboard
│   ├── reconciliation/                  # Live vs backtest recon
│   ├── walk_forward/                    # CPCV + Sobol/Bayesian param opt
│   ├── regime_lab/                      # HMM, PELT, 20 stress scenarios
│   ├── signal_analytics/               # IC/ICIR, alpha decay, quintile analysis
│   └── dashboard/                       # React research dashboard (:5174)
│
├── spacetime/
│   ├── engine/bh_engine.py             # Universal BH backtester (all assets)
│   ├── api/main.py                      # FastAPI (:8765), 15 routes + WebSocket
│   └── web/                             # React Spacetime Arena (:5173)
│
├── lib/
│   ├── srfm_core.py                    # BHState, MinkowskiClassifier, HawkingMonitor
│   ├── agents.py                        # D3QN, DDQN, TD3QN ensemble agents
│   ├── regime.py                        # Regime detector
│   └── risk.py                          # Risk management, stops, circuit breakers
│
├── crates/
│   ├── tick-backtest/                   # Tick-level BH (rayon parallel sweeps)
│   ├── larsa-core/                      # Core BH engine in Rust
│   ├── portfolio-engine/               # Ledoit-Wolf, HRP, Black-Litterman
│   └── risk-engine/                     # VaR/CVaR, Greeks, stress scenarios
│
├── config/
│   ├── instruments.yaml                 # 30+ instruments with CF calibration
│   ├── risk_limits.yaml
│   └── signal_overrides.json           # Hot-reloaded per-symbol multipliers
│
├── docs/                                # ★ Deep documentation
│   ├── bh_physics.md                   # BH engine: Minkowski, mass, Hawking, delta
│   ├── iae_architecture.md             # IAE: genome, hypothesis, causal, regime
│   ├── execution_stack.md              # L2 orderbook, smart router, supervisor
│   ├── wave4_backtest.md               # EventCalendar, Granger lead, ML signal
│   ├── statistical_tooling.md          # All Julia + R modules reference
│   └── stack_overview.md              # Full tech stack with integration diagram
│
├── run_full_analysis.py                # Macro + on-chain + alt data + IAE pipeline
├── run_iae_analysis.py                 # IAE idea miner (63K trades)
├── docker-compose.yml                  # 5-service deployment
├── Dockerfile.python                   # 2-stage Python build
├── Makefile                            # 60+ targets
└── .github/workflows/                  # CI/CD: Python/Rust/Go/TS/Julia
```

---

## Service Endpoints

| Service | Command | Port | Flag |
|---|---|---|---|
| Live Trader | `python tools/live_trader_alpaca.py` | — | **LIVE** |
| Process Supervisor | `python scripts/supervisor.py` | `:8790` | **INFRA** |
| IAE API | `cd idea-engine && go run cmd/api/main.go` | `:8767` | **API** |
| IAE Event Bus | `go run cmd/bus/main.go` | `:8768` | **MESSAGE BUS** |
| IAE Scheduler | `go run cmd/scheduler/main.go` | `:8769` | **SCHEDULER** |
| IAE Webhook | `go run cmd/webhook/main.go` | `:8770` | **WEBHOOK** |
| IAE Dashboard | `cd idea-engine/idea-dashboard && npm run dev` | `:5175` | **UI** |
| Research API | `cd infra/research-api && go run main.go` | `:8766` | **API** |
| Spacetime API | `python run_api.py` | `:8765` | **API** |
| Spacetime Web | `cd spacetime/web && npm run dev` | `:5173` | **UI** |
| Research Dashboard | `cd research/dashboard && npm run dev` | `:5174` | **UI** |
| Live Monitor (CLI) | `python -m research.live_monitor.cli monitor run` | — | **MONITOR** |

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
```

### IAE

```bash
python -m idea_engine.db.migrate
python -m idea_engine.ingestion.pipeline --verbose
python -m idea_engine.ingestion.pipeline --miners time_of_day,bh_physics --dry-run
cd idea-engine/rust && cargo build --release
cd idea-engine/rust && ./target/release/genome-engine --generations 100 --pop-size 200
```

### Julia Statistical Tooling

```bash
julia julia/src/SystemicRisk.jl
julia julia/src/CryptoDefi.jl
julia idea-engine/stats-service/julia/volatility_surface.jl
julia idea-engine/stats-service/julia/alpha_research.jl
julia julia/notebooks/27_volatility_surface_analysis.jl
```

### R Statistical Analysis

```bash
Rscript r/R/regime_models.R
Rscript r/R/systemic_risk.R
Rscript idea-engine/stats-service/r/volatility_surface.R
Rscript r/research/ml_backtesting.R
Rscript r/research/cross_asset_study.R
```

### Build / Test

```bash
pytest tests/ -v
cargo test --workspace
cd idea-engine && go test ./...
cd idea-engine/idea-dashboard && npm test
```

---

## BH Physics Reference

→ **[Full deep dive with worked example](docs/bh_physics.md)**

| Component | Formula | Interpretation |
|---|---|---|
| MinkowskiClassifier | `ds² = c²dt² − dx²` | TIMELIKE (ds²>0) = ordered, causal; SPACELIKE = anomalous |
| BH Formation | `mass >= BH_FORM (1.92)` | Gravitational well forms; EMA asymptotes to 2.0 |
| Mass Accrual | `mass = 0.97×mass + 0.03×min(2, 1+ctl×0.1)` | Consecutive timelike bars build conviction |
| Mass Decay | `mass *= BH_DECAY (0.924)` | Noise bars bleed mass away |
| Hawking Monitor | `T_H = 1/(8πM)` | Cold well = stable signal; hot well = reduce size |
| Delta Score | `tf_score × mass × ATR` | Expected dollar move → allocation signal |
| OU Overlay | `dX = θ(μ−X)dt + σdW` | Mean reversion on flat BH; 8% equity |
| Mayer Dampener | `scale = min(1, 2×MA200/price)` | Reduces size when price is extended |
| BTC Lead | `alt_score *= (1 + btc_active × 0.3)` | BTC activation boosts correlated alts |
| Dynamic CORR | `0.25 base → 0.60 when 30d pair-corr > 0.60` | Stress regime portfolio risk reduction |

---

## Key Parameters

| Parameter | Default | IAE Tuned | Effect |
|---|---|---|---|
| `BH_FORM` | 1.92 | — | Mass threshold for BH activation |
| `BH_DECAY` | 0.924 | — | Mass bleed rate on noise bars |
| `CF` (per instrument) | 0.001–0.025 | — | Minkowski speed of light |
| `CORR` | dynamic | dynamic | Cross-asset correlation (0.25/0.60) |
| `GARCH_TARGET_VOL` | 1.20 | **0.90** | Target annualized volatility |
| `OU_FRAC` | 0.08 | — | OU mean-reversion allocation |
| `MIN_HOLD` | 4 | **8** | Minimum bars before exit |
| `BLOCKED_ENTRY_HOURS` | {} | **{1,13,14,15,17,18}** | UTC hours blocked for entries |
| `BOOST_HOURS` | {} | **{3,9,16,19}** | UTC hours with 1.25x size boost |
| `WINNER_PROTECTION_PCT` | 0.001 | **0.005** | Threshold to let winners run |
| `OU_DISABLED_SYMBOLS` | {} | **{AVAX,DOT,LINK}** | Momentum symbols, skip OU |
| `DELTA_MAX_FRAC` | 0.40 | — | Max single-instrument allocation |
| `MIN_TRADE_FRAC` | 0.03 | — | Minimum equity shift to rebalance |

*IAE Tuned = parameter updated by IAE analysis of 63,993 backtest trades.*

---

## IAE Live Research Output

The IAE ingested 63,993 backtest trades (Jan 2024 – Apr 2026) and produced 10 actionable ideas. All 9 high-confidence ideas are now live in `tools/live_trader_alpaca.py`.

```
#1 [91%] EXIT RULE: Raise min_hold_bars 4 → 8
   1-bar holds: avg P&L=-169, WR=35.5% (16,939 trades = 26.5% of all trades)
   5-12 bar holds: avg P&L=+111, WR=46.4%. Eliminating fast exits is the
   single highest-leverage change.

#2 [88%] ENTRY TIMING: Block entries at hours 1, 13, 14, 15, 18 UTC
   These hours: avg P&L=-131/trade vs -10 baseline. WR drops to 37%.
   Hour 1 UTC worst (-179/trade, 33.2% WR) — thin Asian/European overlap.

#3 [85%] CROSS-ASSET: BTC as signal only, reduce BTC direct trade
   BTC is the worst P&L instrument (-156K) but the lead signal for alts.
   Cut BTC cf_scale to 0.5, boost alt allocation 1.4x when BTC-lead fires.

#4 [82%] INSTRUMENT FILTER: Remove GRT + SOL, shrink AVAX/DOT/LINK
   5 symbols = -390K combined loss. GRT (37.4% WR), SOL (36.4% WR).

#5 [80%] POSITION SIZING: GARCH target_vol 120% → 90%
   Overtrading in high-vol regimes. Tightening GARCH cuts ~25% of trades.

#6 [78%] EXIT RULE: Winner protection 0.1% → 0.5%
   48+ bar trades avg +610/trade (only 419 trades). Cutting winners too early.
```

Backtest comparison after applying all 6 ideas:

| Metric | Baseline | After IAE |
|---|---|---|
| Trades | 63,993 | 59,326 (-7%) |
| Win rate | 41.4% | 43.0% (+1.6pp) |
| MC median 12m | ~$678K | $1.72M |
| MC blowup rate | — | 0% |

→ **[IAE architecture deep dive](docs/iae_architecture.md)**

---

## Performance Notes

Key findings (2021–2026, 19 crypto pairs):
- **LARSA v1 (ES futures):** +274% over backtest window
- **2024 standalone:** +26%, driven by BTC and SOL regime
- **Full period CAGR:** -11% (crypto bear market dominated)
- **Monte Carlo (10,000 paths):** Median outcome captures distribution of sequential trade ordering; blowup rate: 0% after IAE tuning
- **Wave 4 additions:** EventCalendarFilter + Granger lead + ML signal show further improvement in OOS Sharpe

The backtest engine runs identical BH physics to live trading — GARCH vol scaling, OU overlay, Mayer dampening — no lookahead, no future data.

→ **[Wave 4 backtest deep dive](docs/wave4_backtest.md)**
