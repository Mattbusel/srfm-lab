# Technology Stack Overview

Complete reference for every language, runtime, and infrastructure component in the srfm-lab codebase. Covers what each component does, where it lives, how to run it, and its key dependencies.

---

## Approximate Line Count by Language

| Language | LOC | Role |
|---|---|---|
| Python | ~235K | Live trading, IAE pipeline, research |
| Julia | ~100K | Statistical tooling, financial math |
| R | ~45K | Statistical tests, HMM, regime analysis |
| TypeScript/React | ~42K | Dashboards, visualization UIs |
| Go | ~38K | IAE microservices, research API |
| Rust | ~24K | Genome engine, Monte Carlo, tick backtest |
| C/C++ | ~15K | SIMD indicators, L3 orderbook, matrix ops |
| Zig | ~8K | ITCH 5.0 decoder, low-latency orderbook |
| SQL | ~5K | Schema migrations, DuckDB analytics |

---

## Python (~235K LOC)

### What it does
Python is the integration layer: live trading execution, end-to-end backtesting, the Idea Acceleration Engine (IAE) pipeline, research notebooks, and data ingestion. It orchestrates all other components via subprocess, HTTP, and shared SQLite/DuckDB databases.

### Where it lives
```
live_trader_alpaca.py          # top-level live trading entry point
crypto_backtest_mc.py          # Monte Carlo crypto backtest runner
run_api.py                     # research API launcher
run_full_analysis.py           # full research analysis pipeline
run_iae_analysis.py            # IAE pipeline runner
idea-engine/                   # IAE microservices (Python)
research/                      # research modules and notebooks
strategies/                    # strategy definitions
execution/                     # execution layer
infra/                         # infrastructure helpers
```

### How to run
```bash
# Live trading
python live_trader_alpaca.py --config config/live.toml

# Monte Carlo backtest
python crypto_backtest_mc.py --symbol BTC-USD --n-paths 10000

# Full research pipeline
python run_full_analysis.py

# IAE pipeline
python run_iae_analysis.py
```

### Key dependencies
- `alpaca-py` -- Alpaca Markets REST + streaming WebSocket
- `pandas` / `numpy` -- data frames and numerical arrays
- `scipy` / `statsmodels` -- statistical tests, time series models
- `scikit-learn` -- ML preprocessing and models
- `sqlalchemy` -- ORM for SQLite (warehouse)
- `duckdb` -- analytical queries over parquet/CSV
- `httpx` / `aiohttp` -- async HTTP for IAE inter-service calls
- `pyproject.toml` -- package metadata; `requirements.txt` -- pinned deps

---

## Go (~38K LOC)

### What it does
Go implements the four IAE microservices and the standalone research API. All services communicate over the internal event bus (pub/sub topics) and expose HTTP endpoints. The research API at `:8766` is separate from the IAE cluster and serves the research dashboard.

### Where it lives
```
idea-engine/idea-api/          # IAE API gateway        :8767
idea-engine/bus/               # Event bus              :8768
idea-engine/scheduler/         # Experiment scheduler   :8769
idea-engine/webhook-service/   # Webhook dispatcher     :8770
infra/research-api/            # Research API           :8766
```

### How to run
```bash
# Individual service
cd idea-engine/idea-api && go run main.go
cd idea-engine/bus && go run main.go
cd idea-engine/scheduler && go run main.go
cd idea-engine/webhook-service && go run main.go
cd infra/research-api && go run main.go

# Via docker-compose (all services)
docker-compose up iae-api iae-bus iae-scheduler iae-webhook
```

### Service responsibilities
- **idea-api (:8767)** -- REST + WebSocket gateway; routes experiment requests, streams live results, manages the idea database (`idea_engine.db`)
- **bus (:8768)** -- pub/sub event router with topic persistence; all IAE services subscribe to relevant topics
- **scheduler (:8769)** -- cron-based experiment lifecycle management, budget enforcement, dispatcher coordination
- **webhook-service (:8770)** -- outbound webhook delivery with retry, fan-out to external endpoints
- **research-api (:8766)** -- serves research dashboard with regime data, signal analytics, portfolio metrics

### Key dependencies
- Standard library only for most services; `go.mod` in each service directory
- SQLite via `mattn/go-sqlite3` (idea-api)

---

## Rust (~24K LOC)

### What it does
Rust handles the most performance-sensitive compute workloads: multi-objective genetic optimization (NSGA-II), Monte Carlo simulation, a counterfactual oracle for hypothesis testing, tick-level backtesting, and portfolio/risk crates used as libraries.

### Where it lives
```
crates/
  idea-genome-engine/    # NSGA-II genetic optimizer
  monte-carlo-engine/    # Monte Carlo simulation engine
  counterfactual-engine/ # Counterfactual oracle
  tick-backtest/         # Tick-level backtester
  portfolio-engine/      # Portfolio construction
  risk-engine/           # Risk metrics
  larsa-core/            # LARSA strategy core
  larsa-wasm/            # WASM build of larsa-core
  signal-evolution/      # Signal parameter evolution
  smart-order-router/    # SOR routing logic
  order-flow-engine/     # Order flow analytics
  options-engine/        # Options pricing
  regime-detector/       # Regime classification
  rl-exit-optimizer/     # RL-based exit timing
  parallel-backtest/     # Parallel strategy evaluation
  fractal-engine/        # Fractal/multi-scale analysis
  fix-engine/            # FIX protocol engine
  network-graph/         # Network risk graph
  orderbook-sim/         # Limit orderbook simulation
  srfm-tools/            # Shared utilities
Cargo.toml               # workspace manifest
Cargo.lock
```

### How to run
```bash
# Build all crates
cargo build --release --workspace

# Run genome optimizer
cargo run -p idea-genome-engine --release -- --config config/genome.toml

# Run tick backtest
cargo run -p tick-backtest --release -- --data data/ticks/BTC.csv

# Run Monte Carlo engine
cargo run -p monte-carlo-engine --release -- --n-paths 100000
```

### Key dependencies
- `rayon` -- data-parallel iterators (genome, MC, parallel backtest)
- `serde` / `serde_json` -- serialization
- `tokio` -- async runtime (fix-engine, order-flow-engine)
- `ndarray` -- N-dimensional arrays (portfolio-engine, risk-engine)
- `wasm-bindgen` -- WASM FFI (larsa-wasm)

---

## TypeScript / React (~42K LOC)

### What it does
Three separate Vite/React single-page applications providing interactive research dashboards, IAE monitoring, and the spacetime visualization interface.

### Where it lives
```
idea-engine/idea-dashboard/    # IAE dashboard          :5175
dashboard/                     # Research dashboard     :5174
spacetime/                     # Spacetime web UI       :5173
```

Each directory contains a standard Vite project structure: `src/`, `index.html`, `package.json`, `vite.config.ts`, `tailwind.config.js`.

### How to run
```bash
# IAE dashboard
cd idea-engine/idea-dashboard && npm install && npm run dev  # :5175

# Research dashboard
cd dashboard && npm install && npm run dev                   # :5174

# Spacetime
cd spacetime && npm install && npm run dev                   # :5173

# Production build (all)
npm run build  # in each directory
```

### Key libraries
- **Vite** -- build tooling and dev server
- **React 18** -- UI framework
- **Recharts** -- time series charting (equity curves, P&L)
- **D3** -- custom visualizations (network graphs, volatility surfaces)
- **Tailwind CSS** -- utility-first styling
- **TypeScript** -- type safety across all three apps

### Dashboard capabilities
- **IAE dashboard** -- experiment tracking, genome evolution visualization, live idea scoring, walk-forward results
- **Research dashboard** -- regime overlays, signal analytics, factor heatmaps, portfolio risk attribution
- **Spacetime** -- multi-timeframe pattern visualization, BH confluence surface, cross-asset correlation matrices

---

## Julia (~100K LOC)

### What it does
Julia provides all performance-critical financial mathematics: stochastic process simulation, volatility surface calibration, Bayesian inference, alpha signal pipeline, and market microstructure estimation. Runs as a long-lived service (stats service) to avoid JIT compilation overhead on each request.

### Where it lives
```
julia/src/                              # 42 production modules
idea-engine/stats-service/julia/        # service-layer modules with HTTP routes
idea-engine/stats-service/julia/routes/ # HTTP route handlers
```

See `docs/statistical_tooling.md` for complete module reference.

### How to run
```bash
# Start stats service (keeps Julia warm)
cd idea-engine/stats-service && julia --project -e "include(\"server.jl\")"

# Run a specific module interactively
julia --project=julia julia/src/BHPhysics.jl

# Run tests
cd julia && julia --project -e "using Pkg; Pkg.test()"
```

### Key dependencies (Julia packages)
- `Distributions.jl` -- probability distributions
- `StatsBase.jl` -- statistical functions
- `Turing.jl` -- probabilistic programming / MCMC
- `DifferentialEquations.jl` -- SDE simulation
- `Optim.jl` -- numerical optimization
- `HTTP.jl` -- service HTTP server
- `DataFrames.jl` -- tabular data

---

## R (~45K LOC)

### What it does
R provides statistical testing infrastructure, HMM regime modeling, walk-forward validation, White's Reality Check, and visualization for research reports. Complements Julia for workloads where CRAN package ecosystems (rugarch, depmixS4, strucchange) are more mature than Julia equivalents.

### Where it lives
```
r/R/                              # 25+ production modules
r/research/                       # 15 standalone research scripts
idea-engine/stats-service/r/      # service-layer modules + routes/
idea-engine/stats-service/r/utils/
```

See `docs/statistical_tooling.md` for complete module reference.

### How to run
```bash
# Source a module
Rscript -e "source('r/R/regime_models.R'); fit_hmm(returns, n_states=3)"

# Run a research script
Rscript r/research/regime_trading_study.R

# Start R stats service endpoint
Rscript idea-engine/stats-service/r/server.R
```

### Key dependencies (R packages)
- `rugarch` -- GARCH family estimation
- `depmixS4` -- HMM regime models
- `strucchange` -- structural break tests
- `vars`, `urca` -- VAR models, cointegration
- `copula` -- copula estimation
- `CVXR` -- convex optimization
- `xgboost` -- gradient boosting
- `ggplot2` -- visualization
- `highfrequency` -- realized variance estimation

---

## C / C++ (~15K LOC)

### What it does
C++ provides the lowest-latency native components: 20 SIMD-accelerated technical indicators, an AVX2-optimized Level 3 orderbook, and matrix factorization routines used by the portfolio engine.

### Where it lives
```
native/orderbook/          # L3 orderbook (AVX2)
  orderbook.hpp
  feed_handler.cpp
  backtester.hpp
  backtest_main.cpp
native/matrix/             # Matrix operations
  matrix.hpp
  decomposition.cpp
  factor_model.cpp
cpp/signal-engine/         # 20 SIMD indicators
  include/
  src/
  tests/
  benchmarks/
```

### How to run
```bash
# Build orderbook
cd native/orderbook && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# Build signal engine
cd cpp/signal-engine && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# Run orderbook backtest
./native/orderbook/build/backtest_main --data data/ticks/BTC.csv

# Run indicator benchmarks
./cpp/signal-engine/build/benchmarks/bench_indicators
```

### Key features
- 20 indicators in `cpp/signal-engine/src/` with SIMD (SSE4.2 / AVX2) vectorization
- AVX2 orderbook in `native/orderbook/` processes millions of updates/second
- Matrix ops in `native/matrix/` expose a C API consumed by Rust FFI (`larsa-core`)

---

## Zig (~8K LOC)

### What it does
Zig implements a high-performance NASDAQ ITCH 5.0 binary protocol decoder and a low-latency limit orderbook designed for sub-microsecond message processing.

### Where it lives
```
native/zig/
  src/
    main.zig          # entry point
    decoder.zig       # ITCH 5.0 message decoder
    orderbook.zig     # limit orderbook
    protocol.zig      # protocol types
    feed.zig          # network feed handler
    market_maker.zig  # MM simulation
    simulation.zig    # simulation harness
    risk.zig          # real-time risk checks
    stats.zig         # running statistics
    allocator.zig     # custom arena allocator
    network.zig       # UDP/TCP feed ingestion
    writer.zig        # binary output writer
    bench.zig         # benchmarks
  build.zig
```

### How to run
```bash
cd native/zig
zig build                          # debug build
zig build -Doptimize=ReleaseFast   # optimized build
./zig-out/bin/srfm-itch --feed data/itch/sample.bin
zig build test                     # run tests
zig build bench                    # run benchmarks
```

---

## SQL (~5K LOC)

### What it does
SQL defines the persistent storage schema: 16 sequential SQLite migrations for the main warehouse, DuckDB analytical queries for batch research, and BH-specific user-defined function extensions.

### Where it lives
```
warehouse/migrations/              # 16 SQL migration files
  001_initial.sql
  002_add_regime_periods_timeframe.sql
  003_add_trade_exit_reason.sql
  004_add_mc_simulations.sql
  005_add_instrument_cf_columns.sql
  006_add_funding_rates.sql
  007_add_strategy_run_tags.sql
  008_add_bh_confluence.sql
  009_add_iv_surface.sql
  010_add_risk_events.sql
  011_add_reconciliation.sql
  012_add_walkforward.sql
  013_add_signal_analytics.sql
  014_add_regime_lab.sql
  015_add_agent_training.sql
  016_add_portfolio_lab.sql
warehouse/duckdb/                  # DuckDB analytical queries
warehouse/schema/                  # schema documentation
```

### How to run
```bash
# Apply all migrations (handled by Python startup)
python -c "from warehouse.db import apply_migrations; apply_migrations()"

# DuckDB research query
duckdb -c "SELECT * FROM read_parquet('data/ohlcv/*.parquet') LIMIT 10"
```

---

## Infrastructure

### Docker Compose

The `docker-compose.yml` at the repo root defines 5 services and can bring up the full stack.

```bash
docker-compose up          # all services
docker-compose up iae-api iae-bus iae-scheduler iae-webhook stats-service
docker-compose build       # rebuild images
```

Services use `Dockerfile.python` (Python services) and per-service Dockerfiles in each Go/Julia service directory.

### Supervisor (`supervisor.py` on :8790)

A lightweight Python process supervisor manages non-containerized local development runs. Exposes a control API at `:8790`.

```bash
python infra/supervisor.py          # start supervisor
curl http://localhost:8790/status   # process status
curl -X POST http://localhost:8790/restart/stats-service
```

### CI/CD (`.github/workflows/`)

Three workflow files cover the full stack:
- `ci.yml` -- runs on every push: Python tests (pytest), Rust tests (cargo test), Go tests (go test ./...), TypeScript lint+build (npm run build), Julia tests
- `backtest.yml` -- runs backtests on PRs targeting main; posts Sharpe/drawdown summary as PR comment
- `release.yml` -- builds release binaries (Rust), Docker images, and publishes artifacts on version tags

---

## Component Integration Diagram

```
                         ┌─────────────────────────────────────────────────────┐
                         │                  DATA LAYER                          │
                         │   SQLite (warehouse/)    DuckDB (warehouse/duckdb/) │
                         │   Parquet (data/)        Redis (infra/timeseries/)  │
                         └────────────────┬────────────────────────────────────┘
                                          │
           ┌──────────────────────────────▼──────────────────────────────────┐
           │                    COMPUTE LAYER                                  │
           │                                                                   │
           │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
           │  │  Julia Stats │  │   R Stats    │  │  Rust Crates         │   │
           │  │  Service     │  │  Service     │  │  (genome, MC,        │   │
           │  │  (julia/)    │  │  (r/)        │  │   tick-backtest,     │   │
           │  │              │  │              │  │   portfolio-engine)  │   │
           │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
           │         │                 │                       │               │
           │  ┌──────▼─────────────────▼───────────────────────▼───────────┐  │
           │  │              C++ / Zig Native Layer                         │  │
           │  │   cpp/signal-engine (SIMD indicators)                       │  │
           │  │   native/orderbook (AVX2 L3 orderbook)                      │  │
           │  │   native/zig (ITCH decoder, low-latency orderbook)          │  │
           │  └─────────────────────────────────────────────────────────────┘  │
           └──────────────────────────────┬──────────────────────────────────┘
                                          │
           ┌──────────────────────────────▼──────────────────────────────────┐
           │                    SERVICE LAYER (Go)                             │
           │                                                                   │
           │  ┌──────────────────────────────────────────────────────────┐    │
           │  │              IAE Event Bus  :8768                         │    │
           │  │  (pub/sub backbone -- all services subscribe here)        │    │
           │  └───┬──────────────┬──────────────┬──────────────┬─────────┘    │
           │      │              │              │              │               │
           │  ┌───▼───┐   ┌──────▼───┐   ┌──────▼───┐   ┌────▼────────┐     │
           │  │ IAE   │   │Scheduler │   │Webhook   │   │Research API │     │
           │  │ API   │   │  :8769   │   │  :8770   │   │   :8766     │     │
           │  │ :8767 │   └──────────┘   └──────────┘   └─────────────┘     │
           │  └───────┘                                                        │
           └──────────────────────────────┬──────────────────────────────────┘
                                          │
           ┌──────────────────────────────▼──────────────────────────────────┐
           │               PYTHON APPLICATION LAYER                            │
           │                                                                   │
           │  live_trader_alpaca.py    ──────────────────► Alpaca Markets      │
           │  crypto_backtest_mc.py   ──► Rust MC engine                      │
           │  idea-engine/ (IAE)      ──► Go services (HTTP)                  │
           │  research/               ──► Julia/R stats services (HTTP)       │
           │  strategies/             ──► Rust crates (subprocess/FFI)        │
           └──────────────────────────────┬──────────────────────────────────┘
                                          │
           ┌──────────────────────────────▼──────────────────────────────────┐
           │                  PRESENTATION LAYER (TypeScript)                  │
           │                                                                   │
           │  Research Dashboard :5174  ◄─── Research API :8766               │
           │  IAE Dashboard      :5175  ◄─── IAE API :8767 (WebSocket)        │
           │  Spacetime Web      :5173  ◄─── Research API :8766               │
           └─────────────────────────────────────────────────────────────────┘
```

### Key integration paths

1. **Live trading.** `live_trader_alpaca.py` → Alpaca WebSocket feed → Python strategy layer → Rust `smart-order-router` → Alpaca order API → fills written to SQLite.

2. **IAE pipeline.** `run_iae_analysis.py` → IAE API (:8767) → Bus (:8768) → Scheduler dispatches experiments → Julia/R stats services compute signals → results stored in `idea_engine.db` → IAE dashboard streams live via WebSocket.

3. **Genome optimization.** IAE scheduler → Rust `idea-genome-engine` (NSGA-II) → evaluates population using Rust `parallel-backtest` and `risk-engine` → returns Pareto front → stored in SQLite → visible in IAE dashboard.

4. **Research pipeline.** `run_full_analysis.py` → Python research modules → calls Julia stats service (HTTP) for heavy computation → calls R stats service for HMM/tests → results stored in DuckDB → visualized in research dashboard (:5174).

5. **Tick data path.** Zig ITCH decoder → binary event stream → C++ AVX2 orderbook → Rust `order-flow-engine` → Python strategy layer or Rust `tick-backtest`.

---

## Development Quick Reference

```bash
# Full stack (Docker)
docker-compose up

# Local development (no Docker)
python infra/supervisor.py                          # start supervisor :8790
python run_api.py &                                 # Python research API
cd idea-engine/idea-api && go run main.go &         # IAE API :8767
cd idea-engine/bus && go run main.go &              # Bus :8768
julia --project idea-engine/stats-service/server.jl &  # Julia stats service
Rscript idea-engine/stats-service/r/server.R &     # R stats service
cd dashboard && npm run dev &                       # Research dashboard :5174
cd idea-engine/idea-dashboard && npm run dev &      # IAE dashboard :5175

# Run all tests
pytest tests/                    # Python
cargo test --workspace           # Rust
go test ./... (per service dir)  # Go
julia --project -e "Pkg.test()"  # Julia
Rscript -e "testthat::test_dir('r/tests')" # R

# Build native components
cd cpp/signal-engine && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
cd native/zig && zig build -Doptimize=ReleaseFast
```
