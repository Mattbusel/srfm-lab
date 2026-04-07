# SRFM Lab System Architecture

## Overview

SRFM Lab is a multi-language quant research and live trading platform built around the Black Hole (BH) physics momentum signal. The system has three main operational modes: research/backtesting, paper trading, and live trading.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SRFM Lab -- Full System                                  │
│                                                                                  │
│  ┌────────────────────────────────┐  ┌────────────────────────────────────────┐  │
│  │       Research Pipeline        │  │         Live Trading Pipeline           │  │
│  │                                │  │                                         │  │
│  │  data/ (parquet)               │  │  tools/live_trader_alpaca.py            │  │
│  │     ↓                          │  │     │                                   │  │
│  │  spacetime/engine/             │  │     │ every 15min bar                   │  │
│  │    data_loader.py              │  │     ↓                                   │  │
│  │    bh_engine.py                │  │  BH engine (3 TF)                       │  │
│  │    mc.py                       │  │     ↓                                   │  │
│  │    sensitivity.py              │  │  Rebalance logic                        │  │
│  │    correlation.py              │  │     ↓                                   │  │
│  │    archaeology.py              │  │  Alpaca SDK                             │  │
│  │     ↓                          │  │  (place orders)                         │  │
│  │  spacetime/reports/            │  │     ↓                                   │  │
│  │  (HTML/JSON/PNG output)        │  │  spacetime/cache/                       │  │
│  │                                │  │  live_state.json                        │  │
│  └────────────────────────────────┘  └────────────────────────────────────────┘  │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                        API Layer                                          │    │
│  │                                                                           │    │
│  │  spacetime/api/main.py  (FastAPI, port 8000)                             │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │/backtest │  │   /mc    │  │/sensitiv │  │/correlat │  │  /report │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────────────────────────┐ │    │
│  │  │/archaeol │  │/instrume │  │  WS:/ws/live  WS:/ws/replay            │ │    │
│  │  └──────────┘  └──────────┘  └────────────────────────────────────────┘ │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌──────────────────────┐  ┌───────────────────────────────────────────────┐    │
│  │    Go Gateway        │  │             Frontend (React/Vite)              │    │
│  │  cmd/gateway/        │  │  spacetime/web/   (port 5173 dev)              │    │
│  │  port 9000           │  │  terminal/        (port 5174 dev)              │    │
│  │  ┌──────────────┐    │  │  ┌─────────────┐  ┌──────────────────────┐   │    │
│  │  │ Rate limiting│    │  │  │ Arena UI    │  │  Terminal UI          │   │    │
│  │  │ Auth / JWT   │    │  │  │ (backtest,  │  │  (portfolio monitor) │   │    │
│  │  │ WS fanout    │    │  │  │  MC, charts)│  │                      │   │    │
│  │  └──────────────┘    │  │  └─────────────┘  └──────────────────────┘   │    │
│  └──────────────────────┘  └───────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                       Rust Extensions (PyO3)                              │    │
│  │  crates/srfm_core/                                                        │    │
│  │    MinkowskiClassifier    BlackHoleDetector    GeodesicAnalyzer           │    │
│  │    GravitationalLens      HawkingMonitor                                  │    │
│  │  extensions/              (maturin build → .pyd for Python import)        │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │               Analytics Warehouse                                         │    │
│  │  warehouse/schema/        (PostgreSQL 15+)                                │    │
│  │  warehouse/duckdb/        (DuckDB OLAP layer on parquet)                  │    │
│  │  warehouse/migrations/    (incremental schema updates)                    │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Live Trading

```
Alpaca Market Data Streams
    │
    │  WebSocket: CryptoBarsRequest, StockBarsRequest (15m bars)
    ▼
tools/live_trader_alpaca.py
    │
    ├── on_bar_15m() ──────────────────────────────────────────────────┐
    │      BH engine update (3 TFs: 15m, 1h, 1d)                      │
    │      regime classifier update                                     │
    │      tf_score computation                                         │
    │                                                                   │
    ├── check_entries()                                                 │
    │      if tf_score >= MIN_TF_SCORE and no open position:           │
    │          compute position fraction from TF_CAP table             │
    │          submit market order via Alpaca SDK                      │
    │                                                                   │
    ├── check_exits()                                                   │
    │      if BH collapses (mass < bh_collapse):                       │
    │          exit position (market order)                            │
    │      if regime changes to opposite:                              │
    │          exit position                                           │
    │                                                                   │
    └── update_live_state() ───────────────────────────────────────────┘
           write spacetime/cache/live_state.json
               {timestamp, positions, bh_states, equity}
```

The live state JSON is polled by the Go gateway, which fans it out to connected WebSocket clients (dashboard).

---

## Data Flow: Research / Backtesting

```
1. Data Ingestion
   scripts/fetch_polygon.py  →  data/bars/{tf}/{symbol}/YYYY-MM.parquet

2. Backtest Execution
   lean backtest strategies/larsa-v16/  →  results/larsa-v16/{timestamp}/

3. Spacetime Analysis (via API or CLI)
   POST /api/backtest
     { instrument, start, end, parameters }
       ↓
   spacetime/engine/data_loader.py      (load parquet)
       ↓
   spacetime/engine/bh_engine.py        (run BH simulation)
       ↓
   spacetime/engine/mc.py               (Monte Carlo)
   spacetime/engine/sensitivity.py      (parameter sensitivity)
   spacetime/engine/correlation.py      (inter-instrument BH correlations)
       ↓
   spacetime/reports/                   (HTML + JSON output)
       ↓
   Response: { trades, equity_curve, mc_bands, sensitivity_surface }

4. Analytics
   warehouse/duckdb/setup.py           (load parquet → DuckDB)
   warehouse/duckdb/analytics.sql      (analytical queries)
       ↓
   spacetime/reports/                  (CSV exports for dashboards)
```

---

## Data Flow: Monitoring (Live Dashboard)

```
live_trader_alpaca.py
    │ writes every 15min
    ▼
spacetime/cache/live_state.json
    │
    ▼
spacetime/api/main.py  (GET /api/live, WS /ws/live)
    │ polls file, parses JSON
    ▼
cmd/gateway/main.go    (port 9000)
    │ rate limiting, JWT auth, WS fan-out
    ▼
spacetime/web/         (React dashboard)
    │
    ├── PortfolioView: current positions, equity curve
    ├── BHStateView: per-instrument mass gauges, tf_score
    ├── RegimeView: current regime per instrument
    └── AlertView: recent risk events, drawdown alerts
```

---

## Directory Structure

```
srfm-lab/
├── ANATOMY.md                 # Codebase map
├── Makefile                   # Top-level build targets
├── Cargo.toml                 # Rust workspace
├── pyproject.toml             # Python project config
├── docker-compose.yml         # All services
│
├── crates/                    # Rust crates
│   ├── srfm_core/             # Core BH engine (published as Python extension)
│   └── srfm_gateway/          # (optional: gateway written in Rust)
│
├── extensions/                # PyO3 Python extensions built from crates/
│
├── spacetime/                 # Main Python application
│   ├── api/main.py            # FastAPI server
│   ├── engine/
│   │   ├── bh_engine.py       # BH backtester
│   │   ├── mc.py              # Monte Carlo engine
│   │   ├── sensitivity.py     # Parameter sweeps
│   │   ├── correlation.py     # Inter-instrument correlations
│   │   ├── data_loader.py     # Parquet/CSV ingestion
│   │   ├── archaeology.py     # Historical pattern finder
│   │   └── replay.py          # Bar-by-bar replay
│   ├── web/                   # React/Vite frontend (Spacetime Arena)
│   ├── cache/                 # Runtime state files
│   ├── logs/
│   └── reports/
│
├── terminal/                  # Separate React app (portfolio terminal)
│
├── cmd/                       # Go binaries
│   └── gateway/               # Go reverse proxy / WS gateway
│
├── strategies/                # LEAN strategy files
│   ├── larsa-v16/main.py      # Active strategy
│   └── ...
│
├── tools/                     # Utility scripts
│   ├── live_trader_alpaca.py  # Live trader
│   ├── param_sweep.py
│   └── compare.py
│
├── julia/                     # Julia statistical tools
│
├── research/                  # Jupyter notebooks, R scripts
│
├── data/                      # Market data (gitignored)
│   └── bars/{1d,1h,15m}/{symbol}/*.parquet
│
├── results/                   # Backtest results (gitignored)
│
├── warehouse/                 # SQL analytics warehouse (this PR)
│   ├── schema/
│   ├── migrations/
│   └── duckdb/
│
├── docs/                      # Documentation (this PR)
│   ├── theory/
│   ├── guides/
│   ├── api/
│   ├── research/
│   └── architecture.md
│
├── config/                    # YAML configurations
│
└── .github/workflows/         # CI/CD pipelines
```

---

## Technology Stack

| Layer               | Technology              | Version    |
|--------------------|-------------------------|------------|
| Core BH engine     | Rust (PyO3)             | 1.75+      |
| Strategy logic     | Python                  | 3.11+      |
| API server         | FastAPI + uvicorn       | 0.100+     |
| Frontend (Arena)   | React + Vite + TypeScript | 18+ / 5+  |
| Frontend (Terminal)| React + Vite            | 18+        |
| Gateway            | Go                      | 1.21+      |
| Backtesting        | QuantConnect LEAN       | 2.5        |
| Live broker        | Alpaca Markets          | -          |
| OLTP database      | PostgreSQL              | 15+        |
| OLAP database      | DuckDB                  | 0.10+      |
| Statistical tools  | Julia                   | 1.9+       |
| Charts / research  | R (ggplot2, PerformanceAnalytics) | 4+ |
| CI/CD              | GitHub Actions          | -          |
| Containerization   | Docker + docker-compose | -          |

---

## Service Ports

| Service              | Port  | Protocol     |
|---------------------|-------|--------------|
| FastAPI (spacetime)  | 8000  | HTTP/WS      |
| Vite dev (Arena)     | 5173  | HTTP         |
| Vite dev (Terminal)  | 5174  | HTTP         |
| Go gateway           | 9000  | HTTP/WS      |
| PostgreSQL           | 5432  | TCP          |

---

## Security Model

- The Go gateway is the public-facing entry point. The FastAPI server is not exposed publicly.
- JWT tokens are validated at the gateway for all dashboard WebSocket connections.
- Alpaca API keys are stored in environment variables, never in config files.
- Live state JSON contains no credentials, only portfolio state.
- Database passwords use Docker secrets in production.
