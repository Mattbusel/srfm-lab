# SRFM Lab — Relativistic Trading Research

A full-stack quantitative trading research platform built on **Special Relativistic Financial Mechanics (SRFM)** — from raw tick data to live paper trading, across 9 languages and 410K+ lines of code.

> Mad scientist workshop. Everything automated, everything measurable, rapid iteration at scale.

---

## Quick Start

```bash
# Live paper trader (Alpaca, $1M paper account)
python tools/live_trader_alpaca.py

# Spacetime Arena API + web UI
python run_api.py                  # FastAPI on :8765
cd spacetime/web && npm run dev   # React on :5173

# Research dashboard (reconciliation, walk-forward, signal analytics, regime lab)
cd research/dashboard && npm install && npm run dev  # React on :5174

# Research API server (Go, serves all research modules)
cd infra/research-api && go run main.go  # :8766

# Live/backtest reconciliation — diagnose slippage, drift, attribution
python -m research.reconciliation.cli recon run \
  --live tools/backtest_output/live_trades.db \
  --backtest tools/backtest_output/crypto_trades.csv \
  --output research/reports/

# Walk-forward + CPCV parameter optimization (Sobol/Bayesian)
python -m research.walk_forward.cli wf optimize \
  --trades tools/backtest_output/crypto_trades.csv \
  --method sobol --n-iter 200

# Signal analytics: IC decay, factor attribution, quintile analysis
python -m research.signal_analytics.cli signal report \
  --trades tools/backtest_output/crypto_trades.csv

# Regime lab: stress test 20 historical scenarios
python -m research.regime_lab.cli regime stress \
  --trades tools/backtest_output/crypto_trades.csv

# Live trading monitor (terminal dashboard)
python -m research.live_monitor.cli monitor run \
  --db tools/backtest_output/live_trades.db

# Rust tick-level backtest engine (10x faster parameter sweeps)
cargo run -p tick-backtest -- sweep --data-dir data/ --sym BTC --n-trials 1000

# Crypto backtest + Monte Carlo
python tools/crypto_backtest_mc.py

# BH engine backtest (ES/NQ/any instrument)
python spacetime/engine/bh_engine.py

# Genetic algorithm parameter optimizer
python ml/genetic/cli.py evolve --instrument BTC/USD

# Backtest all 29 strategies
python scripts/backtest_all_strategies.py

# Walk-forward analysis (legacy)
python tools/walk_forward_engine.py

# Factor analysis
python tools/factor_analysis.py

# Run regime analysis
python scripts/run_regime_analysis.py
```

---

## BH Physics — Core Primitives

> These are the fundamental building blocks. Everything else is built on top of them.

| Primitive | File | Description |
|-----------|------|-------------|
| `BHState` | `lib/srfm_core.py` | Single-instrument BH state machine. Core primitive — use this everywhere |
| `ATRTracker` | `tools/live_trader_alpaca.py:181` | EMA-based ATR. `update(h,l,c)` → `.atr` |
| `BullScaleTracker` | `tools/live_trader_alpaca.py` | Trend multiplier (3.0 bull / 1.0 bear) via EMA200/12/26/50 |
| `LARSAEngine` | `tools/live_trader_alpaca.py:194` | 3-timeframe (daily/hourly/15m) engine with delta scoring. The live trader brain |
| `BHEngine` | `spacetime/engine/bh_engine.py` | Universal backtester — multi-TF, MFE/MAE, regime per trade |

**BH Physics Reference:**

| Formula | Meaning |
|---------|---------|
| `ds² = c²dt² − dx²` | TIMELIKE (ordered) vs SPACELIKE (anomalous). CF tunes the speed of light |
| `mass += γ` on TIMELIKE | Mass accretes from causal bars. Ceiling ~2.0 |
| `mass *= decay` on SPACELIKE | Mass leaks on noise bars |
| `active = mass >= bh_form` | BH formation threshold (typically 1.5) |
| `tf_score = 4×daily + 2×hourly + 1×15m` | Multi-timeframe conviction score (max=7) |
| `delta_score = tf_score × mass × ATR` | Expected dollar move per bar — the allocation signal |

**Per-timeframe CF scaling** (in `LARSAEngine`):
- `cf_1d = base_cf × 5.0` — daily bars move 0.5-1%, needs higher CF
- `cf_1h = base_cf` — baseline
- `cf_15m = base_cf × 0.35` — 15m bars are noisy, lower threshold

---

## Live Trader

**`tools/live_trader_alpaca.py`** — Alpaca paper trader, crypto 24/7 + stocks market hours

Key constants to tune:
```python
STALE_15M_MOVE  = 0.001   # Cut losers moving < 0.1% per 15m bar
DELTA_MAX_FRAC  = 0.75    # Max single-instrument allocation
MIN_TRADE_FRAC  = 0.03    # 3% equity shift needed to trigger rebalance
MIN_HOLD        = 3       # Minimum bars before direction reversal
TAIL_FIXED_CAPITAL = 1_000_000  # Size positions off this equity
```

**Allocation logic** (delta-proportional):
```
score = tf_score × BH_mass × ATR
share = score / total_score   (capped at DELTA_MAX_FRAC)
```

**Exit rules:**
1. BH signal dies (daily + hourly both inactive) → close
2. Stale-15m: losing AND move < 0.1% on 15m bar → close immediately
3. Profitable (0.1%+ from entry) → position size locked, won't rotate out

**Rebalance triggers:**
- Every hourly bar close (forced)
- Every 15m bar close (threshold-gated, MIN_TRADE_FRAC=3%)
- Every 60s poll (threshold-gated)

---

## Repo Structure

```
srfm-lab/
│
├── 🔴 PRIMITIVES & CORE
│   ├── lib/srfm_core.py              # BHState, MinkowskiClassifier, HawkingMonitor
│   ├── lib/agents.py                 # D3QN, DDQN, TD3QN ensemble agents
│   ├── lib/regime.py                 # Regime detection
│   └── lib/risk.py                   # Risk management, stops, circuit breakers
│
├── 🟢 LIVE TRADING
│   ├── tools/live_trader_alpaca.py   # ★ Main live trader — Alpaca paper/live
│   └── tools/download_alpaca.py      # Historical data downloader
│
├── 🔵 BACKTESTING & RESEARCH TOOLS
│   ├── tools/crypto_backtest_mc.py   # ★ Crypto BH backtest + Monte Carlo
│   ├── tools/walk_forward_engine.py  # Walk-forward analysis engine
│   ├── tools/factor_analysis.py      # Fama-MacBeth, IC/ICIR, factor decay
│   ├── tools/execution_analyzer.py   # Almgren-Chriss, IS, slippage analysis
│   ├── tools/stress_testing.py       # 10 historical stress scenarios
│   ├── tools/alpha_decay.py          # Signal half-life, optimal rebalance freq
│   └── tools/local_backtest.py       # Quick local BH backtest
│
├── 🟣 SPACETIME ARENA (full research platform)
│   ├── run_api.py                    # ★ Launch FastAPI server (:8765)
│   ├── spacetime/engine/bh_engine.py # Universal BH backtester
│   ├── spacetime/engine/mc.py        # Regime-aware Monte Carlo
│   ├── spacetime/engine/sensitivity.py # Parameter sensitivity sweeps
│   ├── spacetime/engine/correlation.py # BH activation correlation
│   ├── spacetime/engine/archaeology.py # QC CSV trade DB
│   ├── spacetime/engine/replay.py    # Bar-by-bar replay engine
│   ├── spacetime/api/main.py         # FastAPI: 15 routes + WebSocket
│   ├── spacetime/reports/generator.py # 9-section PDF reports
│   └── spacetime/web/                # React/TS frontend (:5173)
│
├── 🟡 ML / AI
│   ├── ml/rl_agent/                  # PPO + SAC + DQN + Transformer RL agents
│   ├── ml/nlp_alpha/                 # FinBERT news sentiment → alpha signals
│   ├── ml/genetic/                   # Genetic algorithm strategy optimizer
│   └── strategies/ml_alpha/          # ML-based strategy implementations
│
├── 🟠 STRATEGIES (29 total)
│   ├── strategies/larsa-v16/         # ★ Current best BH strategy
│   ├── strategies/momentum/          # Momentum strategies
│   ├── strategies/mean_reversion/    # Mean reversion strategies
│   ├── strategies/volatility/        # Vol strategies
│   ├── strategies/crypto/            # Crypto-specific strategies
│   └── strategies/event_driven/      # Event-driven strategies
│
├── ⚙️  RUST CRATES (compiled, high-performance)
│   ├── crates/tick-backtest/         # ★ NEW: Tick-level BH backtest engine (rayon parallel sweeps)
│   ├── crates/larsa-core/            # ★ Core BH engine in Rust
│   ├── crates/orderbook-sim/         # L2 orderbook, Hawkes process
│   ├── crates/portfolio-engine/      # Ledoit-Wolf, HRP, Black-Litterman
│   ├── crates/risk-engine/           # VaR/CVaR, Greeks, stress scenarios
│   ├── crates/regime-detector/       # HMM, PELT, Kalman filters
│   ├── crates/data-pipeline/         # 15 indicators, CPCV splitter
│   ├── crates/fix-engine/            # FIX 4.2/4.4 protocol
│   ├── crates/options-engine/        # BSM/Heston/SABR/SVI pricing
│   └── crates/smart-order-router/    # Multi-venue SOR
│
├── 🐹 GO INFRA
│   ├── infra/research-api/           # ★ NEW: Research API server (:8766) — 14 routes + WebSocket
│   ├── infra/gateway/                # ★ Market data gateway (17 indicators, regime)
│   ├── infra/monitor/                # Trade journal, alert backtester
│   ├── infra/timeseries/             # InfluxDB, DuckDB factor model
│   ├── infra/grpc/                   # gRPC microservices (market/strategy/risk/portfolio)
│   ├── infra/event-bus/              # Redis pub/sub event bus
│   └── infra/websocket-hub/          # Scalable WebSocket broadcasting
│
├── ⚛️  REACT FRONTENDS
│   ├── research/dashboard/           # ★ NEW: Research dashboard (8 pages: recon/WF/signals/regime/MC)
│   ├── terminal/                     # ★ Trading terminal (6 pages + options chain)
│   └── dashboard/                    # Executive P&L dashboard (Tremor)
│
├── 🔬 JULIA QUANT SUITE
│   ├── julia/src/BHPhysics.jl        # BH engine, walk-forward, cross-sectional
│   ├── julia/src/Stochastic.jl       # GARCH, Heston, Hawkes, OU, Merton JD
│   ├── julia/src/Statistics.jl       # Sharpe, Hurst, Ljung-Box, bootstrap CI
│   ├── julia/src/Optimization.jl     # HRP, Black-Litterman, CVaR, Kelly
│   ├── julia/src/Visualization.jl    # 14 plot functions
│   ├── julia/src/Bayesian.jl         # Turing.jl MCMC, Bayesian CF estimation
│   ├── julia/src/FactorModel.jl      # Fama-MacBeth, Barra risk model
│   ├── julia/src/OptimalExecution.jl # Almgren-Chriss, Obizhaeva-Wang
│   ├── julia/src/InterestRates.jl    # Vasicek, CIR, HJM, LMM
│   ├── julia/src/VolatilitySurface.jl # SVI, SABR, Dupire local vol
│   ├── julia/src/NetworkAnalysis.jl  # MST, community detection, CoVaR
│   ├── julia/src/MachineLearning.jl  # LSTM, XGBoost, GP regression
│   └── julia/notebooks/              # 6 research notebooks
│
├── 🔬 RESEARCH (simulation lab)
│   ├── research/reconciliation/      # ★ NEW: Live vs backtest recon — slippage, drift, attribution
│   ├── research/walk_forward/        # ★ NEW: CPCV walk-forward, Sobol/Bayesian param opt, DSR
│   ├── research/param_explorer/      # ★ NEW: Saltelli Sobol, Morris, GP Bayesian opt, OAT
│   ├── research/regime_lab/          # ★ NEW: HMM, PELT, Heston/GARCH gen, 20 stress scenarios
│   ├── research/signal_analytics/    # ★ NEW: IC/ICIR, factor attribution, alpha decay, quintiles
│   ├── research/agent_training/      # ★ NEW: Pure-numpy RL (D3QN/TD3/PPO), PER, curriculum
│   ├── research/portfolio_lab/       # ★ NEW: HRP, Black-Litterman, risk parity, DCC-GARCH
│   ├── research/execution_research/  # ★ NEW: TCA, Almgren-Chriss order splitting, market impact
│   ├── research/live_monitor/        # ★ NEW: Live trading dashboard, diagnostics, alerts
│   ├── research/factor_model/        # Factor construction, IC, attribution
│   ├── research/regime_analysis/     # Regime detection comparison
│   ├── research/options/             # Options pricing, Greeks
│   ├── research/alternative_data/    # Alt data pipelines
│   ├── research/onchain/             # Crypto on-chain (DeFi, whales, NVT)
│   ├── research/options_flow/        # GEX, unusual flow, vol regime
│   ├── research/micro_structure/     # OFI, VPIN, spread decomposition
│   └── research/notebooks/           # 8 research notebooks
│
├── ⚡ C/C++/ZIG (native performance)
│   ├── extensions/fast_indicators/   # ★ C extension: 20 indicators + bh_backtest_c
│   ├── native/orderbook/             # Lock-free L3 orderbook (C++17 + AVX2)
│   ├── native/matrix/                # SIMD matrix ops (AVX2)
│   ├── native/ringbuffer/            # mmap tick store
│   └── native/zig/                   # Zig ITCH 5.0 decoder + orderbook
│
├── 📊 R STATISTICAL SUITE
│   ├── r/bh_analysis.R               # BH state reconstruction, ggplot2
│   ├── r/factor_research.R           # Fama-French, IC, Newey-West
│   ├── r/regime_models.R             # HMM, Markov switching, GARCH-DCC
│   ├── r/portfolio_optimization.R    # quadprog, PortfolioAnalytics, BL
│   ├── r/backtesting.R               # PerformanceAnalytics, bootstrap CI
│   └── r/visualization.R             # ggplot2 chart suite
│
├── 🗄️  WAREHOUSE
│   ├── warehouse/schema/             # SQLite: 6 schema files, 30 named queries
│   └── warehouse/duckdb/             # DuckDB analytics + BH UDFs
│
├── 🧪 TESTS
│   └── tests/                        # 10 test files
│
├── 📚 DOCS
│   ├── docs/theory/bh_physics.md     # Full mathematical derivation
│   ├── docs/theory/monte_carlo.md    # MC theory, Kelly derivation
│   ├── docs/theory/portfolio_theory.md # BL, HRP, correlation-adjusted sizing
│   └── docs/architecture.md          # System architecture + data flows
│
└── 🔧 SCRIPTS & CONFIG
    ├── scripts/backtest_all_strategies.py
    ├── scripts/optimize_portfolio.py
    ├── scripts/run_regime_analysis.py
    ├── scripts/stress_test.py
    ├── config/instruments.yaml       # All 30+ instruments with CF calibration
    ├── config/risk_limits.yaml
    ├── Makefile                      # 60+ targets
    └── .github/workflows/            # CI/CD: Python/Rust/Go/TS/Julia
```

---

## SRFM Physics Reference

| Component | Formula | Interpretation |
|-----------|---------|----------------|
| MinkowskiClassifier | `ds² = c²dt² − dx²` | TIMELIKE = ordered, causal; SPACELIKE = anomalous velocity |
| BlackHoleDetector | mass accretes on TIMELIKE | Well forms when `mass ≥ bh_form` |
| HawkingMonitor | `T_H = 1/(8πM)` | Cold well = stable signal; hot well = reduce size |
| GravitationalLens | `μ = (u²+2) / (u√(u²+4))` | BH amplifies signal from other indicators |
| ProperTimeClock | `dτ² = dt² − (dx/c)²` | Proper time gates entries |
| Delta Score | `tf × mass × ATR` | Expected dollar move — the allocation signal |

---

## Key Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `CF` (base) | `instruments.yaml` | 0.001–0.003 | Minkowski speed of light. Higher = fewer TIMELIKE bars |
| `cf_1d` | `bh_engine.py` | `base × 5.0` | Daily CF scaling |
| `cf_15m` | `bh_engine.py` | `base × 0.35` | 15m CF scaling |
| `bh_form` | `instruments.yaml` | 1.5 | Mass threshold for BH activation |
| `STALE_15M_MOVE` | `live_trader_alpaca.py` | 0.001 | Cut losers < 0.1% move per 15m |
| `DELTA_MAX_FRAC` | `live_trader_alpaca.py` | 0.75 | Max single-asset allocation |
| `MIN_TRADE_FRAC` | `live_trader_alpaca.py` | 0.03 | Minimum shift to trigger rebalance |

---

## Instrument Universe

**Crypto (24/7):** BTC, ETH, SOL, XRP, AVAX, LINK, DOT, UNI, AAVE, LTC, BCH, ADA, DOGE, SHIB, FIL, GRT, BAT, CRV, SUSHI  
**Equities/ETFs:** SPY, QQQ, DIA, USO, GLD, TLT, UNG, VIXY  
**Futures (backtest):** ES, NQ, YM, CL, GC, NG, ZB, VX

---

## Baselines

| Strategy | Return | Instrument | Notes |
|----------|--------|-----------|-------|
| LARSA v1 | 274% | ES | Original QC baseline — do not modify |
| LARSA v16 | TBD | Multi | Current live version |
| Crypto BH MC | -11% CAGR | 19 coins | 2021–2026; 2024 was +26% |

---

## LOC by Language

| Language | LOC | Key systems |
|----------|-----|------------|
| Python | ~235K | Strategies, ML, research, backtesting, live trader, sim lab (9 new modules) |
| Rust | ~24K | tick-backtest engine, orderbook, portfolio, risk, regime, FIX, options, SOR |
| Go | ~38K | research-api (:8766), gateway, monitor, gRPC, event bus, WebSocket hub |
| TypeScript | ~42K | Research dashboard (:5174), trading terminal, Spacetime Arena web |
| Julia | ~20K | BHPhysics, stochastic, optimization, vol surface, networks |
| C/C++ | ~15K | Fast indicators, L3 orderbook, SIMD matrix, ring buffer |
| Zig | ~8K | ITCH 5.0 decoder, low-latency orderbook |
| R | ~10K | Statistical analysis, ggplot2 visualization |
| SQL | ~5K | Warehouse schema (016 migrations), DuckDB analytics |
| **Total** | **~410K** | |

---

## Running Services

| Service | Command | Port |
|---------|---------|------|
| Spacetime Arena API | `python run_api.py` | 8765 |
| Research API (Go) | `cd infra/research-api && go run main.go` | 8766 |
| Spacetime Arena Web | `cd spacetime/web && npm run dev` | 5173 |
| Trading Terminal | `cd terminal && npm run dev` | 5174 |
| Research Dashboard | `cd research/dashboard && npm run dev` | 5174 |
| P&L Dashboard | `cd dashboard && npm run dev` | 5175 |
| Live Trader | `python tools/live_trader_alpaca.py` | — |
| Live Monitor | `python -m research.live_monitor.cli monitor run` | — |

---

*License: Proprietary. All rights reserved. This repository contains unpublished research and strategy IP. Do not distribute.*
