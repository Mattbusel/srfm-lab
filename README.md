# SRFM Trading Lab

A full-stack quantitative trading research platform built on **Special Relativistic Financial Mechanics (SRFM)** — from raw tick data to live paper trading and autonomous idea discovery, across 9 languages and 410K+ lines of code.

> Mad scientist workshop. Everything automated, everything measurable, rapid iteration at scale.

---

## What Makes This Different

The core innovation is the **Black Hole (BH) Physics Strategy** — a novel signal model derived from special-relativistic mechanics applied to price data. Price bars are classified as *timelike* or *spacelike* using a Minkowski spacetime metric. Mass accumulates on ordered (causal) bars, and a gravitational well forms when mass crosses a threshold — the **black hole formation event** that gates entries.

On top of this sits the **Idea Automation Engine (IAE)** — an autonomous 42K+ LOC research system that runs genetic genome evolution, causal discovery, regime classification, walk-forward validation, and academic paper mining continuously, feeding confirmed patterns back into live strategy parameters.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SRFM Trading Lab                                    │
│                                                                              │
│  ┌─────────────────────┐    ┌──────────────────────────────────────────┐    │
│  │   Live Trader        │    │         Idea Automation Engine (IAE)     │    │
│  │  live_trader_        │    │                                          │    │
│  │  alpaca.py          │    │  ┌──────────┐  ┌──────────┐  ┌────────┐ │    │
│  │                     │◄───┤  │ Genome   │  │Hypothesis│  │Regime  │ │    │
│  │  BH Physics Engine  │    │  │ Evolution│  │Generator │  │Oracle  │ │    │
│  │  GARCH vol forecast │    │  │(NSGA-II) │  │(Bayesian)│  │6 modes │ │    │
│  │  OU mean reversion  │    │  └────┬─────┘  └────┬─────┘  └───┬────┘ │    │
│  │  Mayer Multiple     │    │       │              │             │      │    │
│  │  BTC lead signal    │    │  ┌────▼─────────────▼─────────────▼────┐ │    │
│  │  Island sizing      │    │  │         Event Bus (:8768)            │ │    │
│  └──────┬──────────────┘    │  └────────────────┬─────────────────────┘ │    │
│         │                   │                   │                        │    │
│         │ trades            │  ┌────────────────▼─────────────────────┐ │    │
│         ▼                   │  │    Go API Server (:8767)              │ │    │
│  ┌──────────────┐           │  │    Scheduler (:8769)                  │ │    │
│  │ trade_logger │           │  │    Webhook Service (:8770)            │ │    │
│  │  .py (SQLite)│           │  └────────────────┬─────────────────────┘ │    │
│  └──────┬───────┘           │                   │                        │    │
│         │                   │  ┌────────────────▼─────────────────────┐ │    │
│         │                   │  │    React Dashboard (:5175)            │ │    │
│         │                   │  │    Dark terminal theme, D3 lineage   │ │    │
│         │                   │  └──────────────────────────────────────┘ │    │
│         │                   │                                            │    │
│         │                   │  ┌────────────┐  ┌───────────┐            │    │
│         │                   │  │ Rust MC    │  │ R Stats   │            │    │
│         │                   │  │ Engine     │  │ (HMM,WFA) │            │    │
│         │                   │  └────────────┘  └───────────┘            │    │
│         │                   │  ┌────────────────────────────────────┐   │    │
│         │                   │  │ Julia (Bayesian opt, FCI causal)   │   │    │
│         │                   └──┴────────────────────────────────────┴───┘    │
│         │                                                                     │
│         ▼                                                                     │
│  ┌──────────────────────┐   ┌────────────────────────────────────────────┐   │
│  │  live_monitor/        │   │         crypto_backtest_mc.py              │   │
│  │  CLI terminal dash   │   │  3-TF BH engine + GARCH + OU + Monte Carlo │   │
│  │  Web dashboard       │   │  10,000-sim bootstrap, Mayer dampener      │   │
│  └──────────────────────┘   └────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Data flow:** Alpaca streams → BH Engine → position sizing → orders → SQLite log → IAE ingestion → genome evolution → parameter feedback → live strategy.

---

## Quick Start

### Prerequisites

```bash
pip install alpaca-py pandas numpy scipy statsmodels matplotlib
# Rust (genome engine, Monte Carlo)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Go 1.22+ (API, bus, scheduler, webhook)
# Node 18+ (React dashboard)
```

### Backtest

```bash
# Full crypto BH backtest + 1,000-path Monte Carlo (downloads from Alpaca)
python tools/crypto_backtest_mc.py

# Custom date range and parameters
python tools/crypto_backtest_mc.py \
  --start-date 2022-01-01 \
  --end-date 2025-01-01 \
  --mc-paths 5000 \
  --symbols BTC,ETH,SOL \
  --bh-form 1.92 \
  --corr 0.25 \
  --garch-target-vol 1.20 \
  --output-dir tools/backtest_output \
  --verbose
```

### Live Paper Trading

```bash
# Start live trader (Alpaca paper account, $1M simulated)
python tools/live_trader_alpaca.py

# With options
python tools/live_trader_alpaca.py \
  --paper \
  --log-level INFO \
  --db-path tools/backtest_output/live_trades.db \
  --rebalance-interval 15

# Dry run — log orders but don't submit
python tools/live_trader_alpaca.py --dry-run
```

### Live Monitor

```bash
# Terminal dashboard (curses)
python -m research.live_monitor.cli monitor run \
  --db tools/backtest_output/live_trades.db
```

### IAE Stack

```bash
# 1. Initialize database schema
python -m idea_engine.db.migrate --db idea-engine/idea_engine.db

# 2. Run ingestion pipeline (mines patterns from backtest + live trade history)
python -m idea_engine.ingestion.pipeline \
  --db idea-engine/idea_engine.db \
  --live-db tools/backtest_output/live_trades.db \
  --miners time_of_day,regime_cluster,bh_physics,drawdown \
  --verbose

# 3. Start Go API server
cd idea-engine && go run cmd/api/main.go        # :8767

# 4. Start event bus
go run cmd/bus/main.go                          # :8768

# 5. Start scheduler
go run cmd/scheduler/main.go                    # :8769

# 6. Start webhook service
go run cmd/webhook/main.go                      # :8770

# 7. Start React dashboard
cd idea-engine/dashboard && npm install && npm run dev   # :5175

# 8. Build Rust genome + Monte Carlo engine
cd idea-engine/rust && cargo build --release
```

---

## BH Physics Strategy

### Core Concept

Traditional momentum and mean-reversion signals use ad-hoc smoothing. The BH strategy asks a more fundamental question: **does this price bar represent causal, ordered motion — or anomalous, noisy displacement?**

The answer comes from Minkowski spacetime. Each price bar is classified using:

```
ds² = c²dt² − dx²
```

where `c` is a calibrated speed-of-light threshold (`CF`), `dt` is normalized time, and `dx` is normalized price displacement. If `ds² > 0` the bar is **timelike** (ordered, causal). If `ds² < 0` it is **spacelike** (anomalous velocity — noise).

### Mass Accumulation (EMA-Based)

Mass accumulates via an exponential filter on timelike bars:

```python
# On a timelike bar:
mass = mass * 0.97 + 0.03 * min(2.0, 1 + ctl * 0.1)

# On a spacelike bar:
mass *= BH_DECAY   # 0.924 — mass bleeds away on noise

# Black Hole forms when:
active = mass >= BH_FORM   # BH_FORM = 1.92 (effective reachable ceiling)
```

`ctl` is the count of consecutive timelike bars. `BH_FORM = 1.92` because the EMA asymptotes to 2.0 — 1.92 is the highest reliably reachable value, filtering weak signals while still triggering on sustained trends.

### Multi-Timeframe Conviction

Three timeframes are scored simultaneously:

| Timeframe | CF Multiplier | Weight |
|-----------|--------------|--------|
| Daily     | base × 5.0   | 4      |
| Hourly    | base × 1.0   | 2      |
| 15-minute | base × 0.35  | 1      |

```
tf_score = 4×(daily active) + 2×(hourly active) + 1×(15m active)  # max = 7
delta_score = tf_score × mass × ATR  # expected dollar move — the allocation signal
```

### GARCH(1,1) Online Volatility Forecasting

The strategy targets 120% annualized volatility (crypto volatility regime). GARCH parameters are updated online each bar, and position sizes are scaled so realized vol tracks target:

```python
garch_vol_forecast → position_scale = target_vol / garch_vol_forecast
```

### Ornstein-Uhlenbeck Mean Reversion

When the BH is flat (active but mass not growing), an OU process is fitted to the residuals. The OU position takes `OU_FRAC = 0.08` (8%) of equity, fading the mean-reversion component against the trend component.

### Mayer Multiple Dampener

The Mayer Multiple (price / 200-day EMA) acts as a macro regime dampener. When price is extended far above the 200d EMA, new position sizes are reduced proportionally — avoiding chasing parabolic moves.

### BTC Cross-Asset Lead Signal

BTC is treated as the lead instrument for altcoins. A positive BTC BH activation increases conviction for correlated altcoin signals. BTC BH state is factored into altcoin `delta_score` computation.

### Risk Allocation

```
CORR = 0.25          # cross-asset correlation assumption
N_INST = 20          # number of instruments
CORR_FACTOR = sqrt(N + N*(N-1)*CORR)   # portfolio correlation adjustment
PER_INST_RISK = DAILY_RISK / CORR_FACTOR
```

Position size per instrument:

```
share_i = delta_score_i / sum(delta_scores)   # capped at DELTA_MAX_FRAC = 0.40
position_i = share_i × TAIL_FIXED_CAPITAL × garch_scale × mayer_dampener
```

### Exit Rules

1. **BH dies** — daily + hourly both inactive → close immediately
2. **Stale 15m** — position is losing AND 15m bar moved < 0.1% → close (stop leak)
3. **Profitable lock** — position up > 0.1% from entry → size locked, won't rotate out mid-trend
4. **Min hold** — no direction reversal before `MIN_HOLD = 6` bars

---

## Instrument Universe

**Crypto (24/7):** BTC, ETH, SOL, XRP, AVAX, LINK, DOT, UNI, AAVE, LTC, BCH, DOGE, SHIB, GRT, BAT, CRV, SUSHI, MKR, YFI

**Equities/ETFs (market hours, via Alpaca):**
ES→SPY, NQ→QQQ, YM→DIA, CL→USO, GC→GLD, ZB→TLT, NG→UNG, VX→VIXY

CF (speed-of-light) is calibrated per instrument to match typical volatility regimes:
- BTC: `cf_1d=0.05`, `cf_1h=0.015`, `cf_15m=0.005`
- SPY: `cf_1d=0.005`, `cf_1h=0.001`, `cf_15m=0.0003`

---

## Idea Automation Engine (IAE)

The IAE is an autonomous research system that observes live performance, generates hypotheses, tests them rigorously, and feeds confirmed findings back into strategy parameters — a closed research loop with no human required.

### Module Table

| Module | Language | Description |
|--------|----------|-------------|
| Genome Evolution | Python + Rust | NSGA-II multi-objective optimizer with island model; 52 test suite; evolves strategy parameter sets |
| Hypothesis Generator | Python | Bayesian scoring of candidate hypotheses; priors updated from live performance |
| Causal Discovery | Python | Granger causality + PC algorithm; identifies leading indicators |
| Walk-Forward Engine | Python | Rolling WFA with CPCV (Combinatorial Purged CV); prevents overfitting |
| Regime Oracle | Python | 6-regime classifier: BULL / BEAR / NEUTRAL / CRISIS / RECOVERY / TOPPING |
| Signal Library | Python | 60+ signals; IC tracking against forward returns |
| Feature Store | Python | Computed features with IC decay monitoring; auto-prunes stale features |
| Shadow Runner | Python | N live shadow strategies running in parallel; silent P&L tracking |
| Counterfactual Oracle | Rust | Parameter space exploration; answers "what if BH_FORM were 1.85?" |
| Academic Miner | Python | Scrapes arXiv + SSRN; extracts and scores novel signals |
| Serendipity Generator | Python | Domain analogy + mutation engine; imports signals from physics, biology, network theory |
| Narrative Intelligence | Python | Weekly reports + Slack/email alerts; natural language performance attribution |
| Stats Service | R + Julia | R: HMM, WFA, White's Reality Check. Julia: Bayesian optimization, FCI causal discovery |
| Genealogy Tree | TypeScript/D3 | Force-directed lineage graph; every strategy traces its evolutionary ancestry |
| Live Feedback Loop | Python | Drift detection (KS test, PSI); signal attribution; live performance attribution |
| Experiment Tracker | Python | MLflow-style tracker; SQLite-backed; every run logged with parameters + metrics |
| Risk Engine | Python | VaR, CVaR, tail risk; HRP portfolio optimizer; per-instrument risk attribution |
| Data Quality Monitor | Python | Missing bar detection, outlier flagging, feed health |
| Rust Monte Carlo | Rust | High-performance MC simulation engine; path-dependent payoff modeling |
| Ingestion Pipeline | Python | 4-stage pipeline: load → mine → filter (bootstrap) → persist |

### IAE Services

| Service | Port | Description |
|---------|------|-------------|
| Go API | :8767 | REST API — hypotheses, genomes, signals, experiments |
| Event Bus | :8768 | Internal pub/sub; all modules communicate via events |
| Scheduler | :8769 | Cron-based task orchestration; triggers miners, WFA, genome runs |
| Webhook Service | :8770 | External integrations; receives Alpaca webhooks, pushes alerts |
| React Dashboard | :5175 | Dark terminal theme; D3 genealogy tree; live charts via Recharts |

---

## Service Endpoints

| Service | Command | Port |
|---------|---------|------|
| Live Trader | `python tools/live_trader_alpaca.py` | — |
| Live Monitor (CLI) | `python -m research.live_monitor.cli monitor run --db tools/backtest_output/live_trades.db` | — |
| IAE API | `cd idea-engine && go run cmd/api/main.go` | 8767 |
| IAE Event Bus | `go run cmd/bus/main.go` | 8768 |
| IAE Scheduler | `go run cmd/scheduler/main.go` | 8769 |
| IAE Webhook | `go run cmd/webhook/main.go` | 8770 |
| IAE Dashboard | `cd idea-engine/dashboard && npm run dev` | 5175 |
| Spacetime Arena API | `python run_api.py` | 8765 |
| Research API (Go) | `cd infra/research-api && go run main.go` | 8766 |
| Spacetime Web | `cd spacetime/web && npm run dev` | 5173 |
| Research Dashboard | `cd research/dashboard && npm run dev` | 5174 |

---

## Development Commands

### Backtesting

```bash
# Full crypto backtest + Monte Carlo
python tools/crypto_backtest_mc.py --verbose

# Custom parameters
python tools/crypto_backtest_mc.py \
  --start-date 2023-01-01 \
  --mc-paths 5000 \
  --symbols BTC,ETH,SOL,XRP \
  --bh-form 1.92 \
  --corr 0.25 \
  --garch-target-vol 1.20 \
  --ou-frac 0.08 \
  --no-plot \
  --verbose

# BH engine backtest (legacy, all instruments)
python spacetime/engine/bh_engine.py

# Rust tick-level parameter sweeps (10x faster)
cargo run -p tick-backtest -- sweep --data-dir data/ --sym BTC --n-trials 1000

# Walk-forward + CPCV parameter optimization
python -m research.walk_forward.cli wf optimize \
  --trades tools/backtest_output/crypto_trades.csv \
  --method sobol --n-iter 200

# Stress test 20 historical scenarios
python -m research.regime_lab.cli regime stress \
  --trades tools/backtest_output/crypto_trades.csv
```

### Live Trading

```bash
# Start live paper trader
python tools/live_trader_alpaca.py --paper --log-level INFO

# Dry run (no order submission)
python tools/live_trader_alpaca.py --dry-run --log-level DEBUG

# Custom rebalance interval (minutes)
python tools/live_trader_alpaca.py --rebalance-interval 30
```

### IAE

```bash
# Initialize / migrate schema
python -m idea_engine.db.migrate

# Run ingestion pipeline (all miners)
python -m idea_engine.ingestion.pipeline --verbose

# Run ingestion pipeline (specific miners, dry run)
python -m idea_engine.ingestion.pipeline \
  --miners time_of_day,bh_physics \
  --dry-run \
  --verbose

# Build Rust engine
cd idea-engine/rust && cargo build --release

# Run genome evolution (Go scheduler triggers this automatically)
cd idea-engine/rust && ./target/release/genome-engine --generations 100 --pop-size 200
```

### Research

```bash
# Live vs backtest reconciliation
python -m research.reconciliation.cli recon run \
  --live tools/backtest_output/live_trades.db \
  --backtest tools/backtest_output/crypto_trades.csv \
  --output research/reports/

# Signal analytics: IC decay, factor attribution, quintile analysis
python -m research.signal_analytics.cli signal report \
  --trades tools/backtest_output/crypto_trades.csv

# R stats service (HMM, White's Reality Check)
Rscript idea-engine/stats/hmm_analysis.R

# Julia Bayesian optimization
julia idea-engine/stats/bayes_opt.jl
```

### Build / Test

```bash
# Python tests
pytest tests/ -v

# Rust (all crates)
cargo test --workspace

# Go
cd idea-engine && go test ./...

# TypeScript dashboard
cd idea-engine/dashboard && npm test
```

---

## File Structure

```
srfm-lab/
│
├── tools/
│   ├── live_trader_alpaca.py        # ★ Live Alpaca paper trader (BH Physics + GARCH + OU)
│   ├── crypto_backtest_mc.py        # ★ Crypto BH backtest + Monte Carlo engine
│   ├── walk_forward_engine.py       # Walk-forward analysis
│   ├── factor_analysis.py           # Fama-MacBeth, IC/ICIR, factor decay
│   └── backtest_output/             # SQLite DBs, CSV trade logs, PNG charts
│
├── idea-engine/                     # ★ Idea Automation Engine (42K+ LOC)
│   ├── db/
│   │   ├── migrate.py               # Schema migration runner
│   │   ├── schema.sql               # Base schema
│   │   └── migrations/              # Module schema extensions
│   ├── ingestion/
│   │   ├── pipeline.py              # 4-stage ingestion pipeline
│   │   ├── config.py                # Paths and parameters
│   │   ├── loaders/                 # Backtest, live trade, walk-forward loaders
│   │   ├── miners/                  # TimeOfDay, RegimeCluster, BHPhysics, Drawdown
│   │   └── statistical_filters/     # Bootstrap filter (BH correction)
│   ├── genome/                      # Rust NSGA-II genome evolution
│   ├── hypothesis/                  # Bayesian hypothesis generator
│   ├── causal/                      # Granger + PC algorithm causal discovery
│   ├── walk_forward/                # CPCV walk-forward engine
│   ├── regime/                      # 6-regime oracle
│   ├── signals/                     # 60+ signal library with IC tracking
│   ├── shadow/                      # Shadow strategy runner
│   ├── counterfactual/              # Rust counterfactual oracle
│   ├── academic/                    # arXiv + SSRN miner
│   ├── serendipity/                 # Domain analogy + mutation engine
│   ├── narrative/                   # Weekly reports, alerts
│   ├── stats/                       # R (HMM, WFA) + Julia (Bayesian opt, FCI)
│   ├── genealogy/                   # D3 force-directed lineage tree
│   ├── feedback/                    # Drift detection + attribution
│   ├── experiments/                 # MLflow-style experiment tracker
│   ├── risk/                        # VaR, CVaR, HRP optimizer
│   ├── data_quality/                # Feed health monitor
│   ├── rust/                        # Rust Monte Carlo + genome engine
│   ├── cmd/                         # Go API (:8767), bus (:8768), scheduler (:8769), webhook (:8770)
│   ├── dashboard/                   # React/TS + Vite + Recharts + D3 (:5175)
│   └── idea_engine.db               # SQLite (WAL mode) — patterns, hypotheses, experiments
│
├── infra/
│   ├── observability/
│   │   └── trade_logger.py          # SQLite trade logger (WAL mode)
│   ├── research-api/                # Go research API (:8766)
│   ├── gateway/                     # Market data gateway (17 indicators)
│   ├── grpc/                        # gRPC microservices
│   └── event-bus/                   # Redis pub/sub
│
├── research/
│   ├── live_monitor/                # ★ Terminal CLI + web dashboard for live trades
│   ├── reconciliation/              # Live vs backtest recon
│   ├── walk_forward/                # CPCV walk-forward, Sobol/Bayesian param opt
│   ├── regime_lab/                  # HMM, PELT, 20 stress scenarios
│   ├── signal_analytics/            # IC/ICIR, alpha decay, quintile analysis
│   └── dashboard/                   # React research dashboard (:5174)
│
├── spacetime/
│   ├── engine/bh_engine.py          # Universal BH backtester
│   ├── api/main.py                  # FastAPI (:8765), 15 routes + WebSocket
│   └── web/                         # React Spacetime Arena (:5173)
│
├── lib/
│   ├── srfm_core.py                 # BHState, MinkowskiClassifier, HawkingMonitor
│   ├── agents.py                    # D3QN, DDQN, TD3QN ensemble agents
│   ├── regime.py                    # Regime detector
│   └── risk.py                      # Risk management, stops, circuit breakers
│
├── crates/
│   ├── tick-backtest/               # Tick-level BH backtest (rayon parallel sweeps)
│   ├── larsa-core/                  # Core BH engine in Rust
│   ├── portfolio-engine/            # Ledoit-Wolf, HRP, Black-Litterman
│   └── risk-engine/                 # VaR/CVaR, Greeks, stress scenarios
│
├── julia/src/
│   ├── BHPhysics.jl                 # BH engine, walk-forward, cross-sectional
│   ├── Stochastic.jl                # GARCH, Heston, Hawkes, OU, Merton JD
│   └── Bayesian.jl                  # Turing.jl MCMC, Bayesian CF estimation
│
├── r/
│   ├── bh_analysis.R                # BH state reconstruction
│   └── regime_models.R              # HMM, Markov switching, GARCH-DCC
│
├── config/
│   ├── instruments.yaml             # All 30+ instruments with CF calibration
│   └── risk_limits.yaml
│
├── Makefile                         # 60+ targets
└── .github/workflows/               # CI/CD: Python/Rust/Go/TS/Julia
```

---

## Stack

| Language | LOC | Key Systems |
|----------|-----|-------------|
| Python | ~235K | Live trader, backtesting, IAE pipeline, ML, research modules |
| TypeScript/React | ~42K | IAE dashboard (:5175), research dashboard (:5174), trading terminal |
| Go | ~38K | IAE API (:8767), bus (:8768), scheduler (:8769), webhook (:8770), research API (:8766) |
| Rust | ~24K | Genome engine, Monte Carlo, counterfactual oracle, tick backtest, portfolio/risk |
| Julia | ~20K | BH physics, GARCH/OU stochastic, Bayesian optimization, FCI causal discovery |
| C/C++ | ~15K | Fast indicators (20 signals), L3 orderbook (AVX2), SIMD matrix ops |
| R | ~10K | HMM, White's Reality Check, WFA, portfolio optimization, ggplot2 |
| Zig | ~8K | ITCH 5.0 decoder, low-latency orderbook |
| SQL | ~5K | SQLite schema (16 migrations), DuckDB analytics, BH UDFs |
| **Total** | **~410K** | |

---

## Performance Notes

The backtest engine runs the identical BH physics used in live trading, including GARCH vol scaling, OU overlay, and Mayer Multiple dampening — no lookahead, no future data.

Key backtest findings (2021–2026, 19 crypto pairs):
- **2021:** Strong bull market; BH activations clustered around breakouts — high hit rate
- **2022:** Bear market; BH_FORM = 1.92 significantly reduced drawdown vs lower thresholds — fewer false activations
- **2024:** +26% year, driven by BTC and SOL regime; BTC lead signal added meaningful edge to altcoin timing
- **Full period CAGR:** -11% (crypto bear market dominated); 2024 standalone: +26%
- **Monte Carlo (1,000 paths):** Median outcome captures the distribution of sequential trade ordering; blowup rate (equity < 10% of peak) computed across all paths
- **Key insight:** The GARCH vol targeting + CORR=0.25 portfolio construction significantly reduces tail risk vs equal-weight allocation

LARSA v1 baseline (ES futures, QuantConnect): **+274%** over backtest window — the BH physics signal on a single trending instrument.

---

## BH Physics Reference

| Component | Formula | Interpretation |
|-----------|---------|----------------|
| MinkowskiClassifier | `ds² = c²dt² − dx²` | TIMELIKE (ds²>0) = ordered, causal; SPACELIKE = anomalous |
| BH Formation | `mass >= BH_FORM` | Well forms at 1.92; EMA asymptotes to 2.0 |
| Mass Accrual | `mass = 0.97×mass + 0.03×min(2, 1+ctl×0.1)` | Consecutive timelike bars build conviction |
| Mass Decay | `mass *= 0.924` | Noise bars bleed mass away |
| Hawking Monitor | `T_H = 1/(8πM)` | Cold well = stable; hot well = reduce size |
| Delta Score | `tf_score × mass × ATR` | Expected dollar move — the allocation signal |
| OU Overlay | `dX = θ(μ−X)dt + σdW` | Mean reversion on flat BH; 8% equity |
| Mayer Dampener | `scale = min(1, 2×MA200/price)` | Reduces size when price is extended |
| BTC Lead | `alt_score *= (1 + btc_active × 0.3)` | BTC activation boosts correlated alts |

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CF` (base) | 0.001–0.025 | Minkowski speed of light — calibrated per instrument |
| `BH_FORM` | 1.92 | Mass threshold for BH activation; higher = fewer but stronger signals |
| `CORR` | 0.25 | Cross-asset correlation for portfolio risk calculation |
| `GARCH_TARGET_VOL` | 1.20 | Target annualized volatility (120%) |
| `OU_FRAC` | 0.08 | OU mean-reversion position fraction |
| `DELTA_MAX_FRAC` | 0.40 | Max single-instrument allocation |
| `MIN_TRADE_FRAC` | 0.03 | Minimum equity shift to trigger rebalance |
| `STALE_15M_MOVE` | 0.001 | Cut losers moving < 0.1% per 15m bar |
| `MIN_HOLD` | 6 | Minimum bars before direction reversal |

---


