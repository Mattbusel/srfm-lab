# Rust Crates Reference

All Rust crates in `crates/` and `idea-engine/rust/`. Together they form the
performance-critical computation layer: genome evolution, parallel backtesting, Monte
Carlo simulation, portfolio optimization, risk analytics, and execution optimization.

~24K lines of Rust. All crates compile with `cargo build --release` from the workspace
root. Test coverage via `cargo test --workspace`.

---

## Workspace Layout

```
Cargo.toml                    -- workspace manifest
crates/
  counterfactual-engine/      -- parameter sensitivity analysis
  data-pipeline/              -- market data ingest and normalization
  execution-optimizer/        -- Almgren-Chriss optimal execution
  fix-engine/                 -- FIX protocol codec
  fractal-engine/             -- Hurst exponent, fractal dimension, wavelet
  idea-genome-engine/         -- NSGA-II multi-objective genome evolution
  larsa-core/                 -- Rust port of Python BH physics engine
  larsa-wasm/                 -- WebAssembly build of larsa-core
  microstructure-engine/      -- adverse selection, bid-ask decomposition
  monte-carlo-engine/         -- 10K-path regime-aware Monte Carlo
  network-graph/              -- counterparty network (DebtRank)
  options-engine/             -- Black-Scholes, implied vol, Greeks
  order-flow-engine/          -- order book imbalance prediction
  parallel-backtest/          -- distributed backtest via channels
  portfolio-engine/           -- Ledoit-Wolf, HRP, Black-Litterman
  real-time-analytics/        -- rolling Sharpe, max DD, P&L
  regime-detector/            -- 6-class regime classifier
  rl-exit-optimizer/          -- Q-learning exit strategies
  risk-engine/                -- VaR/CVaR, Greeks, stress scenarios
  signal-evolution/           -- signal mutation and crossover operators
  smart-order-router/         -- spread-tier order routing
  srfm-tools/                 -- CLI utilities
  tick-backtest/              -- event-driven tick-level simulator
idea-engine/rust/
  genome-engine/              -- NSGA-II genome evolution (IAE-facing binary)
  counterfactual-oracle/      -- counterfactual scenario runner (IAE-facing)
```

---

## Core Trading Crates

### larsa-core

Byte-for-byte Rust port of `lib/srfm_core.py`. Produces identical outputs for identical
input sequences. Used by `tick-backtest` and `parallel-backtest` for high-throughput
historical evaluation.

```
crates/larsa-core/src/
  lib.rs       -- public API: MinkowskiClassifier, BlackHoleDetector
  bh_state.rs  -- mass accumulation state machine
  agents.rs    -- agent_d3qn, agent_ddqn, agent_td3qn signal functions
  regime.rs    -- MarketRegime detection (BULL/BEAR/SIDEWAYS/HIGH_VOL)
  features.rs  -- 31-element feature vector construction
```

Key structs:
- `BHState` - mass, active flag, proper time, CF
- `MinkowskiResult` - ds2, is_timelike, proper_time
- `AgentOutput` - d3qn, ddqn, td3qn signals, composite

Verification: a test fixture runs both the Python and Rust engines on the same 10K bar
sequence and asserts output agreement to 1e-10.

### tick-backtest

Event-driven tick-level backtesting engine. Processes raw trades/quotes, aggregates
into OHLCV bars, runs the BH engine, and records fills with realistic slippage.

```
crates/tick-backtest/src/
  lib.rs         -- public API
  simulator.rs   -- event loop: tick -> bar -> signal -> order -> fill
  matcher.rs     -- limit order matching with spread cost
  parallel.rs    -- rayon work-stealing parallel parameter sweeps
  fee_model.rs   -- per-instrument maker/taker fee schedule
```

Usage:
```bash
cargo run -p tick-backtest -- sweep \
  --data-dir data/ --sym BTC --n-trials 1000 \
  --cf-min 0.30 --cf-max 0.90 --output results/sweep.csv
```

The `sweep` command runs N independent backtests in parallel (rayon thread pool) across
a grid of CF values. Each trial uses a different random seed for Monte Carlo sampling.

### monte-carlo-engine

10,000-path regime-aware Monte Carlo simulation. Resamples from historical trade
sequences, preserving regime structure so that bull/bear/sideways blocks appear at
historically plausible frequencies.

```
crates/monte-carlo-engine/src/
  lib.rs       -- MonteCarloEngine, PathConfig, PathResult
  sampler.rs   -- block bootstrap, regime-aware resampling
  paths.rs     -- GBM, GARCH-filtered, Heston path generation
  analyzer.rs  -- drawdown, VaR, Sharpe quantiles across paths
```

Key method:
```rust
let engine = MonteCarloEngine::new(config);
let results = engine.run(historical_trades, n_paths=10_000);
// results.sharpe_p5 -- 5th percentile Sharpe across paths
// results.blowup_rate -- fraction of paths with drawdown > threshold
// results.median_equity_12m -- median 12-month equity
```

Called from `tools/crypto_backtest_mc.py` via subprocess. Input: historical trade CSV.
Output: distribution statistics as JSON.

---

## IAE-Facing Crates

### idea-genome-engine

NSGA-II (Non-dominated Sorting Genetic Algorithm II) multi-objective genome evolution.
Optimizes BH physics parameters across two objectives simultaneously: maximize Sharpe
ratio and minimize maximum drawdown.

```
crates/idea-genome-engine/src/
  lib.rs        -- public API
  genome.rs     -- genome encoding: 30+ parameters as f64 vector
  nsga2.rs      -- NSGA-II: fast non-dominated sort, crowding distance
  evaluator.rs  -- parallel backtest evaluation via rayon
  operators.rs  -- SBX crossover, polynomial mutation, tournament selection
  constraints.rs -- hard constraints: BH_FORM in [1.80, 2.00], CF > 0
```

Genome parameters include: CF per timeframe, BH_FORM, MIN_HOLD_BARS, OU_FRAC,
GARCH_TARGET_VOL, CORR thresholds, blocked entry hours, boost hours, per-instrument
CF scales, and winner protection threshold.

Population: 100 genomes. Generations: 100. Total evaluations: ~10,000 backtests,
each running the full 2021-present history. Runtime: 8-15 minutes on 8-core machine.

```bash
cd idea-engine/rust
cargo build --release
./target/release/genome-engine \
  --generations 100 --pop-size 200 \
  --data-dir ../../data/ \
  --output ../../config/genome_result.json
```

### counterfactual-engine

Parameter sensitivity analysis using Latin Hypercube Sampling (LHS) and Sobol
sequences. Answers: "if I perturb CF by 10%, how much does Sharpe change?"

```
crates/counterfactual-engine/src/
  lib.rs          -- CounterfactualEngine, SensitivityResult
  sampler.rs      -- LHS sampler, Sobol sequence generator, neighborhood sampler
  sensitivity.rs  -- SobolAnalyzer (first/total order indices), MorrisScreening
  main.rs         -- CLI entry point
```

Workflow:
1. Receive best genome from `idea-genome-engine`
2. Sample N perturbations around it using LHS
3. Evaluate each perturbation via `tick-backtest`
4. Compute Sobol first-order sensitivity indices
5. Return parameters ranked by sensitivity (most sensitive = needs tightest tuning)

Output drives the IAE loop's confidence in parameter robustness: low sensitivity to
a parameter means the genome is stable there.

---

## Portfolio and Risk Crates

### portfolio-engine

Portfolio construction using three methods:

**Ledoit-Wolf shrinkage** (`src/ledoit_wolf.rs`):
- Estimates the covariance matrix with shrinkage toward scaled identity
- Reduces estimation error for small sample sizes relative to N instruments
- Used when N instruments > 15 and sample length < 252 bars

**Hierarchical Risk Parity** (`src/hierarchical.rs`):
- Builds a linkage matrix from the correlation matrix
- Allocates via recursive bisection: equal risk contribution per cluster
- No matrix inversion required; works with rank-deficient covariance matrices

**Black-Litterman** (`src/black_litterman.rs`):
- Prior: market-cap-weighted equilibrium returns
- Views: IAE-generated signals as private views with confidence weights
- Posterior: Bayesian blend of prior and views
- Output: view-adjusted expected returns fed into mean-variance optimizer

```bash
cargo run -p portfolio-engine -- optimize \
  --returns data/returns.csv --method hrp \
  --output results/weights.json
```

### risk-engine

VaR, CVaR, Greeks, and historical stress scenarios.

```
crates/risk-engine/src/
  var.rs     -- parametric VaR, historical VaR, Monte Carlo VaR
  cvar.rs    -- expected shortfall (CVaR / ES)
  greeks.rs  -- delta, gamma, vega, theta, rho for options
  stress.rs  -- historical scenario shocks (COVID, LUNA, FTX, Apr-2026)
  limits.rs  -- position limit checker against risk_limits.yaml
```

Stress scenarios:
- COVID (Feb-Mar 2020): -50% equity, +80% VIX
- LUNA collapse (May 2022): -99% LUNA, -60% BTC, -70% ETH
- FTX collapse (Nov 2022): -75% FTX, -30% BTC sector
- Apr-2026: most recent scenario, loaded from historical data

### execution-optimizer

Almgren-Chriss optimal execution for large orders. Minimizes the trade-off between
market impact cost and timing risk.

```
crates/execution-optimizer/src/
  almgren_chriss.rs  -- continuous-time optimal liquidation trajectory
  twap.rs            -- uniform time slicing
  vwap.rs            -- volume-proportional slicing
  impact.rs          -- linear and square-root market impact models
  schedule.rs        -- discrete execution schedule given trajectory
```

For the live trader, this applies to the equity instruments where individual orders
exceed 0.5% of the instrument's ADV. Crypto orders are small enough that Almgren-Chriss
is not needed.

---

## Signal and Analysis Crates

### fractal-engine

Fractal geometry analysis for regime detection and signal extraction.

```
crates/fractal-engine/src/
  hurst.rs              -- Hurst exponent (R/S analysis, DFA, wavelet)
  fractal_dimension.rs  -- box-counting and variation method fractal dimension
  wavelet.rs            -- Haar and Daubechies wavelet decomposition
  regime_detector.rs    -- fractal-based regime classification
  similarity.rs         -- DTW similarity between price subsequences
  pattern_library.rs    -- pattern matching against historical motifs
```

Hurst exponent interpretation:
- H > 0.5: persistent (trending) process -- BH formation expected
- H < 0.5: anti-persistent (mean-reverting) process -- OU overlay dominant
- H = 0.5: random walk -- no signal edge

The Hurst estimate feeds into the `regime-detector` crate as a feature alongside
the BH mass to improve regime classification.

### regime-detector

6-class regime classifier that combines:
- BH mass and active flags (from larsa-core)
- Hurst exponent (from fractal-engine)
- Macro factors: VIX level, yield curve slope, DXY momentum
- On-chain: BTC MVRV z-score, exchange net flows

Output classes:
1. STRONG_BULL -- BH active, H > 0.6, positive macro
2. WEAK_BULL -- BH active, H 0.5-0.6
3. SIDEWAYS -- BH inactive, H near 0.5
4. WEAK_BEAR -- BH active on short TF only, H > 0.5 (downtrend)
5. STRONG_BEAR -- BH active on all TFs, negative macro
6. HIGH_VOL -- any regime with VIX > 35 or realized vol > 2x EMA

The 6-class output drives agent signal weights in the live trader and position sizing
multipliers in the portfolio engine.

### order-flow-engine

Microstructure-level order flow prediction.

```
crates/order-flow-engine/src/
  imbalance.rs    -- bid-ask volume imbalance (Easley et al. PIN model)
  toxicity.rs     -- VPIN: volume-synchronized probability of informed trading
  pressure.rs     -- trade pressure: buy-initiated vs sell-initiated
  prediction.rs   -- short-term direction forecast from order flow features
```

Inputs are L2 order book snapshots from the market-data service. Output is a
directional signal with a 1-5 bar horizon, used as an intra-bar timing overlay.

### microstructure-engine

```
crates/microstructure-engine/src/
  adverse_selection.rs    -- Glosten-Milgrom adverse selection component
  bid_ask_decomp.rs       -- Roll's bid-ask spread decomposition
  price_impact.rs         -- Kyle's lambda estimation (matches Python MI proxy)
  amihud.rs               -- Amihud illiquidity ratio
```

The Kyle's lambda estimate from this crate and the `MI_norm` component of the
quaternion navigation layer are computed by different methods but measure the same
quantity. Comparing them is a planned validation step.

---

## Infrastructure Crates

### fix-engine

FIX 4.4 / FIX 5.0 protocol parser and encoder for direct broker connectivity.

```
crates/fix-engine/src/
  codec.rs          -- FIX message serialization/deserialization
  parser.rs         -- tag-value pair parser with field validation
  session.rs        -- FIX session (logon, heartbeat, sequence numbers)
  store.rs          -- message store for replay and gap fill
  types.rs          -- FIX field types (side, order type, TIF, etc.)
  messages/
    new_order_single.rs       -- order submission
    order_cancel_request.rs   -- cancel
    execution_report.rs       -- fill parsing
    market_data_request.rs    -- market data subscription
    market_data_snapshot.rs   -- L2 snapshot parsing
```

Used for direct connectivity to prime brokers that do not provide REST/WebSocket APIs.
Not used in the current Alpaca-based paper trading setup.

### data-pipeline

Market data ingest and normalization pipeline.

```
crates/data-pipeline/src/
  ohlcv.rs       -- OHLCV bar struct and validation
  normalizer.rs  -- price/volume normalization, split adjustment
  splitter.rs    -- train/test/validation split with purging
  features.rs    -- feature construction from OHLCV
  indicators.rs  -- fast technical indicator computation
  lib.rs         -- DataPipeline: ingest -> normalize -> split -> features
```

The `splitter.rs` implements embargo windows and purged K-fold splitting for the
walk-forward validation engine, matching the CPCV implementation in the IAE.

### network-graph

Counterparty network analysis using DebtRank contagion model.

```
crates/network-graph/src/
  graph.rs        -- directed weighted graph representation
  debtrank.rs     -- DebtRank algorithm: recursive impact estimation
  centrality.rs   -- betweenness, eigenvector, PageRank centrality
  contagion.rs    -- network-wide shock propagation simulation
```

Applied to cross-asset correlation networks: if BTC collapses, DebtRank estimates
the cascading impact on altcoins given their current correlation weights.

### larsa-wasm

WebAssembly build of larsa-core for browser-based backtesting in the Spacetime Arena
web UI. Compiles via `wasm-pack`:

```bash
cd crates/larsa-wasm
wasm-pack build --target web
```

Output goes to `spacetime/web/pkg/` and is imported by the React Spacetime Arena.
Enables in-browser BH signal replay without server round-trips.

### rl-exit-optimizer

Q-learning exit strategy optimizer. Learns a state-action policy for exit timing
using the BH state, GARCH vol, and unrealized P&L as state features.

```
crates/rl-exit-optimizer/src/
  state.rs       -- exit state: (bh_mass, garch_vol, unrealized_pnl, bars_held)
  agent.rs       -- tabular Q-agent with epsilon-greedy exploration
  trainer.rs     -- episode simulation on historical trades
  policy.rs      -- greedy policy extraction for deployment
```

The trained policy is exported as a lookup table. In the live trader, it replaces the
fixed `MIN_HOLD_BARS` rule with a state-dependent exit decision.
Status: experimental, not deployed in LARSA v17.

---

## Shared Patterns

All crates follow a consistent structure:

1. `lib.rs` exports the public API (no CLI logic in library code)
2. `main.rs` provides a CLI binary that calls the library
3. `src/` modules are named by the computation they perform, not by layer
4. `benches/` contains Criterion benchmarks for hot paths
5. Rayon is the standard parallelism primitive (work-stealing thread pool)
6. Serde + serde_json for all external I/O
7. No `unwrap()` in library code; errors use `anyhow::Result`

---

## Running Tests

```bash
# All crates
cargo test --workspace

# Single crate
cargo test -p larsa-core
cargo test -p tick-backtest

# With output (useful for performance tests)
cargo test -p monte-carlo-engine -- --nocapture

# Benchmarks (requires nightly or Criterion)
cargo bench -p portfolio-engine
```

---

## Key Parameters by Crate

| Crate | Key Parameter | Default | Tuned by |
|---|---|---|---|
| larsa-core | BH_MASS_THRESH | 1.92 | Fixed |
| idea-genome-engine | population size | 100 | Config |
| idea-genome-engine | generations | 100 | Config |
| monte-carlo-engine | n_paths | 10,000 | Config |
| tick-backtest | n_trials (sweep) | 1,000 | Config |
| counterfactual-engine | n_samples (LHS) | 500 | Config |
| portfolio-engine | shrinkage target | identity | Ledoit-Wolf |
| risk-engine | VaR confidence | 0.99 | Config |
| execution-optimizer | risk aversion | 1e-6 | Calibrated |
