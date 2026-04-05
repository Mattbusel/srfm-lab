# Idea Automation Engine (IAE) Architecture

## SRFM — Special Relativistic Financial Mechanics

> "The IAE is a perpetual hypothesis machine. It never sleeps, never satisfies
> itself with a single explanation, and never promotes a pattern to live
> trading without first running it through a statistical gauntlet designed
> to destroy it."

---

## Table of Contents

1. [Philosophy and Design Goals](#1-philosophy-and-design-goals)
2. [System Overview — The Autonomous Loop](#2-system-overview)
3. [Ingestion Pipeline — 4 Stages](#3-ingestion-pipeline)
4. [Miners](#4-miners)
5. [Statistical Filters](#5-statistical-filters)
6. [Genome Evolution — NSGA-II](#6-genome-evolution)
7. [Hypothesis Generator](#7-hypothesis-generator)
8. [Walk-Forward Validation — CPCV](#8-walk-forward-validation)
9. [Causal Discovery](#9-causal-discovery)
10. [Regime Oracle](#10-regime-oracle)
11. [Signal Library](#11-signal-library)
12. [Academic Miner Pipeline](#12-academic-miner-pipeline)
13. [Event Bus Architecture](#13-event-bus-architecture)
14. [Database Schema](#14-database-schema)
15. [Serendipity Engine](#15-serendipity-engine)
16. [The Feedback Loop](#16-the-feedback-loop)

---

## 1. Philosophy and Design Goals

The Idea Automation Engine (IAE) was built to solve a specific problem: human
researchers can only evaluate a handful of hypotheses per week. Markets
generate thousands of exploitable edge opportunities per year, but the vast
majority expire before a human can notice, formalise, test, and deploy them.

### Core Principles

**Automation over intuition**: Every edge must be formally specified,
statistically tested, and independently validated before it influences live
capital. Human intuition seeds the system but cannot bypass the statistical
gatekeepers.

**Statistical conservatism**: The system is designed to reject approximately
90% of the patterns it discovers. A low false-discovery rate is worth more
than a high true-discovery rate when the cost of a false positive is a
drawdown in live trading.

**Evolutionary improvement**: Parameters are not set once and left alone.
They are treated as a living genome that evolves under competitive pressure.
The only thing that survives is what actually works out-of-sample.

**Transparency and lineage**: Every live parameter change is traceable to a
pattern, which is traceable to raw trade data, which is traceable to a
specific time window and market regime. The genealogy graph is never pruned.

---

## 2. System Overview

### The Autonomous Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIVE TRADING ENGINE                          │
│              live_trader_alpaca.py + BH Engine                  │
└────────────────────┬────────────────────────────────────────────┘
                     │ trade log, equity series, regime log
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                            │
│  Load → Mine (4 miners) → Filter (bootstrap) → Store (SQLite)  │
└────────────────────┬────────────────────────────────────────────┘
                     │ MinedPattern objects (confirmed only)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                 HYPOTHESIS GENERATOR                            │
│  Templates → Bayesian scoring → Dedup → Compound generation     │
└────────────────────┬────────────────────────────────────────────┘
                     │ Hypothesis objects → idea_engine.db
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GENOME EVOLUTION                              │
│  NSGA-II multi-objective optimisation over parameter space       │
└────────────────────┬────────────────────────────────────────────┘
                     │ Pareto-optimal genomes
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                WALK-FORWARD VALIDATION                          │
│  CPCV + shadow paper trading + OOS degradation check            │
└────────────────────┬────────────────────────────────────────────┘
                     │ validated parameter sets
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LIVE FEEDBACK                                 │
│  Update live_trader_alpaca.py config in near-real-time          │
└─────────────────────────────────────────────────────────────────┘
```

The loop runs continuously. Most discovered patterns are rejected. Those that
survive become experiments. Those experiments that survive OOS validation
become live parameter updates.

---

## 3. Ingestion Pipeline

The ingestion pipeline (`idea-engine/ingestion/pipeline.py`) orchestrates four
sequential stages:

```python
class IngestionPipeline:
    """
    Stages
    ──────
    1. Load data   (backtest / live / walk-forward)
    2. Run miners  (time-of-day / regime-cluster / bh-physics / drawdown)
    3. Filter      (stationary bootstrap + BH correction)
    4. Persist     (write confirmed patterns to idea_engine.db)
    """
```

### Stage 1 — Load

Three data sources are queried:

| Loader              | Source | Data Provided |
|---------------------|--------|---------------|
| `BacktestLoader`    | Backtest results DB | All historical trades with TF score, regime, BH mass at entry |
| `LiveTradeLoader`   | `live_trades.db` from Alpaca | Real executed trades with timestamps, PnL, regime log |
| `WalkForwardLoader` | Walk-forward result files | Per-fold performance statistics |

Each loader returns a typed dataclass (`BacktestResult`, `LiveTradeData`,
`WalkForwardResult`) with normalised column names so miners can operate on
either source without special-casing.

### Stage 2 — Mine

Miners run in parallel across the loaded data. See Section 4 for full
miner descriptions.

### Stage 3 — Filter

Statistical filters eliminate spurious patterns. See Section 5.

### Stage 4 — Persist

Confirmed patterns are written to `idea_engine.db` (WAL-mode SQLite). An
event is emitted to `event_log` with full pipeline run statistics.

### Pipeline CLI

```bash
# Run all miners
python -m idea_engine.ingestion.pipeline

# Dry run (no DB writes)
python -m idea_engine.ingestion.pipeline --dry-run

# Only specific miners
python -m idea_engine.ingestion.pipeline --miners time_of_day,bh_physics

# Verbose logging
python -m idea_engine.ingestion.pipeline --verbose
```

---

## 4. Miners

Each miner implements the interface:

```python
class BaseMiner:
    def mine(self, data: Union[BacktestResult, LiveTradeData]) -> List[MinedPattern]:
        ...
```

A `MinedPattern` carries statistical metadata alongside the feature values
that define the pattern:

```python
@dataclass
class MinedPattern:
    pattern_id:      str           # UUID
    miner:           str           # miner class name
    pattern_type:    str           # 'time_of_day', 'regime_cluster', etc.
    label:           str           # human-readable
    description:     str
    feature_json:    dict          # feature values (e.g. {"hour": 14, "regime": "BULL"})
    sample_size:     int
    p_value:         float
    effect_size:     float
    win_rate:        float
    avg_pnl:         float
    avg_pnl_baseline: float        # baseline comparison group
    sharpe:          float
    status:          PatternStatus  # new | confirmed | rejected
```

### TimeOfDayMiner

Discovers performance edges by UTC hour. Groups all trades by entry hour,
computes win rate and average PnL per bucket, then compares each hour against
the baseline (all other hours combined) using a bootstrap test.

```python
# Example pattern produced
{
  "label":       "long_edge_hour_14_UTC",
  "description": "Trades entered 14:00–14:59 UTC show +0.32 Sharpe lift",
  "feature_json": {"hour": 14, "direction": "long"},
  "win_rate":    0.63,
  "avg_pnl":     245.0,
  "avg_pnl_baseline": 118.0,
  "effect_size": 0.41,            # Cohen's d
  "p_value":     0.012
}
```

This captures the well-documented "US open" and "London/NY overlap" intraday
edge — when liquidity is highest, BH momentum signals are most reliable.

### RegimeClusterMiner

Uses K-means or DBSCAN on the regime log (bull/bear/sideways/high-vol
transitions) to find performance clusters. Looks for regime sequences where
the BH engine's win rate is significantly above or below baseline.

Key features extracted:
- `regime_at_entry`: the regime classification when the trade opened
- `regime_duration`: how many bars the regime had persisted before entry
- `regime_transition`: whether the trade straddled a regime change

```python
# Example pattern produced
{
  "label":       "bull_regime_duration_gt20_edge",
  "description": "BULL regime persisting >20 bars shows 0.58 win rate vs 0.47 baseline",
  "feature_json": {"regime": "BULL", "min_duration": 20},
  "win_rate":    0.58,
  "effect_size": 0.29,
  "p_value":     0.003
}
```

### MassPhysicsMiner (BHPhysics)

Analyses the relationship between BH mass at entry and subsequent trade
outcomes. Key hypotheses this miner tests:

1. **Mass threshold**: Do trades with `bh_mass_at_entry > X` outperform?
2. **Mass velocity**: Does rapidly growing mass (second derivative positive)
   predict stronger outcomes?
3. **Timeframe agreement**: Does having all 3 TFs active vs just daily matter?
4. **Post-reform trades**: Do trades that form after a recent BH collapse
   (reform memory active) perform differently?

```python
# Example pattern produced
{
  "label":       "high_mass_entry_edge",
  "description": "Entries with bh_mass > 3.5 show Sharpe 1.82 vs 0.94 baseline",
  "feature_json": {"min_bh_mass": 3.5, "tf_score": 7},
  "sharpe":      1.82,
  "effect_size": 0.55,
  "p_value":     0.001
}
```

### DrawdownMiner

Analyses drawdown patterns to identify structural weaknesses:

1. **Regime-specific drawdowns**: Does the BH engine have outsized drawdowns
   during specific regime types?
2. **Drawdown recovery**: How long does recovery take after a drawdown > X%?
3. **Consecutive loss streaks**: Are there temporal clusters of losing trades?
4. **Correlation with VIX regimes**: Do drawdowns cluster with vol spikes?

```python
# Example pattern produced
{
  "label":       "high_vol_drawdown_risk",
  "description": "HIGH_VOL regime → 2.3× expected drawdown vs BULL",
  "feature_json": {"regime": "HIGH_VOL", "drawdown_multiplier": 2.3},
  "max_dd":      -0.118,
  "p_value":     0.008
}
```

---

## 5. Statistical Filters

### Stationary Bootstrap

The pipeline uses a stationary block bootstrap (Politis & Romano, 1994) rather
than naive resampling. Standard bootstrap breaks temporal dependencies;
the stationary bootstrap preserves autocorrelation structure by sampling
overlapping blocks of variable length.

```
n_resamples = 1000         (default)
alpha       = 0.05         (significance threshold)
min_effect  = 0.20         (minimum Cohen's d or equivalent)
```

For each candidate pattern:

```python
def filter_patterns(patterns, alpha=0.05, min_effect=0.20, n_resamples=1000):
    for pattern in patterns:
        # 1. Bootstrap null distribution
        null_dist = bootstrap_statistic(pattern.trade_pnls, n=n_resamples)

        # 2. p-value: fraction of null samples exceeding observed effect
        p_val = (null_dist >= pattern.effect_size).mean()

        # 3. Effect size gate
        if pattern.effect_size < min_effect:
            pattern.status = PatternStatus.REJECTED
            continue

        # 4. Significance gate
        if p_val > alpha:
            pattern.status = PatternStatus.REJECTED
            continue

        pattern.status = PatternStatus.CONFIRMED
```

### Benjamini-Hochberg FDR Correction

When testing multiple patterns simultaneously, the family-wise error rate
inflates. The pipeline applies Benjamini-Hochberg (BH) FDR correction:

```
Sort patterns by p_value ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
Threshold k* = max{k : p_(k) ≤ k × α / m}
Reject H_0 for all p_(i) ≤ p_(k*)
```

At `alpha = 0.05` with 100 candidate patterns, the BH correction approximately
reduces the effective threshold from 0.05 to ~0.02–0.03 depending on the
distribution of p-values. This means the bar for significance rises when
many patterns are tested — preventing the multiple comparison fallacy.

### Minimum Effect Size Gate

Statistical significance is necessary but not sufficient. A pattern with
`p = 0.001` but Cohen's `d = 0.05` is practically useless. The engine
requires `min_effect = 0.20` (a "small-medium" effect in Cohen's framework)
before a pattern is promoted.

Effect size types used:
- **Cohen's d**: for continuous PnL comparisons
- **Cliff's delta**: for ordinal or non-normal distributions
- **Eta-squared**: for multi-group regime comparisons

---

## 6. Genome Evolution

The genome engine treats every strategy parameter set as a "genome" that can
be evolved under natural selection. Parameters are encoded as a chromosome
vector; fitness is measured by backtested performance; the population evolves
through crossover and mutation.

### NSGA-II Multi-Objective Optimisation

NSGA-II (Non-Dominated Sorting Genetic Algorithm II, Deb et al. 2002) is used
because strategy optimisation inherently involves competing objectives:

| Objective       | Direction | Notes |
|-----------------|-----------|-------|
| Sharpe Ratio    | Maximise  | Risk-adjusted return |
| Max Drawdown    | Minimise  | Worst peak-to-trough loss |
| Turnover        | Minimise  | Trading costs, implementation friction |

No single solution dominates all three. NSGA-II finds the **Pareto front** —
the set of solutions where improving one objective requires worsening another.

```
Generation 0 (random population)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Non-dominated sort → Pareto fronts F1, F2, F3...       │
│  Crowding distance → diversity preservation              │
│  Selection → top 50% by rank + crowding distance        │
│  Crossover → SBX (Simulated Binary Crossover)           │
│  Mutation → Polynomial mutation                          │
│  Evaluation → Backtest each new individual               │
└─────────────────────────────────────────────────────────┘
         │
         ▼
Generation 1, 2, ... N
```

### Parameter Chromosome Encoding

Each genome encodes:

```python
{
  # BH Physics parameters
  "cf_btc":         0.005,   # speed of light per instrument
  "cf_eth":         0.007,
  "bh_form":        1.5,
  "bh_decay":       0.95,

  # Regime gates
  "bull_size":      1.0,     # max position in BULL regime
  "bear_size":      0.0,     # max position in BEAR (long-only)
  "highvol_size":   0.40,    # position cap in HIGH_VOL

  # OU overlay
  "ou_kappa":       0.15,    # mean reversion speed
  "ou_equity_frac": 0.08,    # fraction of equity for OU sleeve

  # GARCH scaling
  "target_vol":     0.90,

  # Signal weights
  "btc_lead_boost": 0.30,
  "mayer_scale_on": True,
}
```

### Mutation Operators

- **Gaussian perturbation** (continuous params): `new = old + N(0, σ)` where
  `σ = 0.05 × param_range`
- **Swap mutation** (discrete/boolean params): flip with probability `p_mut`
- **Creep mutation** (bounded params): small random steps within bounds
- **Macro-mutation**: occasional large jumps to escape local optima (10% of
  mutations)

### Crossover

Simulated Binary Crossover (SBX) for continuous parameters — generates
offspring distributed around the parents with controllable spread. Index
`η_c = 2` (moderate spread) is used.

```
offspring_1 = 0.5 × (parent_1 + parent_2) - 0.5 × β × |parent_2 - parent_1|
offspring_2 = 0.5 × (parent_1 + parent_2) + 0.5 × β × |parent_2 - parent_1|
where β is sampled from the SBX distribution with η_c
```

### OOS Degradation Guard

Before any genome is promoted to active status, its OOS (out-of-sample)
performance is compared to its IS (in-sample) fitness:

```
oos_degradation = (sharpe_IS - sharpe_OOS) / sharpe_IS

If oos_degradation > 0.40:   # >40% Sharpe degradation OOS
    genome.status = "archived"   # do not promote
```

---

## 7. Hypothesis Generator

The hypothesis generator converts `MinedPattern` objects into formal
`Hypothesis` objects — structured, testable claims about the market.

### Architecture

```
MinedPattern
     │
     ▼
Templates (route by pattern_type)
     │
     ├─► TimeOfDayTemplate → "Morning session BH entries show higher win rate"
     ├─► RegimeTemplate    → "BH signals in BULL + ctl>10 dominate"
     ├─► MassTemplate      → "High-mass entries at formation outperform"
     └─► DrawdownTemplate  → "Reduce size during HIGH_VOL regime entry"
     │
     ▼
HypothesisScorer (Bayesian-adjusted composite score)
     │
     ▼
Deduplicator (semantic similarity vs existing hypotheses in DB)
     │
     ▼
CompoundGenerator (pairwise combinations of high-scoring novel hypotheses)
     │
     ▼
HypothesisPrioritizer (re-rank pending queue)
     │
     ▼
HypothesisStore → idea_engine.db (hypotheses table)
```

### Bayesian Prior and Posterior

Each hypothesis begins with `prior_prob = 0.5`. The prior updates as
experiments are run:

```
posterior = (likelihood × prior) / P(evidence)
```

Bayesian updating uses a conjugate Beta prior over the hypothesis probability:

```
prior:     Beta(α₀, β₀)   where α₀ = β₀ = 1 (uniform)
after n experiments with k confirmations:
posterior: Beta(α₀ + k, β₀ + n - k)
E[posterior] = (α₀ + k) / (α₀ + β₀ + n)
```

This means a hypothesis that has been tested 5 times and confirmed 4 times
has posterior `5/7 ≈ 0.71`, regardless of the priors of the individual tests.

### Compound Hypotheses

When two high-scoring hypotheses have complementary parameters, the generator
produces a compound hypothesis with a synergy bonus:

```python
sharpe_delta_compound = min(
    (h1.predicted_sharpe_delta + h2.predicted_sharpe_delta) / 2 * 1.1,  # 10% synergy
    1.5,   # cap
)
```

Compound hypotheses inherit the instruments union of their components and
prefix any conflicting parameter keys with the component type name to avoid
silent clobbers.

---

## 8. Walk-Forward Validation

### Combinatorial Purged Cross-Validation (CPCV)

Standard k-fold cross-validation fails for financial time series because:
1. Data leakage: adjacent folds share autocorrelated observations
2. No purging: trades that straddle fold boundaries contaminate both sets
3. No embargo: observations just outside a fold are still highly correlated

CPCV (De Prado, *Advances in Financial Machine Learning*, 2018) addresses all
three:

```
Combinatorial split of T observations into N groups of size T/N

For each combination of k test groups (k < N):
  training set = observations NOT in test groups, with:
    purge_bars   = embargo buffer before each test window
    embargo_bars = embargo buffer after each test window

OOS Sharpe = average Sharpe across all C(N,k) combinations
```

Default parameters:
```
N = 10 folds
k = 2 test groups per split
purge_bars = 5 (∼5 days lookback exclusion)
embargo_bars = 3 (3-day post-test embargo)
```

This generates `C(10,2) = 45` independent OOS estimates. The distribution of
these 45 Sharpe values characterises model robustness. A genuinely robust
strategy shows a tight distribution; an overfitted strategy shows high
variance.

### Shadow Strategies

Active genomes run in parallel as shadow (paper) strategies:

```
Live Strategy (real capital)
      │
      ├─► Shadow Variant A (param mutation ±5%)
      ├─► Shadow Variant B (param mutation ±10%)
      └─► Shadow Variant C (alternate CF values)
```

Each shadow variant tracks `realized_pnl`, `realized_sharpe`, and
`realized_dd` in real-time on paper fills. If a shadow variant consistently
outperforms the live strategy, it triggers a hypothesis for a parameter
update experiment.

```python
# shadow_variants table
{
  "genome_id":       42,
  "label":           "cf_btc_bump_5pct",
  "params_delta":    {"cf_btc": 0.00525},   # 5% bump
  "realized_sharpe": 1.84,                   # vs live 1.71
  "status":          "running"
}
```

---

## 9. Causal Discovery

### Granger Causality Network

Granger causality tests whether the past values of variable X improve
predictions of variable Y beyond Y's own past values:

```
H₀: X does not Granger-cause Y
F-test: restricted model (AR of Y) vs unrestricted (AR of Y + lags of X)
```

The engine runs pairwise Granger tests across the 19 instruments at lags
1, 5, 10, 20 bars, producing a directed graph where an edge `X → Y` means
X's history helps predict Y's future.

```
Instruments (19):
  Crypto:    BTC, ETH, SOL
  Equity:    ES, NQ, QQQ, SPY
  Energy:    CL, NG
  Metals:    GC, SI
  Bonds:     ZB, ZN
  Forex:     EURUSD, GBPUSD, USDJPY
  Vol:       VIX
  Crypto alt: YM, RTY
```

### PC Algorithm for DAG Construction

The PC algorithm (Spirtes, Glymour & Scheines, 2000) uses conditional
independence tests to identify the causal skeleton:

```
1. Start with fully connected undirected graph
2. Remove edges X -- Y where X ⊥ Y | Z  (conditional independence)
3. Orient colliders: X → Z ← Y where X ⊥ Y
4. Apply Meek rules to orient remaining edges
```

The resulting DAG provides a causal map of which instruments lead others.
Key findings inform the BTC lead signal and cross-asset filters.

### Edge Meanings

An edge `BTC → ETH (lag=4, strength=0.72, p=0.001)` means:
- BTC's value 4 bars ago has a statistically significant (p<0.001) predictive
  relationship with ETH's current value
- This relationship explains 72% of the predictable variance above baseline
- The IAE will boost ETH signals when BTC BH is active at the corresponding
  lag

These edges are stored in `causal_edges` table and updated whenever a new
DAG computation completes via the `causal.dag.updated` event.

---

## 10. Regime Oracle

The regime oracle classifies the current market state into one of 6 regimes
that gate different strategy components.

### 6 Regime Classes

| Regime        | Definition | BH Engine Behaviour |
|---------------|-----------|---------------------|
| **BULL_TREND**    | EMA12 > EMA26 > EMA50 > EMA200, ADX > 25 | Full allocation, pos_floor active |
| **BEAR_TREND**    | EMA12 < EMA26 < EMA50, ADX > 25 | Long-only mode: flat or minimal hedge |
| **MEAN_REVERT**   | ADX < 20, price oscillating near BB mid | Reduce BH allocation, boost OU overlay |
| **VOL_SPIKE**     | ATR ratio > 1.5 (current ATR / 50-bar avg) | Scale back to 40% max allocation |
| **RISK_OFF**      | VIX > 25 OR bond-equity correlation flips | Defensive sizing: max 30% |
| **ACCUMULATION**  | Low ADX, persistent TIMELIKE bars, low ATR | Favour high-mass BH entries |

### Regime Detection Code

```python
def _classify_regime_simple(bh_dir: int, bh_mass: float, atr_ratio: float) -> str:
    if atr_ratio >= 1.5:
        return "HIGH_VOL"
    if bh_dir > 0 and bh_mass > 1.5:
        return "BULL"
    if bh_dir < 0 and bh_mass > 1.5:
        return "BEAR"
    return "SIDEWAYS"
```

The simple classifier in the backtest engine uses BH mass and direction as
proxies. The production `RegimeDetector` class adds:
- 4-indicator composite score (MACD, EMA alignment, ADX, ATR ratio)
- Confidence weighting (0–1 float, stored in `regime_confidence`)
- Transition detection (identifies *when* regimes change, not just current state)

### Regime Gating in Position Sizing

```python
# From TF_CAP, further gated by regime
effective_ceiling = TF_CAP[tf_score] × regime_size_multiplier[regime]

# Example regime multipliers
regime_size_multiplier = {
    "BULL":     1.0,
    "SIDEWAYS": 0.6,
    "HIGH_VOL": 0.4,
    "BEAR":     0.0,   # long-only → flat
}
```

---

## 11. Signal Library

The IAE maintains a library of 60+ signals with IC (Information Coefficient)
tracking. IC measures how well a signal predicts the direction of future
returns:

```
IC = Spearman rank correlation(signal_t, return_{t+h})
```

where `h` is the forward holding period (1, 5, 20 bars).

### IC Decay Curves

Signal weight decays as IC ages:

```
IC_weight(age) = IC_recent × exp(−λ × age_days)
```

`λ` is calibrated per-signal to capture how quickly each signal's predictive
power degrades. Signals with high `λ` are "perishable" — they must be
refreshed frequently. Signals with low `λ` provide durable alpha.

| Signal Category     | Example Signals | Typical IC | Decay Half-Life |
|--------------------|-----------------|------------|----------------|
| BH Mass Features   | bh_mass, ctl, tf_score | 0.08–0.15 | 30–60 days |
| Regime Transitions | regime_change, bull_duration | 0.05–0.12 | 45–90 days |
| Time-of-Day        | hour, session_overlap | 0.03–0.08 | 180+ days |
| Causal Lead        | btc_bh_active, es_momentum | 0.06–0.14 | 20–40 days |
| Volatility         | atr_ratio, vix_level | 0.04–0.10 | 15–30 days |

Signals with IC < 0.02 over a rolling 90-day window are automatically
retired from the active library and flagged for review.

---

## 12. Academic Miner Pipeline

The academic miner continuously ingests research papers and extracts
actionable hypotheses.

### Ingestion Sources

- **arXiv**: Papers tagged `q-fin.TR`, `q-fin.ST`, `cs.LG` (quantitative
  finance + ML for trading)
- **SSRN**: Working papers in asset pricing, market microstructure
- **Internal PDFs**: Research notes, strategy documents

### Relevance Scoring

Papers are scored for relevance using a two-stage filter:

```python
# Stage 1: keyword filter
keywords = [
  "momentum", "trend following", "mean reversion", "black hole",
  "spacetime", "regime", "volatility targeting", "cross-asset",
  "causal", "information theory", "entropy", "fractal", "Hawking",
  "GARCH", "stochastic volatility", "Kelly criterion"
]

# Stage 2: embedding similarity to existing confirmed patterns
relevance_score = cosine_similarity(paper_embedding, pattern_embeddings.mean())
```

Papers with `relevance_score > 0.65` are fully processed; those scoring
0.45–0.65 are queued for human review.

### Idea Extraction

For relevant papers, an LLM extracts structured ideas:

```python
# academic_ideas table entry
{
  "title":       "Momentum decay modulated by volatility regime",
  "description": "Paper shows momentum alpha decays faster in high-vol regimes. Implies dynamic BH_DECAY should increase when ATR ratio > 1.5",
  "applicable_to": "bh_physics,regime",
  "implementation_notes": "Add regime-conditional BH_DECAY: normal=0.95, high_vol=0.85",
  "priority":   3,
  "status":     "backlog"
}
```

Extracted ideas feed directly into the hypothesis generator as "academic"
source patterns, with a prior probability boost (`prior_prob = 0.65` vs the
standard `0.50`) because academic findings have survived peer review.

---

## 13. Event Bus Architecture

All IAE modules communicate asynchronously through a pub/sub event bus running
on port **:8768** (Go HTTP service).

### Bus Implementation

```
idea-engine/bus/
├── main.go          # HTTP server, subscription management
├── router.go        # Topic routing, fan-out delivery
├── persistence.go   # Message persistence (replay capability)
├── topics.go        # Canonical topic name constants
└── adapters/        # Language adapters (Python, Go)
```

### Topic Definitions

```go
const (
    TopicPatternsDiscovered   = "patterns.discovered"
    TopicHypothesesCreated    = "hypotheses.created"
    TopicGenomeEvaluated      = "genome.evaluated"
    TopicShadowCycleComplete  = "shadow.cycle.complete"
    TopicCounterfactualDone   = "counterfactual.done"
    TopicAcademicIdeaExtracted = "academic.idea.extracted"
    TopicSerendipitySurprise  = "serendipity.surprise"
    TopicCausalDagUpdated     = "causal.dag.updated"
    TopicExperimentCompleted  = "experiment.completed"
)
```

### Message Flow Diagram

```
ingestion/pipeline.py
    │ patterns.discovered
    ▼
hypothesis/generator.py
    │ hypotheses.created
    ▼
genome/engine.py
    │ genome.evaluated
    ▼
shadow runner
    │ shadow.cycle.complete
    ▼
live-feedback/updater.py
    │ (writes config files)
    ▼
live_trader_alpaca.py reloads config
```

### HTTP API

```
POST /publish          # Publish a message to a topic
GET  /subscribe/:topic # SSE stream for a topic
GET  /topics           # List all registered topics
GET  /health           # Service health check
```

### Persistence and Replay

The bus persists all messages to a ring buffer. On restart, subscribers can
request replay from a given timestamp — this ensures no events are lost if
a module is temporarily offline.

---

## 14. Database Schema

The IAE uses a single SQLite database (`idea_engine.db`) in WAL (Write-Ahead
Logging) mode for concurrent read access from the dashboard and pipeline.

### Core Tables

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
```

#### patterns

```sql
CREATE TABLE patterns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source           TEXT    NOT NULL,      -- 'backtest', 'live', 'walk_forward'
    miner            TEXT    NOT NULL,      -- miner class name
    pattern_type     TEXT    NOT NULL,      -- 'time_of_day', 'regime_cluster', ...
    label            TEXT    NOT NULL,
    feature_json     TEXT,                  -- JSON feature dict
    sample_size      INTEGER NOT NULL,
    p_value          REAL,
    effect_size      REAL,
    effect_size_type TEXT,                  -- 'cohens_d', 'cliffs_delta', ...
    win_rate         REAL,
    avg_pnl          REAL,
    sharpe           REAL,
    confidence       REAL,                  -- 0-1 post-bootstrap
    status           TEXT    DEFAULT 'new'  -- new|confirmed|rejected|promoted
);
```

#### hypotheses

```sql
CREATE TABLE hypotheses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    prior_prob      REAL    DEFAULT 0.5,
    posterior_prob  REAL,
    status          TEXT    DEFAULT 'open',  -- open|testing|confirmed|refuted|parked
    priority        INTEGER DEFAULT 5,        -- 1=highest, 10=lowest
    source_pattern_ids TEXT,                  -- JSON list of pattern IDs
    genome_id       INTEGER REFERENCES genomes(id)
);
```

#### genomes

```sql
CREATE TABLE genomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    parent_id       INTEGER REFERENCES genomes(id),
    generation      INTEGER DEFAULT 0,
    params_json     TEXT    NOT NULL,        -- full parameter dict
    fitness_sharpe  REAL,
    fitness_cagr    REAL,
    fitness_dd      REAL,
    composite_score REAL,
    is_oos_sharpe   REAL,                    -- OOS Sharpe ratio
    oos_degradation REAL,                    -- (IS-OOS)/IS
    status          TEXT    DEFAULT 'candidate'  -- candidate|active|archived
);
```

#### experiments

```sql
CREATE TABLE experiments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    genome_id     INTEGER REFERENCES genomes(id),
    design        TEXT    NOT NULL,  -- 'backtest', 'paper', 'live', 'shadow_ab'
    config_json   TEXT,              -- full config snapshot
    result_json   TEXT,              -- result summary
    p_value       REAL,
    effect_size   REAL,
    outcome       TEXT               -- confirmed|refuted|inconclusive|pending
);
```

#### causal_edges

```sql
CREATE TABLE causal_edges (
    cause_type  TEXT    NOT NULL,   -- 'pattern', 'hypothesis', 'factor'
    cause_id    INTEGER NOT NULL,
    effect_type TEXT    NOT NULL,
    effect_id   INTEGER NOT NULL,
    method      TEXT    DEFAULT 'granger',  -- 'granger', 'pc', 'manual', 'llm'
    strength    REAL,               -- 0-1
    lag_bars    INTEGER
);
```

#### academic_papers + academic_ideas

Track ingested research papers and the structured ideas extracted from them.
`academic_ideas.hypothesis_id` links each idea directly to the hypothesis
it spawned.

### Genealogy Tables

`genealogy_nodes` and `genealogy_edges` store a directed graph connecting
every entity in the system:

```
paper_idea → hypothesis → experiment → genome → shadow_variant → live config
```

This lineage is immutable — every change to live parameters can be traced
back through the graph to the raw trade data and paper that inspired it.

---

## 15. Serendipity Engine

The serendipity engine generates novel signal ideas by applying domain
analogies and random mutations — structured randomness in hypothesis space.

### Domain Analogy Mappings (Physics → Finance)

| Physics Concept      | Financial Mapping                        |
|---------------------|------------------------------------------|
| Hawking radiation   | Market "evaporation" of momentum wells  |
| Gravitational lensing | Volume-weighted price distortion        |
| Proper time dilation | Slow-moving convictions persist longer  |
| Event horizon       | Critical threshold beyond which reversals are unlikely |
| Schwarzschild radius | BH_FORM threshold                       |
| Orbital decay       | BH mass decay with BH_DECAY coefficient |
| Frame dragging      | Larger assets (BTC) bending smaller asset trajectories |

The serendipity engine can propose new mappings:

```python
# Example: "If BH mass is an analogue of gravitational mass,
#  what is the financial equivalent of tidal forces?"
# → Differential BH mass between correlated instruments
# → New signal: (bh_mass_btc - bh_mass_eth) as spread indicator
```

### Mutation Engine

The mutation engine randomly perturbs existing confirmed hypotheses:

```python
mutations = [
    "Change instrument scope",        # Apply BTC hypothesis to ETH
    "Invert the signal",               # Fade high-mass instead of following it
    "Add regime conditioning",         # Only apply in BULL regime
    "Change timeframe",                # Test on daily instead of hourly
    "Combine with existing signal",    # AND with Mayer < 2
    "Change direction",                # Test short-side version
]
```

Mutated hypotheses are scored by the `HypothesisScorer` before being added
to the queue. Low-scoring mutations are discarded; high-scoring ones enter
the testing pipeline as first-class hypotheses.

The serendipity engine publishes a `serendipity.surprise` event whenever it
generates a hypothesis with `novelty_score > 0.85` that has no close
duplicate in the existing hypothesis database.

---

## 16. The Feedback Loop

### How Confirmed Patterns Update Live Parameters

The feedback path from discovery to live capital deployment:

```
Step 1: Pattern mined → confirmed by bootstrap filter
         idea_engine.db: patterns.status = 'confirmed'
         Bus: patterns.discovered published

Step 2: Hypothesis generator creates structured hypothesis
         idea_engine.db: hypotheses row inserted
         Bus: hypotheses.created published

Step 3: Genome engine creates parameter variant encoding hypothesis
         idea_engine.db: genomes row (status='candidate')
         Bus: genome.evaluated published after backtest

Step 4: CPCV walk-forward validation
         If oos_degradation < 0.40 and sharpe_OOS > 1.2:
           genome.status = 'active'
         Else:
           genome.status = 'archived' (dead end)

Step 5: Shadow paper trading
         idea_engine.db: shadow_variants row (status='running')
         7-day minimum live paper observation window

Step 6: Shadow outperforms live by >0.15 Sharpe for 7+ days:
         → live-feedback/updater.py writes new config
         → live_trader_alpaca.py detects config file change
         → Reloads parameters on next bar boundary
         → Records parameter change in event_log
```

### Near-Real-Time Config Reloading

The live trader checks for config changes on every bar:

```python
# live_trader_alpaca.py (simplified)
def _maybe_reload_config(self):
    config_mtime = os.path.getmtime(CONFIG_PATH)
    if config_mtime > self._config_loaded_at:
        new_config = load_config(CONFIG_PATH)
        self._apply_config_delta(new_config)
        self._config_loaded_at = config_mtime
        log.info("Config reloaded: %s", new_config.version)
```

Config changes only take effect at bar boundaries to avoid mid-bar
inconsistencies. The change is logged to `event_log` with the full config
diff so the update is fully auditable.

### Feedback Velocity

| Stage             | Typical Duration | Notes |
|-------------------|-----------------|-------|
| Mine → confirm    | 1–5 minutes      | Bootstrap filter is CPU-bound |
| Confirm → hypothesis | < 1 second   | Template matching is fast |
| Hypothesis → genome | 10–60 minutes | Depends on backtest length |
| Genome → CPCV     | 30–120 minutes  | 45 OOS folds |
| CPCV → shadow     | 7 days minimum  | Non-negotiable live observation |
| Shadow → live update | 5 minutes    | Config write + reload |

**Total minimum latency**: ~7 days from pattern discovery to live parameter
change. This is deliberate — speed is not the goal; statistical validity is.

### Safeguards Against Feedback Loops

The IAE is designed to avoid self-fulfilling or self-reinforcing feedback:

1. **Source data quarantine**: Patterns are only mined from data that existed
   *before* the strategy was last updated. Trades taken under a new parameter
   set are quarantined for 30 days before they feed the next mining cycle.

2. **Change magnitude limits**: No single config update can change any
   parameter by more than 20% in one step. Large changes require two
   sequential updates separated by at least 7 days.

3. **Audit trail**: Every parameter in the live config has a `source_genome_id`
   and `source_experiment_id` that trace exactly why it has the value it does.

4. **Human review gate**: Any update affecting `target_vol`, `bh_form`, or
   position sizing caps requires a human confirmation flag before deployment.
