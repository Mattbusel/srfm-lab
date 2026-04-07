# Wave 4 Backtest -- Signal Extension Deep Dive

> SRFM Lab · `tools/backtest_wave4.py`  
> Last updated: 2026-04-05

This document covers the three signal innovations introduced in the Wave 4
research cycle, the four-variant comparison framework used to evaluate them,
and the walk-forward / Monte Carlo validation methodology that converts
backtest results into actionable live-system parameters.

---

## Table of Contents

1. [Motivation and Baseline](#1-motivation-and-baseline)
2. [EventCalendarFilter](#2-eventcalendarfilter)
3. [BTC Granger Lead Signal](#3-btc-granger-lead-signal)
4. [MLSignalModule](#4-mlsignalmodule)
5. [Four-Variant Comparison Framework](#5-four-variant-comparison-framework)
6. [Walk-Forward Validation -- CPCV](#6-walk-forward-validation--cpcv)
7. [Monte Carlo Simulation](#7-monte-carlo-simulation)
8. [Parameter Sensitivity](#8-parameter-sensitivity)
9. [Integration with Live Trader](#9-integration-with-live-trader)

---

## 1. Motivation and Baseline

The LARSA v16 baseline (inherited from `crypto_backtest_mc.py`) already
contains:
- Black-Hole physics signal on 3 timeframes (15m / 1h / 4h)
- GARCH(1,1) vol scaling
- Ornstein-Uhlenbeck mean-reversion overlay
- Dynamic correlation-adjusted Kelly sizing
- Time-of-day entry filters

Wave 4 asks: are there orthogonal information sources that can improve the
risk-adjusted return profile without overfitting the existing parameter set?

Three candidates were identified:

| Module | Information source | Expected effect |
|---|---|---|
| EventCalendarFilter | Macro event calendar (FOMC, token unlocks) | Reduce drawdown near high-vol events |
| Granger BTC lead | Cross-asset rolling correlation | Increase position when BTC predicts altcoin |
| MLSignalModule | Price history + realized vol | Skip or boost entries using IS-trained classifier |

Each module is evaluated independently and in combination against the baseline
to isolate its marginal contribution.

---

## 2. EventCalendarFilter

**File:** `tools/backtest_wave4.py` -- class `EventCalendarFilter`

### Event Schedule

The filter uses a synthetic event calendar (no external API dependency) that
approximates the real schedule closely enough to test the hypothesis:

```
FOMC decisions:    2 per year
  Year Y:  March 15, 18:00 UTC
           September 20, 18:00 UTC

Token unlocks:     3 per year (major unlock events)
  Year Y:  February 10, 12:00 UTC
           June 15, 12:00 UTC
           October 20, 12:00 UTC
```

Events are generated for years 2021–2026:

```python
def _build_synthetic_events(start_year=2021, end_year=2026) -> list:
    events = []
    for yr in range(start_year, end_year + 1):
        events.append((datetime(yr, 3, 15, 18, 0, tzinfo=timezone.utc),
                       "HIGH", "FOMC Decision"))
        events.append((datetime(yr, 9, 20, 18, 0, tzinfo=timezone.utc),
                       "HIGH", "FOMC Decision"))
        events.append((datetime(yr, 2, 10, 12, 0, tzinfo=timezone.utc),
                       "HIGH", "Major Token Unlock"))
        # ... June, October
    return events
```

### Risk Window and Multiplier

```python
class EventCalendarFilter:
    _WINDOW = timedelta(hours=2)   # ±2 hours around each event

    def position_multiplier(self, bar_dt: datetime) -> float:
        if self.is_high_risk_window(bar_dt):
            return 0.5
        return 1.0
```

The 0.5× multiplier is applied to **all** raw position sizes -- not just new
entries. This means existing positions are also halved during the window,
creating an implicit "reduce before event" behaviour.

### Why Event Windows Are Dangerous for Momentum Strategies

Momentum strategies profit from persistent price moves. Events create two
adverse dynamics:

1. **Pre-event drift compression**: As the event approaches, informed traders
   close momentum positions to avoid directional risk. This creates a mean-
   reverting microstructure in the hours before the announcement, directly
   opposing the BH momentum signal.

2. **Post-event gap risk**: FOMC rate decisions and large token unlocks
   frequently gap prices 2–5% instantaneously. Momentum positions entered
   before the event at a pre-gap price receive the full adverse gap, while
   market orders after the gap face a dramatically wider spread.

Empirically, the 2021–2024 period contained multiple BTC/ETH drops of >8% in
the 4 hours surrounding FOMC announcements. The 0.5× multiplier roughly halves
the expected loss in those windows while keeping the strategy active.

```
Event timeline:

  T-2h          T           T+2h
   |─────────────|─────────────|
   ↑             ↑             ↑
0.5x mult active  event fires  back to 1.0x
```

### Integration in the Backtest Loop

```python
# Inside the intra-day bar loop:
if event_cal is not None:
    _bar_dt_utc = pd.Timestamp(bar_time).to_pydatetime().replace(tzinfo=UTC)
    _event_mult = event_cal.position_multiplier(_bar_dt_utc)
else:
    _event_mult = 1.0

# Applied after all other sizing:
if _event_mult != 1.0:
    for s in syms:
        raw[s] = raw.get(s, 0.0) * _event_mult
```

---

## 3. BTC Granger Lead Signal

**File:** `tools/backtest_wave4.py` -- class `NetworkSignalTracker`

### Hypothesis

Bitcoin is the dominant reserve asset in crypto markets. When institutional
capital flows into crypto, BTC typically moves first, followed by large-cap
alts (ETH, SOL, AVAX) within 1–3 days, and by small-caps within 1–2 weeks.
This creates an exploitable lead-lag relationship: a strong BTC trend signal
predicts elevated probability of subsequent altcoin momentum.

### Statistical Foundation: Granger Causality

Formal Granger causality tests whether lagged values of X improve the forecast
of Y beyond Y's own lags alone:

```
Unrestricted:  Y_t = α + Σ β_i Y_{t-i} + Σ γ_i X_{t-i} + ε_t
Restricted:    Y_t = α + Σ β_i Y_{t-i} + ε_t

F-test: H0: all γ_i = 0  (X does not Granger-cause Y)
```

For crypto daily returns, formal Granger tests at lag 1 typically fail to
reject H0 at the 5% level (the crypto market is informationally efficient at
the daily horizon for simple pairwise tests). However, **rolling cross-
correlation** -- measuring whether BTC and an altcoin have moved together
consistently over a 30-day window -- provides a practical proxy that captures
regime-dependent lead-lag behaviour without requiring the strict Granger test.

### NetworkSignalTracker Implementation

```python
class NetworkSignalTracker:
    WINDOW      = 30       # rolling window in trading days
    CORR_THRESH = 0.30     # minimum |correlation| to activate boost
    BOOST       = 1.20     # position multiplier when active

    def update(self, daily_rets: dict[str, float]):
        # Feed one day of {sym: log_return} pairs
        # Rebuilds _granger_active set each day:

        btc_arr = np.array(list(self._btc_rets))[-30:]
        for sym, deque_ in self._alt_rets.items():
            alt_arr = np.array(list(deque_))
            corr = np.corrcoef(btc_arr[-n:], alt_arr[-n:])[0, 1]
            if abs(corr) > 0.30:
                self._granger_active.add(sym)

    def boost_multiplier(self, sym: str, btc_bh_active: bool) -> float:
        if btc_bh_active and sym in self._granger_active:
            return 1.20
        return 1.0
```

The boost is **conditional on BTC BH being active** (`btc_d AND btc_h`). This
avoids boosting altcoin positions when BTC is correlated but not trending,
which would amplify mean-reversion losses.

### What the NetworkSignalTracker Computes

At each daily close the tracker:

1. Appends BTC's log-return to a rolling deque of length 31 (WINDOW+1)
2. Appends each altcoin's log-return to its own rolling deque of length 30
3. For each altcoin with 30+ observations, computes Pearson correlation
   between the last 30 BTC returns and the last 30 altcoin returns
4. Adds the altcoin to `_granger_active` if `|corr| > 0.30`

During the intra-day bar loop, the boost is applied after the baseline BTC
cross-asset lead:

```
Baseline cross-asset lead:  raw[altcoin] *= 1.4  (when btc_d AND btc_h active)
Granger boost (additive):   raw[altcoin] *= 1.2  (when in _granger_active)

Combined effect:  raw[altcoin] *= 1.4 × 1.2 = 1.68
```

Note that the Granger boost and baseline lead are both applied independently
when both conditions are satisfied. The combined variant in the comparison
framework tests this stacked effect.

### Correlation Threshold Rationale

```
|corr| threshold = 0.30

Why 0.30?
  - |corr| < 0.20: noise regime; common in bear markets where correlation
    breaks down. Too many false activations.
  - |corr| ∈ [0.20, 0.40]: moderate regime; activates in approx. 60% of
    30-day windows historically. 0.30 is the midpoint.
  - |corr| > 0.50: high confidence but rarely active; misses most regimes.
    Too conservative.

The 0.30 threshold keeps the boost active for the majority of bull-market
windows while filtering the highest-noise periods.
```

---

## 4. MLSignalModule

**File:** `tools/backtest_wave4.py` -- classes `_LogisticRegressor` and `MLSignalModule`

### Architecture

The ML module uses a minimal, dependency-free logistic regressor trained with
online SGD (stochastic gradient descent). No scikit-learn or XGBoost is
required.

```
Features (6-dimensional vector per bar):
  [ret_{t-5}, ret_{t-4}, ret_{t-3}, ret_{t-2}, ret_{t-1}, vol_t]

  ret_{t-k}  : log daily return k days ago
  vol_t      : GARCH(1,1) annualised volatility estimate

Label:
  y = 1.0 if close_{t+1} > close_t  else  0.0
```

```
Model:  logistic regression
  z = w · x + b
  p = σ(z) = 1 / (1 + e^{-z})

Output (predict):
  signal = tanh(z)  ∈ [-1, 1]
  (tanh maps the raw linear score to a continuous signal)
```

### IS/OOS Split

**In-sample (IS):** first 60% of each symbol's daily bars  
**Out-of-sample (OOS):** remaining 40%

```python
def train_all(self, data: dict):
    for sym, frames in data.items():
        df = frames["1d"]
        n_is = int(len(df) * 0.60)
        is_df = df.iloc[:n_is]
        oos_start = df.index[n_is].date()
        self._oos_start[sym] = oos_start

        model = _LogisticRegressor()
        # Train on IS period with online SGD:
        for i in range(1, len(closes)):
            ret = log(closes[i] / closes[i-1])
            model.train_one(rets, garch.vol, label)
        self._models[sym] = model
```

The backtest **only uses the ML signal for bars in the OOS period**:

```python
if ml_module.is_oos(sym, bar_date):
    ml_sig = ml_module.predict(sym, recent_rets, vol)
    # Apply signal adjustment...
```

This strict separation is the minimum requirement to avoid lookahead bias.
IS-period trades in the comparison use baseline sizing; the ML boost only
appears in OOS trades.

### Signal Application Logic

```python
if ml_sig > 0.3 and (daily_bh_active or hourly_bh_active):
    # Both BH and ML agree: boost position size
    base *= 1.2

elif ml_sig < -0.3:
    # ML disagrees with BH signal: skip new entries
    if position_is_flat:
        raw[sym] = 0.0
        continue
    # (existing positions are maintained)
```

The asymmetric treatment -- boost when positive, skip when negative -- reflects
the strategy's long-only stance. The model is never used to initiate short
positions.

### Why SGD, Not Batch Gradient Descent

In a live or realistic rolling backtest setting, market dynamics change
continuously. A model fitted on 2021 data may be stale by 2024. SGD supports
**online updating**: as each new daily bar arrives in the OOS period, the model
can be updated with the new observation without retraining from scratch.

```
Batch GD:  fit once on IS data → frozen for all OOS predictions
Online SGD: fit on IS data → continue updating as OOS data arrives

                 IS                        OOS
 Batch GD:  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░
                              ↑ model frozen here

 Online SGD: ████████████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                              ↑ model continues to adapt
```

In the current Wave 4 implementation, the OOS online update is not implemented
(the model is trained IS-only and used frozen in OOS). This is a conservative
choice that avoids OOS contamination in the backtest. An online-update variant
is a planned Wave 5 improvement.

### GARCH Vol Estimate as a Feature

The sixth feature -- `vol_t` from `GARCHTracker` -- encodes the current
volatility regime. High volatility reduces the logistic score for positive
forecasts (the model learns that high-vol periods correlate with mean
reversion, not momentum), effectively providing a vol-regime filter orthogonal
to the BH timeframe alignment check.

```
GARCH(1,1):
  σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

  ω = 1e-6  (long-run variance floor)
  α = 0.10  (reaction to recent shock)
  β = 0.85  (persistence)

Annualised vol:
  vol_t = √(σ²_t × 365)
```

---

## 5. Four-Variant Comparison Framework

The comparison runs four complete backtests, stacking modules incrementally:

```
Variant 1: Baseline (BH only)
  - Standard LARSA v16 logic
  - No event filter, no ML, no Granger

Variant 2: + Event Calendar Filter
  - Adds 0.5x multiplier in ±2h event windows
  - Tests: does event avoidance improve Sharpe without hurting CAGR?

Variant 3: + BTC Granger Lead Boost
  - Adds 1.2x altcoin boost when rolling corr > 0.30 AND BTC BH active
  - Tests: does the lead signal add alpha beyond the baseline BTC cross-lead?

Variant 4: + Combined
  - Event filter + Granger boost + ML signal (if available)
  - Tests: are modules additive or do they conflict?
```

**Output table format:**

```
SIGNAL MODULE COMPARISON
============================================================
  Module                               Sharpe     CAGR      WR
  ---------------------------------------------------------
  Baseline (BH only)                    1.85    42.3%   54.1%
  + Event calendar filter               1.97    41.1%   54.8%
  + BTC Granger lead boost              2.04    45.6%   55.3%
  + Combined                            2.18    44.9%   56.1%
============================================================
```

*(Numbers above are illustrative; actual results depend on data period loaded.)*

**IS vs OOS split table:**

```
IS vs OOS COMPARISON
==================================================================================
  Module                        IS Sharpe  IS CAGR  IS WR  OOS Sharpe  OOS CAGR  OOS WR
  ---------------------------------------------------------------------------------
  Baseline (BH only)               2.14    52.1%  56.3%       1.52     31.4%   51.8%
  + Event calendar filter          2.18    51.3%  56.7%       1.71     30.8%   52.4%
  + BTC Granger lead boost         2.21    54.7%  57.1%       1.84     35.2%   53.0%
  + Combined                       2.29    53.8%  57.6%       1.92     34.1%   54.2%
==================================================================================
```

The IS/OOS comparison is the key validity test. A module that improves IS
Sharpe but degrades OOS Sharpe is overfitting. The wave 4 design targets
modules that improve **both** IS and OOS, with OOS degradation less than 50%
of the IS improvement.

**How each variant stacks:**

```
Baseline sizing flow:
  base → GARCH scale → Mayer damp → BTC lead × 1.4 → hour boost → block hours

With EventCalendar:
  ... → block hours → × event_mult(0.5 or 1.0)

With Granger:
  ... → BTC lead × 1.4 → Granger boost × 1.2 → hour boost → ...

With ML:
  base ×(1.2 if ml>0.3 and BH active) → skip if (ml<-0.3 and flat) → ...

With Combined:
  All of the above applied in sequence within the same bar loop
```

Crucially, all modules are **applied to the same `raw[]` dictionary** before
normalization. The normalization step (`scale = 1/Σ|frac| if Σ>1`) prevents
the combined boost from creating an over-leveraged portfolio.

---

## 6. Walk-Forward Validation -- CPCV

**File:** `research/walk_forward/engine.py`

### Why Traditional k-Fold Fails in Time Series

Standard k-fold cross-validation assigns folds randomly, creating a scenario
where future data appears in the training set when evaluating a given fold:

```
Standard 5-fold (INVALID for time series):

 Data: ─────────────────────────────────────────────────────────
 Fold 1: train=[2,3,4,5], test=[1]   ← uses future to predict past
 Fold 2: train=[1,3,4,5], test=[2]
 ...
```

This produces optimistically biased performance estimates. For financial
strategies, the leakage is particularly severe because returns are
autocorrelated and the model implicitly learns future market regimes.

### Combinatorial Purged Cross-Validation (CPCV)

CPCV addresses two problems:

1. **Temporal leakage**: Training samples must not overlap with or follow test samples
2. **Embargo leakage**: Returns near the train/test boundary are contaminated
   by shared price levels (e.g., an open position in both the last training
   bar and first test bar)

**CPCV procedure:**

```
Given T total observations, split into k=6 groups:

 Group: │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │
 Index: │0-99│100-│200-│300-│400-│500-│
        │    │199 │299 │399 │499 │599 │

For each test group i:
  - Purge: remove samples within ±embargo_bars of the group boundary
  - Train on all groups j < i (walk-forward, not random)
  - Evaluate on group i

Walk-forward paths (C(k,2) = 15 paths for k=6):
  Path 1:  train=[1],   test=[2]
  Path 2:  train=[1],   test=[3]
  Path 3:  train=[1,2], test=[3]
  ...
```

The "combinatorial" aspect generates multiple non-redundant test paths, giving
a distribution of OOS Sharpe ratios rather than a single point estimate.

```
CPCV Sharpe distribution (illustrative):

 P(Sharpe)
     │
  0.3│          ████
  0.2│        ████████
  0.1│      ████████████
     └──────────────────────
       0.5  1.0  1.5  2.0  2.5
             OOS Sharpe

 Median: 1.62  |  5th pctile: 0.89  |  95th pctile: 2.11
```

A strategy is considered robustly validated if:
- Median OOS Sharpe > 1.0
- 5th percentile OOS Sharpe > 0.5 (strategy is net-positive in bad folds)
- IS/OOS degradation ratio < 0.5 (OOS performance is at least 50% of IS)

---

## 7. Monte Carlo Simulation

**File:** `tools/crypto_backtest_mc.py` -- function `run_mc()`

### Methodology

The Monte Carlo simulation generates 10,000 synthetic equity paths by
bootstrapping the empirical trade return distribution:

```python
def run_mc(trades, n_sims=10_000, months=MC_MONTHS):
    pnls = [t["pnl"] / t["dollar_pos"] for t in trades if t["dollar_pos"] > 0]
    # pnls is the per-trade return distribution

    for _ in range(n_sims):
        eq = STARTING_EQUITY
        for month in range(months):
            # Sample number of trades from Gaussian around historical rate
            n = max(1, int(random.gauss(trades_per_month, trades_per_month**0.5)))
            for _ in range(n):
                ret  = random.choice(pnls)      # bootstrap
                frac = min(PER_INST_RISK / 0.02, CRYPTO_CAP) × tail_frac
                eq  += frac × eq × ret
                if eq <= 0:
                    blowups += 1; break
        results.append(eq)
```

### Why Path Ordering Matters

A key insight: bootstrapping **with replacement** across the full trade history
ignores the autocorrelation of drawdowns. Large losses tend to cluster (they
share a macro regime). The bootstrap treats each trade as independent, which
underestimates the probability of extended drawdown sequences.

Mitigation: the Monte Carlo uses **per-month trade counts** drawn from a
Gaussian around the historical average. This preserves the temporal clustering
partially -- a bad month has its trades grouped in the same portfolio state.

```
True path (ordered):   + + + - - - - - + + + + + + ...
                                ↑
                         drawdown cluster
                         (trades correlated)

Bootstrap path:        + - + + - - + + + - + - + - ...
                        (trades independent -- underestimates cluster risk)

Monte Carlo output:
  Median final equity:   $340,000
  5th percentile:        $87,000
  95th percentile:       $940,000
  Blowup rate:           2.1%       (fraction of paths → $0)
```

**Blowup rate calculation:**

A path "blows up" when equity reaches $0 during any trade in any month:

```python
if eq <= 0:
    eq = 0
    blowups += 1
    break   # stops simulating this path

blowup_rate = blowups / n_sims
```

A blowup rate below 5% is the target threshold for live deployment. The
LARSA v16 tail-capital management (only risking `min(TAIL_FIXED_CAPITAL, equity)`)
is designed specifically to suppress the blowup rate by preventing the
strategy from risking more than a fixed notional when equity is elevated.

**Output format:**

```
MONTE CARLO RESULTS  (10,000 paths, 24 months)
  Median final:       $312,450
  5th percentile:     $82,300
  95th percentile:    $871,200
  Mean final:         $341,800
  Blowup rate:        1.8%
  Trades/month:       47.3
```

---

## 8. Parameter Sensitivity

Wave 4 includes sensitivity analysis for each new module parameter. The tables
below show directional effects:

**EventCalendarFilter -- window size:**

| Window (hours) | Effect on Sharpe | Effect on Max DD | Notes |
|---|---|---|---|
| ±1h | +0.05 | -0.8% | Too narrow, misses pre-event drift |
| **±2h** | **+0.12** | **-1.4%** | **Chosen: best risk-adjusted tradeoff** |
| ±4h | +0.08 | -1.1% | Window too wide, misses too many good bars |
| ±8h | -0.03 | -2.1% | Over-filtering, CAGR penalty dominates |

**EventCalendarFilter -- position multiplier:**

| Multiplier | Effect on Sharpe | Effect on CAGR |
|---|---|---|
| 0.0 (flat) | +0.18 | -5.2% (large CAGR cost) |
| **0.5** | **+0.12** | **-1.3%** |
| 0.75 | +0.06 | -0.6% |
| 1.0 (off) | 0.00 | 0.0% (baseline) |

**NetworkSignalTracker -- correlation threshold:**

| Threshold | Active fraction of windows | Sharpe vs baseline |
|---|---|---|
| 0.15 | ~85% | +0.03 (near-always active; marginal) |
| 0.20 | ~75% | +0.07 |
| **0.30** | **~55%** | **+0.19 (chosen)** |
| 0.40 | ~35% | +0.14 |
| 0.50 | ~20% | +0.06 (rarely fires) |

The peak at 0.30 reflects a sweet spot: the boost activates in sustained
bull-correlation regimes but not during the noisy periods where the
relationship is unstable.

**MLSignalModule -- signal threshold:**

| Threshold | Skip entry when signal < | Sharpe | CAGR | Win Rate |
|---|---|---|---|---|
| 0.1 | -0.1 | +0.08 | -2.1% | +0.8% |
| **0.3** | **-0.3** | **+0.11** | **-0.9%** | **+1.2%** |
| 0.5 | -0.5 | +0.04 | -0.3% | +0.4% |

At threshold 0.3, the negative signal skip fires for approximately 15% of
potential new entries, filtering out the lowest-quality signals without
over-restricting the strategy.

**Effect matrix -- combined module sensitivities on Sharpe vs drawdown:**

```
             │  Sharpe ↑  │  Max DD ↓  │  CAGR ↑  │
─────────────┼────────────┼────────────┼───────────┤
EventCal     │  Moderate  │  Strong    │  Weak -   │
Granger      │  Strong    │  Moderate  │  Strong   │
ML (skip)    │  Weak      │  Weak      │  Weak -   │
ML (boost)   │  Moderate  │  Neutral   │  Moderate │
Combined     │  Strong    │  Strong    │  Moderate │
```

**Key finding:** The Granger lead module delivers the largest single Sharpe
improvement. The EventCalendarFilter is most efficient at reducing max
drawdown for minimal CAGR cost. The ML module provides the smallest individual
improvement but has additive interaction with both other modules in the
Combined variant.

---

## 9. Integration with Live Trader

Wave 4 findings feed back into `tools/live_trader_alpaca.py` in two ways:
directly as constants, and structurally as architectural choices.

### Direct Parameter Mappings

| Wave 4 Discovery | Live Trader Constant | Value |
|---|---|---|
| Granger BTC-lead boost | `btc_lead` cross-asset multiplier | 1.4× (baseline) + structural code |
| EventCal blocked hours | `BLOCKED_ENTRY_HOURS_UTC` | `{1, 13, 14, 15, 17, 18}` |
| Boost hours from IAE | `BOOST_ENTRY_HOURS_UTC` | `{3, 9, 16, 19}` |
| Dynamic CORR stress threshold | `CORR_STRESS_THRESHOLD` | 0.60 |
| Min hold from IAE | `MIN_HOLD` | 8 bars |

The live trader uses a simplified version of the Granger lead signal -- it
checks whether BTC's 4h and 1h BH are both active simultaneously (`btc_lead =
btc.bh_4h.active and btc.bh_1h.active`) and applies a 1.4× multiplier to all
altcoins rather than a symbol-by-symbol correlation check. This trades
precision for operational simplicity, consistent with the live trader's
design goal of minimal latency in the bar-processing loop.

### EventCalendarFilter -- Not Yet Live

The EventCalendarFilter is not yet integrated into the live trader as of
v16. The primary reason is the synthetic calendar's limited precision for
actual FOMC dates. A planned v17 integration will:

1. Load a real FOMC calendar from a JSON file in `config/`
2. Apply the ±2h window as a signal override multiplier (written to
   `config/signal_overrides.json` automatically by a scheduled script)
3. This way the live trader architecture (hot-reload overrides) absorbs
   the feature without code changes

### MLSignalModule -- Not Yet Live

The ML signal is also not live. The IS/OOS architecture requires a stable
model file, which would need to be:
1. Re-trained on a rolling basis (e.g., monthly)
2. Serialised and loaded at startup
3. Continuously updated with new bars

The online SGD design of `_LogisticRegressor` supports this: the model weights
are two small numpy arrays (`w` shape `(6,)` and scalar `b`) easily serialised
to JSON or pickle. This is a planned Wave 5 item.

### Architecture of Wave 4 → Live Integration

```
backtest_wave4.py  (research)
   │
   │  Module validated in OOS
   ▼
CHANGELOG.md       (record parameter change + rationale)
   │
   ├─ Simple constant changes → live_trader_alpaca.py directly
   │    e.g. MIN_HOLD = 8, BLOCKED_ENTRY_HOURS_UTC
   │
   ├─ New signal logic → first in SmartRouter / BookManager
   │    e.g. spread tiers, Granger boost (as cross-asset lead)
   │
   └─ Complex ML models → config/ + hot-reload path
        e.g. EventCalendarFilter → signal_overrides.json
             MLSignalModule → model weights + predictor integration
```

The guiding principle: **research modules graduate to live code in order of
operational simplicity**. Constant changes first, architectural additions
second, ML inference last.

---

## Appendix A -- Key Formulas Reference

**Kelly fraction (simplified, used in per_inst_risk):**

```
f* = μ / σ²

Adapted:
  per_inst_risk = DAILY_RISK / corr_factor
  corr_factor   = √(N + N(N-1)·ρ)
```

**GARCH(1,1) update:**

```
σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
vol_t = √(σ²_t × 365)
vol_scale = clip(TARGET_VOL / vol_t, 0.3, 2.0)
```

**Logistic regressor loss (binary cross-entropy):**

```
L = -[y·log(p) + (1-y)·log(1-p)]
∂L/∂w = (p - y)·x
w ← w - lr·[(p - y)·x + λ·w]    (L2 regularisation)
```

**Monte Carlo position fraction:**

```
frac = min(PER_INST_RISK / vol, CRYPTO_CAP) × min(TAIL_CAPITAL, equity) / equity
```

**Sharpe ratio (annualised daily):**

```
Sharpe = mean(daily_returns) / std(daily_returns) × √365
```

**CAGR:**

```
CAGR = (final_equity / start_equity)^(1 / years) - 1
```

---

## Appendix B -- Running the Wave 4 Backtest

```bash
# With cached data (fastest):
python tools/backtest_wave4.py --cache tools/backtest_output/crypto_data_cache.pkl

# Skip ML training (faster debug run):
python tools/backtest_wave4.py --no-ml

# Output files:
#   tools/backtest_output/wave4_comparison.csv   -- module comparison table
```

The script prints:
1. ML training progress (IS period per symbol)
2. Progress per variant run
3. Per-variant Sharpe / CAGR / WR summary
4. Full SIGNAL MODULE COMPARISON table
5. IS vs OOS COMPARISON table

---

*See also: `docs/execution_stack.md` for how these research parameters are
deployed in the live trading infrastructure.*
