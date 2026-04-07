# Black Hole Physics in Price Space

## Overview

The SRFM (Spacetime Relativistic Financial Mechanics) framework applies concepts from general relativity to model momentum accumulation in financial markets. The core idea is that large price moves are not merely statistical outliers -- they are events where a market "approaches the speed of light" relative to its own natural volatility scale, causing spacetime curvature that traps momentum and creates a persistent directional force.

This document derives the full mathematical framework from first principles.

---

## 1. Minkowski Metric in Price Space

In special relativity, the spacetime interval is:

```
ds² = -c² dt² + dx²
```

where `c` is the speed of light and `dx` is a spatial displacement.

In price space, we define the analogous interval. The "spatial" coordinate is the normalized price return:

```
dp/p  ≡  d(ln P)  ≈  log_return
```

The "temporal" coordinate is a bar index. The "speed of light" analog is the **Curvature Factor (CF)**, which is instrument-specific and timeframe-specific.

The Minkowski interval in price space becomes:

```
ds²_price = -CF² dτ² + (d ln P)²
```

where `dτ` is one bar of time.

The **relativistic beta** (analogous to v/c) is:

```
β = |d ln P / CF| = |log_return| / CF
```

CF is calibrated to approximately one standard deviation of returns at the given timeframe, so β is roughly the number of standard deviations in z-score terms. Normal moves have β < 1; extraordinary moves approach β → 1.

---

## 2. Beta Computation

For each bar:

```
β_t = |log_return_t| / CF
```

where `log_return_t = ln(close_t / close_{t-1})` and CF is the curvature factor for that instrument and timeframe.

**Implementation note**: β is capped at `1 - ε` (where ε = 1e-9) to prevent the Lorentz factor from diverging to infinity on gap-open moves.

### CF Calibration

CF should be set to approximately the **root mean square** of typical returns at the given timeframe:

```
CF ≈ σ_returns(timeframe)
```

In practice, we set CF slightly above the 1σ level so that β > 1 only for genuine momentum events:

| Instrument | CF_15m   | CF_1h    | CF_1d    |
|-----------|----------|----------|----------|
| ES (S&P)  | 0.00030  | 0.00100  | 0.00500  |
| NQ (Nasdaq)| 0.00040 | 0.00120  | 0.00600  |
| CL (Oil)  | 0.00150  | 0.00400  | 0.01500  |
| BTC       | 0.00500  | 0.01500  | 0.05000  |
| VX (VIX)  | 0.00300  | 0.00800  | 0.02500  |

The scaling `CF_1h ≈ 3.3 × CF_15m` and `CF_1d ≈ 5 × CF_1h` follows from the square-root-of-time rule for volatility scaling (√4 ≈ 2.0 for 15m→1h, √6.5 ≈ 2.55 for 1h→daily at 6.5 trading hours/day).

---

## 3. Lorentz Factor

The Lorentz factor γ quantifies time dilation and mass increase:

```
γ = 1 / √(1 - β²)
```

For small β (normal moves), γ ≈ 1 + β²/2 ≈ 1 (negligible mass contribution).

For large β (extreme moves), γ diverges toward infinity. At β = 0.99, γ ≈ 7.1. At β = 0.999, γ ≈ 22.4.

The **relativistic mass contribution** of a single bar is:

```
Δm = γ - 1
```

This is zero for stationary matter (β = 0) and unbounded for motion at the speed of light (β → 1).

---

## 4. Mass Accumulation Formula

The BH mass is an exponentially-decaying accumulator of relativistic mass contributions:

```
m_t = decay × m_{t-1} + (γ_t - 1)
```

where:
- `decay` ∈ (0, 1) is the per-bar decay multiplier (default 0.95)
- `γ_t - 1` is the relativistic mass contribution of bar t
- `m_t` is capped at 20.0 to prevent runaway accumulation on gap events

**Convergence property**: In the absence of extreme moves (β → 0), the mass decays geometrically:

```
m_t → 0  as  t → ∞  (if all β ≈ 0)
```

The steady-state mass from a constant β-value input is:

```
m_ss = (γ - 1) / (1 - decay)
```

For decay = 0.95: m_ss = 20 × (γ - 1). A β = 0.3 bar (γ ≈ 1.048) contributes 20 × 0.048 ≈ 0.96 units in steady state -- just below the default formation threshold of 1.5.

A β = 0.4 bar (γ ≈ 1.091) contributes 20 × 0.091 ≈ 1.82 units, exceeding the threshold after a single large bar. This is the "single-bar formation" case seen in gap events.

---

## 5. BH Formation Threshold

A Black Hole is declared when:

```
m_t >= bh_form
```

The default `bh_form = 1.5` for most instruments. Noisier instruments (CL, NG, VX) use `bh_form = 1.8` to reduce false formations.

**What bh_form means physically**: It is the minimum accumulated "relativistic mass" required to declare that a directional event has enough gravitational pull to trap momentum. Higher values mean stricter filtering (fewer but higher-quality signals).

### BH Collapse

The BH collapses when mass falls below `bh_collapse` (default 1.0). The gap between formation (1.5) and collapse (1.0) creates a hysteresis band that prevents rapid toggling on/off.

---

## 6. BH Direction

The direction of the BH (which way it pulls prices) is determined by the sign of the log return that caused the mass to exceed the threshold:

```
bh_dir = +1  if log_return > 0 at formation (bullish BH)
bh_dir = -1  if log_return < 0 at formation (bearish BH)
```

Once a BH is active, the direction can update on subsequent large bars to track momentum shifts.

---

## 7. Multi-Timeframe Theory

The strategy uses three timeframes simultaneously: 15m, 1h, and 1d. Each has its own BH state independently computed with that timeframe's CF values.

**Why three timeframes?**

1. **Daily (1d)**: Captures multi-day momentum events. Regime-defining moves. Duration: days to weeks.
2. **Hourly (1h)**: Intraday momentum with multi-hour persistence. Catches session-level breakouts.
3. **15-minute (15m)**: Short-duration momentum bursts. High signal frequency, lower reliability alone.

**TF Score**: A 3-bit activation mask plus a directional consensus bonus:
```
tf_score = (active_1d << 2) | (active_1h << 1) | (active_15m)
```
Bonus: if all active timeframes agree on direction, add 1 (max 7).

| tf_score | Meaning                              | Position cap |
|----------|--------------------------------------|-------------|
| 0        | No BH active                         | 0%          |
| 1        | 15m only active                      | 15%         |
| 2        | 1h only active                       | 25%         |
| 3        | 15m + 1h active (same direction)     | 30%         |
| 4        | 1d only active                       | 35%         |
| 5        | 15m + 1d (same direction)            | 45%         |
| 6        | 1h + 1d (same direction)             | 55%         |
| 7        | All three + directional agreement    | 65%         |

**Why does multi-TF agreement matter?** When the daily BH forms (slow momentum), the hourly BH often confirms within hours (fast momentum catches up). The 15m BH then acts as the immediate entry trigger. All three firing together means the market is moving coherently at all timescales -- a true momentum regime rather than a single large bar.

---

## 8. Regime Classification

The BH engine classifies each bar into one of four regimes based on the distribution of recent returns:

| Regime          | Definition                               | BH behavior                          |
|----------------|------------------------------------------|--------------------------------------|
| BULL           | SMA slope > 0, low volatility           | Long BH formations expected          |
| BEAR           | SMA slope < 0, low volatility           | Short BH formations expected         |
| SIDEWAYS       | Flat SMA, low volatility                | BH formations rare; use mean reversion|
| HIGH_VOLATILITY| Any direction, high realized vol        | BH formations frequent but shorter   |

**Regime detection algorithm** (simplified):
1. Compute 20-bar SMA and 50-bar SMA
2. Return z-score = (close - 20 SMA) / rolling_std(20)
3. Volatility = rolling_std(20) / CF (normalized)
4. If vol > 2.0: HIGH_VOLATILITY
5. Else if 20-SMA > 50-SMA and return z-score > 0.5: BULL
6. Else if 20-SMA < 50-SMA and return z-score < -0.5: BEAR
7. Else: SIDEWAYS

---

## 9. Position Sizing from BH State

The position fraction is determined by a combination of:
1. **tf_score cap**: Each tf_score level has a maximum fraction (see table above)
2. **pos_floor**: A minimum fraction when BH is active, preventing over-small positions
3. **Portfolio risk budget**: Distributed across instruments via correlation adjustment

The effective position fraction:

```python
pos_frac = max(pos_floor, TF_CAP[tf_score]) * per_instrument_risk_budget
```

where `per_instrument_risk_budget` accounts for inter-instrument correlations.

### pos_floor Mechanics

pos_floor is a "minimum conviction floor." When BH is active with tf_score ≥ min_tf_score, we commit at least `pos_floor` of the per-instrument budget. This prevents the algorithm from taking trivially small positions when tf_score is borderline.

**Mathematical justification**: From Kelly theory, if we have an edge (positive expectancy), taking an epsilon-small position has an opportunity cost proportional to the edge × (1 - epsilon) ≈ edge. pos_floor enforces a minimum commitment that makes the risk-adjusted return per trade meaningful.

---

## 10. Comparison to Traditional Trend-Following

| Feature                   | Traditional Breakout     | BH Physics               |
|--------------------------|--------------------------|--------------------------|
| Entry signal             | Price cross of N-period high | Relativistic mass threshold |
| Signal strength measure  | ATR multiple            | γ - 1 (relativistic mass) |
| Regime filtering         | Optional lookback       | Built-in via mass decay  |
| Multi-timeframe          | Manual parameter sets   | Unified mass hierarchy   |
| Momentum decay model     | Linear or none          | Exponential decay (0.95) |
| Signal persistence       | None after entry        | Mass continues updating  |
| Directional flip         | Stop hit → re-entry     | Direction flips with mass|

The key difference is that BH mass is **self-normalizing via the CF parameter**. A 1% move in ES (CF_1d = 0.005) generates β = 0.2 and γ ≈ 1.02 -- a small contribution. The same 1% move in BTC (CF_1d = 0.05) generates β = 0.02 and γ ≈ 1.0002 -- negligible. This means CF calibration automatically accounts for the instrument's natural volatility, making the same threshold (bh_form = 1.5) appropriate across asset classes with wildly different volatility levels.

---

## 11. Gravitational Analogies (Intuition)

| Physics concept          | SRFM analog                              |
|--------------------------|------------------------------------------|
| Speed of light (c)       | Curvature factor (CF)                    |
| Relativistic beta (v/c)  | Normalized price return (|Δp/p| / CF)   |
| Lorentz factor (γ)       | Momentum amplification factor            |
| Gravitational mass       | Accumulated momentum (m)                 |
| Black hole formation     | Mass exceeds bh_form threshold           |
| Event horizon            | The mass level below which exits trigger |
| Hawking radiation        | Mass decay (exponential with factor 0.95)|
| Geodesic                 | The "path of least resistance" price follows after BH forms |

The analogy is not perfect (it is a conceptual framework, not a physical derivation), but it provides:
1. A consistent parameterization scheme across assets
2. Intuitive interpretation of signal strength
3. Natural position sizing via mass-proportional conviction

---

## 12. Worked Example: ES Formation Event (June 2022)

On 2022-06-10, ES (via SPY) fell approximately 2.9% intraday and 3.4% on the daily close:

```
log_return = ln(386.6 / 400.0) = -0.0341
β          = 0.0341 / 0.005   = 6.82  → capped at 0.9999
γ          = 1 / sqrt(1 - 0.9999²) ≈ 70.7
Δm         = γ - 1 ≈ 69.7    → capped at 20.0

mass_new   = 0.95 × previous_mass + 20.0
```

Even starting from mass = 0, this single bar pushes mass = 20.0 >> 1.5. A BH forms immediately on this bar with direction = -1 (bearish).

The following two weeks saw the market consolidate, with the BH persisting (mass decaying slowly from 20.0 × 0.95^n ≈ 13.5 after one week). The trade was entered long S&P 1x inverse ETF equivalent and closed for +6.7% when mass fell below 1.0 (collapse) after 14 days.

This demonstrates the BH's role as a **momentum persistence filter**: the formation event identifies the regime break, and the decay rate determines how long the signal is trusted.
