# Portfolio Theory for BH-Based Strategies

## Overview

The multi-instrument LARSA strategy requires a portfolio theory framework for allocating capital across instruments. This document covers mean-variance optimization, risk parity, Hierarchical Risk Parity (HRP), and the BH-specific correlation measure that should be used for diversification analysis.

---

## 1. Mean-Variance vs Risk Parity vs HRP

### Mean-Variance (Markowitz)

Classical Markowitz optimization finds weights that minimize portfolio variance for a given expected return target:

```
min  w' Σ w
s.t. w' μ = μ_target
     w' 1 = 1
```

**Problems for SRFM use**:
1. Requires estimating expected returns `μ` — notoriously noisy
2. Concentrates heavily in low-variance assets (bonds), ignoring momentum signals
3. Optimization is extremely sensitive to covariance matrix estimation errors
4. Does not account for the BH signal structure

### Risk Parity

Risk parity assigns weights such that each instrument contributes equally to total portfolio variance:

```
w_i = (1/σ_i) / Σ_j (1/σ_j)    (simplified: inverse volatility weighting)
```

The full risk parity with correlations:

```
w_i × (Σw)_i = constant for all i
```

where `(Σw)_i` is the i-th component of the vector `Σw`.

**Advantages for SRFM**:
- No expected return estimation required
- Naturally diversifies across asset classes with different volatility levels
- Well-calibrated to handle the wide volatility range (VX vs ZB)

**Disadvantages**:
- Ignores the BH signal (treats all instruments as equally tradeable at all times)
- Equal risk contribution ≠ equal expected return contribution

### Hierarchical Risk Parity (HRP)

HRP (Lopez de Prado, 2016) uses hierarchical clustering of the correlation matrix to assign weights. The algorithm:

1. Compute the correlation matrix of asset returns
2. Compute a distance matrix: `d_ij = sqrt((1 - ρ_ij) / 2)`
3. Cluster assets hierarchically (single-linkage)
4. Recursive bisection: assign weights inversely proportional to cluster variance

**Why HRP is preferred for SRFM**:
- Robust to estimation error in the covariance matrix
- Handles the block-diagonal structure (equities correlated, crypto correlated, uncorrelated to each other)
- No matrix inversion required (avoids numerical instability)
- Naturally positions the portfolio along correlation clusters

---

## 2. Black-Litterman with BH-Based Views

The Black-Litterman model allows incorporation of "views" (directional signals) into the prior market equilibrium portfolio weights.

### Standard Black-Litterman

```
μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} × [(τΣ)^{-1}Π + P'Ω^{-1}Q]
```

where:
- `Π` = equilibrium excess returns (from CAPM: `Π = λΣw_market`)
- `P` = pick matrix (which instruments have views)
- `Q` = view returns
- `Ω` = view uncertainty matrix (diagonal)
- `τ` = scaling factor (typically 1/T where T = estimation window)

### BH Views

The BH engine provides directional views that can be mapped to Black-Litterman:

| tf_score | BH active? | Direction | View magnitude (Q) | View confidence (1/Ω_ii) |
|---------|-----------|-----------|-------------------|--------------------------|
| 7       | Yes (all TFs) | Aligned | `bh_mass_1d × CF_1d × 252` | High (0.9) |
| 4-6     | Partial    | Mixed | Proportional to mass | Medium (0.5) |
| 1-3     | Partial    | Mixed | Lower | Low (0.2) |
| 0       | No         | None | 0 | 0 (no view) |

The view magnitude formula:

```python
view_return = direction × bh_mass_1d × cf_1d × annualization_factor
```

This translates BH mass (an abstract momentum unit) into an expected return in the Black-Litterman framework.

**Practical application**: Use BL weights only as a soft guide. The actual position fractions are constrained by the `TF_CAP` table in the LARSA engine.

---

## 3. BH Activation Correlation vs Price Correlation

### Why standard price correlation understates diversification

The standard Pearson correlation of daily returns between ES and NQ is approximately 0.92 — they appear nearly identical for diversification purposes. Yet in practice, BH formations on NQ often precede or follow ES formations by 0-2 bars and carry independent information.

The relevant quantity for BH-based portfolio theory is **BH activation correlation**: how often do two instruments have BH events simultaneously?

### Computing BH activation correlation

```sql
-- See warehouse/schema/05_queries.sql Q18
WITH daily_activations AS (
    SELECT instrument_id, DATE(timestamp) AS dt, MAX(active::INT) AS was_active
    FROM bh_state_1d
    GROUP BY instrument_id, DATE(timestamp)
)
SELECT
    CORR(a.was_active::FLOAT, b.was_active::FLOAT) AS bh_corr
FROM daily_activations a
JOIN daily_activations b ON b.dt = a.dt AND b.instrument_id > a.instrument_id
```

### Empirical findings

| Pair          | Price Return Corr | BH Activation Corr |
|--------------|-------------------|--------------------|
| ES / NQ      | 0.92              | 0.65               |
| ES / CL      | 0.35              | 0.18               |
| ES / GC      | -0.05             | 0.12               |
| ES / ZB      | -0.30             | 0.08               |
| ES / VX      | -0.75             | 0.21               |
| BTC / ETH    | 0.88              | 0.72               |
| ES / BTC     | 0.45              | 0.22               |
| CL / NG      | 0.50              | 0.25               |

**Key insight**: BH activation correlations are consistently **lower** than price return correlations. This is because BH events are regime-triggered (extreme moves), not just price-level movements. Two instruments can be highly correlated in daily returns while having uncorrelated BH events if their extreme moves happen at different times.

**Portfolio implication**: The effective diversification from adding more instruments is **greater** than price correlation would suggest. The LARSA correlation factor of 0.35 used in position sizing is calibrated from BH activation correlation, not price return correlation.

---

## 4. Correlation-Adjusted Position Sizing

The per-instrument risk budget accounts for portfolio-level correlation:

```python
N = 8                   # number of instruments
rho = 0.35              # average pairwise BH activation correlation
corr_factor = sqrt(N + N × (N-1) × rho)   # ≈ 5.54
per_inst_risk = portfolio_daily_risk / corr_factor   # ≈ 0.00181 (0.18% daily)
```

This formula derives from the variance of an equal-weighted portfolio:

```
σ²_portfolio = (σ_inst / N)² × [N + N(N-1)ρ] = σ²_inst × [1 + (N-1)ρ] / N
```

Setting `σ_portfolio = target` and solving for `σ_inst`:

```
σ_inst = σ_portfolio × sqrt(N / [1 + (N-1)ρ])
       = σ_portfolio × sqrt(N) / sqrt(1 + (N-1)ρ)
       ≡ σ_portfolio × corr_factor_inv
```

---

## 5. Gear 1 / Gear 2 Capital Allocation

LARSA v16 uses a two-gear structure:

**Gear 1 (Tail Capture)**: Fixed $3M bucket allocated to BH trend-following across 8 instruments. This never increases above $3M regardless of total portfolio size — it is always "primed" for the next BH event.

**Gear 2 (Harvest Mode)**: Everything above $3M is allocated to Z-score mean reversion in SIDEWAYS regimes. This bucket grows as profits accumulate.

The capital allocation model:

```
Total equity = $T
Gear 1 capital = min($3M, T × 0.80)   # 80% cap prevents gear 1 domination
Gear 2 capital = max(0, T - $3M)
```

This structure separates two fundamentally different alpha sources:
1. **Event-driven alpha** (Gear 1): High variance, high return, requires stable capital base
2. **Carry alpha** (Gear 2): Low variance, consistent return, grows with scale

---

## 6. Regime-Based Allocation Switching

When a regime transition occurs, the portfolio allocation rules change:

| From → To              | Action                                           |
|------------------------|--------------------------------------------------|
| SIDEWAYS → BULL        | Exit all Gear 2 mean-reversion positions; activate Gear 1 BH tracking |
| SIDEWAYS → BEAR        | Same as above (bearish BH signals fire)          |
| BULL/BEAR → SIDEWAYS   | Deactivate Gear 1 entries; begin Gear 2 Z-score tracking |
| ANY → HIGH_VOLATILITY  | Reduce all position sizes by 50%; require higher tf_score threshold |
| HIGH_VOLATILITY → ANY  | Gradually restore position sizes over 5 bars     |

The regime transition logic prevents whipsawing by requiring the regime to persist for at least 3 bars before switching gears.
