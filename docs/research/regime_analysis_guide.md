# Regime Analysis Guide

## Overview

Regime detection determines whether the market is trending (BULL/BEAR), consolidating (SIDEWAYS), or experiencing elevated volatility (HIGH_VOLATILITY). Each regime warrants a different trading approach.

---

## Regime Detection Algorithm

The regime classifier uses three inputs:
1. **SMA ratio**: 20-bar SMA vs 50-bar SMA (trend direction)
2. **Return z-score**: (close - 20-SMA) / rolling_std (magnitude of trend)
3. **Normalized volatility**: rolling_std / CF (how volatile relative to the instrument's baseline)

```python
def classify_regime(closes, cf):
    sma20 = closes.rolling(20).mean()
    sma50 = closes.rolling(50).mean()
    std20 = closes.pct_change().rolling(20).std()
    z = (closes - sma20) / (sma20 * std20 + 1e-9)
    norm_vol = std20 / cf

    if norm_vol > 2.0:
        return "HIGH_VOLATILITY"
    elif sma20 > sma50 and z > 0.5:
        return "BULL"
    elif sma20 < sma50 and z < -0.5:
        return "BEAR"
    else:
        return "SIDEWAYS"
```

---

## Regime Database Queries

### Current regime for all instruments

```sql
SELECT i.symbol, cs.regime_1d AS regime, cs.updated_at
FROM bh_current_state cs
JOIN instruments i ON i.id = cs.instrument_id
ORDER BY cs.regime_1d, i.symbol;
```

### Regime duration history

```sql
-- Q13_regime_duration_stats from 05_queries.sql
SELECT symbol, regime, avg_duration_bars, avg_regime_return_pct
FROM regime_duration_stats
ORDER BY symbol, regime;
```

### Active regime periods with BH state

```sql
SELECT
    i.symbol,
    rp.regime,
    rp.started_at,
    DATE_PART('day', NOW() - rp.started_at)    AS days_in_regime,
    cs.tf_score,
    cs.mass_1d
FROM regime_periods rp
JOIN instruments   i  ON i.id  = rp.instrument_id
JOIN bh_current_state cs ON cs.instrument_id = rp.instrument_id
WHERE rp.ended_at IS NULL    -- active regimes only
ORDER BY i.symbol;
```

---

## Interpreting Regime Transitions

### SIDEWAYS → BULL/BEAR (The key transition)

This is the highest-value transition to catch. The BH mass starts accumulating during the transition, often triggering a formation event within 1-3 bars of the regime change.

**How to detect in advance**:
- Watch for BH mass approaching but not yet crossing bh_form (mass = 1.0-1.4)
- SMA crossover imminent (gap between 20-SMA and 50-SMA narrowing)
- Volatility beginning to rise (norm_vol increasing toward 1.5)

### BULL → HIGH_VOLATILITY

This is the most dangerous transition — a correction within an uptrend. The BH may fire short (bearish) while the broader trend is still bullish.

**Filter**: Require regime confirmation for 3 bars before switching position direction.

### HIGH_VOLATILITY → any direction

High-volatility regimes are followed by regime clarity within 5-15 bars historically. The market "resolves" the volatility either upward (BULL) or downward (BEAR). Monitor which way the BH fires during HIGH_VOL — it often predicts the resolution direction.

---

## Regime Performance Benchmark

From backtests across all runs, expected performance by regime:

| Regime | Avg trades/month | Win rate | Avg return |
|--------|-----------------|---------|-----------|
| BULL | 2-3 | 62% | +1.2% |
| BEAR | 1-2 | 58% | +0.9% |
| SIDEWAYS | 4-6 (Gear 2) | 68% | +0.3% |
| HIGH_VOL | 3-5 | 50% | +0.2% |

High-volatility regime performance is lowest because BH signals are noisier (many formation events, shorter duration).
