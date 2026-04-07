# Parameter Calibration Guide

## Overview

The BH engine has four key parameters per instrument per timeframe: `CF`, `bh_form`, `bh_collapse`, and `bh_decay`. This guide covers how to calibrate each from historical data.

---

## CF Calibration

**Goal**: CF should equal approximately the 1-sigma daily return of the instrument.

### Automated calibration

```python
import pandas as pd
import numpy as np

def calibrate_cf(symbol: str, timeframe: str = "1d") -> float:
    from spacetime.engine.data_loader import load_bars
    df = load_bars(symbol, timeframe, "2020-01-01", "2024-01-01")
    log_rets = np.log(df["close"] / df["close"].shift(1)).dropna()
    
    # Use 60-period rolling std, take median (robust to outlier periods)
    rolling_std = log_rets.rolling(60).std()
    cf = float(rolling_std.dropna().median())
    return round(cf, 6)

# Example:
for symbol in ["ES", "NQ", "CL", "GC", "BTC"]:
    cf = calibrate_cf(symbol)
    print(f"{symbol}: CF_1d = {cf:.5f}")
```

### Cross-timeframe scaling

Once you have `CF_1d`, derive other timeframes:

```
CF_1h  = CF_1d  / sqrt(6.5)       # 6.5 trading hours per day
CF_15m = CF_1h  / sqrt(4)          # 4 fifteen-minute bars per hour
```

The sqrt-of-time rule assumes i.i.d. returns -- in practice, this slightly underestimates intraday CF due to intraday volatility clustering, so add a 10% buffer:

```
CF_1h  = CF_1d  / sqrt(6.5) * 1.1
CF_15m = CF_1h  / sqrt(4)   * 1.1
```

---

## bh_form Calibration

**Goal**: BH formations should occur roughly 4-8 times per year on the daily timeframe.

### Check current formation rate

```sql
-- From warehouse/schema/05_queries.sql Q22
SELECT asset_class, timeframe, formation_rate_pct_per_bar
FROM bh_formation_rate;
```

Target: 4-8 formations per year on daily = 4/252 to 8/252 = 1.6% to 3.2% of bars.

### Adjustment rule

- Formation rate > 3.2% per bar: increase bh_form by 0.2
- Formation rate < 1.6% per bar: decrease bh_form by 0.2
- Repeat until rate is in target range

---

## bh_decay Calibration

**Goal**: BH should persist long enough to capture a meaningful trend move (5-20 bars typically) but not so long that it captures the reversal.

The half-life of the BH mass (time for mass to decay from bh_form to bh_collapse without new momentum input) is:

```
half_life = -log(bh_collapse / bh_form) / log(1/decay)
         = log(bh_form / bh_collapse) / log(1 / decay)
```

For default parameters (bh_form=1.5, bh_collapse=1.0, decay=0.95):
```
half_life = log(1.5/1.0) / log(1/0.95) = 0.405 / 0.051 ≈ 7.9 bars
```

So without additional momentum input, the BH collapses after ~8 bars.

For a faster-decaying BH (decay=0.90): half_life ≈ 3.9 bars.

**Guideline**: Set decay so that half_life ≈ (average_trade_duration / 2). If your average winning trade lasts 15 bars, decay ≈ 0.95 gives half_life ≈ 8 bars -- the BH expires naturally around the time a normal trade would exit.

---

## Min TF Score Calibration

From `Q04_optimal_tf_threshold`:

```sql
SELECT asset_class, tf_score, win_rate_pct, sharpe_proxy, profit_factor
FROM trade_stats_by_tf
ORDER BY asset_class, tf_score;
```

The optimal min_tf_score is where `sharpe_proxy` peaks. For most instruments this is 3-4. Below 3, noise trades drag down the overall Sharpe. Above 5, there are too few trades for statistical significance.

**Practical rule**: Use min_tf_score = 2 for liquid instruments (ES, NQ, BTC), 3 for semi-liquid (CL, ETH), 4 for noisy instruments (NG, VX, small-cap crypto).
