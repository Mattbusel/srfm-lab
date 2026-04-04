# Adding New Instruments

## Overview

To add a new instrument to SRFM Lab, you need to:
1. Calibrate the CF (Curvature Factor) values for each timeframe
2. Add the instrument to the configuration files
3. Fetch historical data
4. Validate BH signal quality

---

## Step 1: Calibrate CF Values

CF should be set to approximately the **RMS of typical returns** at each timeframe. The formula:

```
CF_timeframe ≈ STDDEV(log_returns, window=252_bars)
```

But use a rolling median-of-stddev to avoid outlier contamination:

```python
import pandas as pd
import numpy as np

def calibrate_cf(df: pd.DataFrame, timeframe: str) -> dict:
    """
    df: OHLCV DataFrame with 'close' column
    Returns: {'cf_15m': x, 'cf_1h': y, 'cf_1d': z}
    """
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    
    # Use rolling 60-day window, take median of stddevs
    rolling_std = log_returns.rolling(60).std()
    cf = float(rolling_std.median())
    
    # Scale by timeframe
    # (assumes 15m base: multiply up by sqrt-of-time ratios)
    tf_factors = {'15m': 1.0, '1h': 4.0**0.5, '1d': (26*252)**0.5 / (26)**0.5}
    
    return {
        'cf_15m': cf,
        'cf_1h': cf * (4.0**0.5),      # 4 fifteen-minute bars per hour
        'cf_1d': cf * ((26*252 / 26)**0.5)  # 26 15m bars per day
    }
```

For an equity index ETF with 252-day vol of 15%: `CF_1d ≈ 0.15/sqrt(252) ≈ 0.0094`.

### CF tier reference

| Asset type         | CF_1d range | CF_1h range  | CF_15m range |
|-------------------|-------------|--------------|--------------|
| US equity index   | 0.004-0.006 | 0.001-0.002  | 0.0002-0.0005|
| Commodity (energy)| 0.015-0.025 | 0.004-0.008  | 0.001-0.003  |
| Gold/metals       | 0.007-0.012 | 0.002-0.004  | 0.0006-0.001 |
| Government bonds  | 0.003-0.007 | 0.001-0.002  | 0.0003-0.0006|
| Major forex       | 0.004-0.008 | 0.001-0.003  | 0.0003-0.0007|
| Large-cap crypto  | 0.04-0.08   | 0.012-0.025  | 0.004-0.010  |
| Mid-cap crypto    | 0.08-0.15   | 0.025-0.045  | 0.008-0.015  |
| Small-cap crypto  | 0.15-0.30   | 0.045-0.090  | 0.015-0.030  |
| VIX/volatility    | 0.020-0.035 | 0.006-0.012  | 0.002-0.004  |

---

## Step 2: Set bh_form

Start with `bh_form = 1.5` for most instruments. Increase to 1.8-2.0 for:
- High-noise instruments (NG, VX, small-cap crypto)
- Instruments with frequent false breakouts
- Instruments where BH formations happen more than once per month on average

Validate: Run a backtest with the default bh_form and check Q14 (BH success rate by regime). If success rate is < 50%, raise bh_form.

---

## Step 3: Add to Configuration Files

### `config/instruments.yaml`

```yaml
instruments:
  MY_NEW_INST:
    name: "My New Instrument"
    asset_class: commodity     # or equity_index, bond, forex, crypto, volatility
    base_currency: USD
    quote_currency: USD
    alpaca_ticker: TICKER
    type: stock                # stock or crypto
    cf_15m: 0.00100
    cf_1h:  0.00330
    cf_1d:  0.01000
    bh_form: 1.5
    bh_collapse: 1.0
    bh_decay: 0.95
    corr_group: commodity_energy
    tick_size: 0.01
    is_active: true
```

### `warehouse/schema/06_seed_data.sql`

Add an `INSERT INTO instruments` row following the existing pattern.

### `tools/live_trader_alpaca.py`

Add to the `INSTRUMENTS` dict:

```python
INSTRUMENTS = {
    ...
    "MY_INST": {
        "ticker": "TICKER",
        "type": "stock",
        "cf_15m": 0.00100,
        "cf_1h": 0.00330,
        "cf_1d": 0.01000,
        "bh_form": 1.5,
    },
}
```

---

## Step 4: Fetch Historical Data

```bash
python scripts/fetch_polygon.py \
    --symbols MY_INST \
    --timeframes 1d 1h 15m \
    --start 2020-01-01
```

---

## Step 5: Validate BH Signal Quality

Run a standalone backtest and check:

```bash
# Single-instrument backtest via API
curl -X POST http://localhost:8000/api/backtest \
  -d '{"instrument": "MY_INST", "start": "2021-01-01", "end": "2024-01-01", "timeframe": "1d"}'
```

Check:
- Are BH formations occurring 2-8 times per year? (Too frequent = CF too high; too rare = CF too low)
- Is win rate for tf_score >= 3 above 55%?
- Is the formation success rate (Q14) above 50%?
- Do formations cluster in BULL/BEAR regimes vs SIDEWAYS?

If formations are too frequent, increase CF by 20% and re-test.
If formations are too rare, decrease CF by 20% and re-test.
