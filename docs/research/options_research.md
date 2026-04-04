# Options Analytics Usage

## Overview

The SRFM lab includes options research tools for hedging BH positions and analyzing volatility regime signals. Options data is stored in `iv_surface` and `atm_iv` tables.

---

## IV Rank as Regime Signal

IV rank measures where current implied volatility sits relative to its historical range:

```
IV_rank = (current_IV - min_IV_1y) / (max_IV_1y - min_IV_1y)
```

**BH × IV rank synergy**:
- High IV rank (> 0.7) + BH formation: Options are expensive; prefer stock/futures entry, not options
- Low IV rank (< 0.3) + BH formation: Options are cheap; consider buying calls/puts to express the BH signal with defined risk
- IV rank > 0.8 with no BH: Potential mean-reversion trade in IV via short straddles

---

## Querying IV Data

```sql
-- Current ATM IV for all instruments
SELECT
    i.symbol,
    a.iv_30d,
    a.iv_rank,
    a.iv_percentile,
    a.timestamp
FROM atm_iv a
JOIN instruments i ON i.id = a.instrument_id
WHERE a.timestamp = (
    SELECT MAX(timestamp) FROM atm_iv WHERE instrument_id = a.instrument_id
)
ORDER BY a.iv_rank DESC;

-- IV term structure (spot to 180d)
SELECT
    i.symbol,
    a.iv_7d, a.iv_30d, a.iv_60d, a.iv_90d, a.iv_180d,
    a.iv_30d - a.iv_7d    AS contango_front,
    a.iv_180d - a.iv_30d  AS contango_back
FROM atm_iv a
JOIN instruments i ON i.id = a.instrument_id
ORDER BY i.symbol;
```

---

## Hedging BH Positions with Options

When a BH forms and a position is entered, consider the following hedge structures:

### 1. Protective put (long position)

Buy 1-month ATM put when entering a long BH position. Cost: IV_30d / 12 ≈ 1-2% premium.

Trade-off: Eliminates left tail (gap-down risk) but reduces net expected return by the premium cost.

Recommended when: IV_rank < 0.3 (cheap puts) AND holding a large position (tf_score 6-7).

### 2. Collar (long position)

Buy ATM put + sell OTM call. Net premium near zero. Caps upside at call strike, floors downside at put strike.

Recommended when: IV_rank > 0.5 (rich options; selling a call subsidizes the put significantly).

### 3. Bear spread (short position)

Buy ATM put + sell OTM put. Fixed cost, defined max profit.

Recommended when: BH fires short AND IV_rank < 0.4.
