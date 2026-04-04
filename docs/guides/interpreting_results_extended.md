# Extended Results Interpretation — Walk-Forward and Cross-Validation

## Walk-Forward Analysis

Walk-forward testing divides the backtest period into in-sample (IS) and out-of-sample (OOS) windows, rolling forward in time. This is the gold standard for detecting overfitting.

### Setup

```
|──────────────|──────|──────────────|──────|──────────────|──────|
  IS Window 1   OOS 1   IS Window 2   OOS 2   IS Window 3   OOS 3
  (252 bars)    (63)   (252 bars)    (63)    (252 bars)    (63)
```

For LARSA, use:
- IS window: 252 trading days (1 year)
- OOS step: 63 trading days (1 quarter)

Run Q15 from `05_queries.sql` to get the walk-forward consistency table.

### Interpreting walk-forward results

| Pattern | Interpretation |
|---------|---------------|
| All quarters positive Sharpe | Robust strategy, consistent alpha |
| 3/4 quarters positive | Acceptable; one quarter may be regime mismatch |
| 2/4 positive | Marginal; investigate the failing quarters |
| 1/4 positive | Likely overfit; parameters are period-specific |
| OOS sharpe > IS sharpe | Unusual but valid; IS period was below-average |

### The degradation ratio

```
degradation = OOS_sharpe / IS_sharpe
```

A healthy degradation ratio is 0.6–0.8 (OOS is 60-80% as good as IS). Values below 0.5 indicate overfitting. Values above 0.9 are suspicious (either IS was unlucky or OOS was lucky).

---

## Cross-Asset Validation

A robust BH strategy should work across multiple asset classes with the same parameters. If a strategy only works on ES but fails on GC and ZB, the parameters may be overfit to US equities.

Test across:
1. **Same asset class, different period**: Does the ES strategy work in 2015-2019 as well as 2020-2024?
2. **Different asset class, same parameters**: Does ES's bh_form=1.5 work on GC and ZB?
3. **Different geographic market**: If the BH concept is valid, it should work on European or Asian equity indices.

Use Q17 (`equity_vs_crypto_bh`) to directly compare performance across asset classes.

---

## Understanding the Sensitivity Tornado Chart

The tornado chart shows which parameter has the most impact on Sharpe:

```
                        BASE SHARPE = 1.84
                        
bh_form       ├────────────────────────┤  [1.62 ──── 1.84 ──── 1.79]
bh_decay      ├──────────────────┤      [1.71 ──── 1.84 ──── 1.81]
min_tf_score  ├──────────────────────┤  [1.55 ──── 1.84 ──── 1.81]
pos_floor     ├─────────┤              [1.79 ──── 1.84 ──── 1.86]
```

**Reading the chart**:
- Width = sensitivity (how much changing the parameter affects Sharpe)
- Left end = value at low perturbation (parameter × 0.75)
- Right end = value at high perturbation (parameter × 1.25)
- Center = base case

**Decision rules**:
- Wide bar = sensitive parameter; be careful about overfitting to exact value
- Narrow bar = stable parameter; safe to use rounded/round values
- Asymmetric bar (left end worse than right end) = parameter is near its optimal region; tighten monitoring

In the example above, `min_tf_score` has an asymmetric left-heavy bar: reducing min_tf_score badly hurts Sharpe (1.55), while increasing it slightly hurts (1.81). This means the current value (2) is on the low end of the stability region; consider using 3.

---

## Equity Curve Decomposition

The equity curve is the sum of contributions from:
1. **Gear 1 (BH tail capture)**: Step-function gains during formation events
2. **Gear 2 (harvest)**: Smooth, gradual gains during SIDEWAYS regimes
3. **Costs**: Commission drag (very small at Alpaca pricing)
4. **Slippage**: Bid-ask spread costs

To decompose:
```sql
-- Gear 1 trades: BH formations, regime BULL/BEAR
SELECT SUM(pnl_dollar) AS gear1_pnl
FROM trades
WHERE regime_at_entry IN ('BULL', 'BEAR') AND tf_score >= 2;

-- Gear 2 trades: SIDEWAYS regime mean reversion
SELECT SUM(pnl_dollar) AS gear2_pnl
FROM trades
WHERE regime_at_entry = 'SIDEWAYS';
```

A healthy LARSA v16 decomposition over a 3-year period:
- Gear 1: ~75% of total P&L (from 10-15 large formation events/year)
- Gear 2: ~25% of total P&L (smooth, consistent harvest income)
- High-vol regime: Often break-even or slightly negative

If Gear 2 is negative, the Z-score mean reversion parameters need adjustment.

---

## Comparing Backtests to Live Results

The "live vs backtest" comparison (Q20) is the most important validation. Common discrepancies:

| Discrepancy | Likely cause | Fix |
|-------------|-------------|-----|
| Live win rate 10% lower than backtest | Look-ahead bias in backtest | Review data loading timestamps |
| Live avg return 30% lower | Slippage underestimated | Increase slippage assumption in backtest |
| Live has more high-vol regime trades | Live data has more gaps | Normalize regime classification logic |
| Live drawdown larger than backtest P5 | Market regime shift | Reduce position sizes until regime stabilizes |
| Live Sharpe > backtest Sharpe | Lucky live period | Normal variation; plan for mean reversion |

The backtest-to-live degradation ratio (similar to IS/OOS) is typically 0.65–0.80 for well-designed strategies. A ratio below 0.5 suggests the backtest is unrealistic.
