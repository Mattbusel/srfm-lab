# Interpreting Results

## Equity Curve

The equity curve is the most important diagnostic tool. A good BH strategy equity curve has the following characteristics:

### Shape indicators

**Good signs**:
- Grows in distinct "steps" separated by flat consolidation periods -- each step is a BH formation event
- Flat periods correspond to SIDEWAYS regime (no BH signals, Gear 2 harvesting small returns)
- Drawdowns are shallow (< 20%) and recover within 30-60 bars
- New highs are made regularly (HWM advances)

**Warning signs**:
- Long flat periods with no new highs (> 90 bars) -- signal drought
- Drawdowns that exceed the MC P5 band -- strategy is underperforming statistical expectations
- Equity curve significantly above MC P95 band -- the backtest period was anomalously favorable
- Sharp vertical drops at specific dates -- likely a regime change that wasn't handled properly

### Drawdown analysis

Key questions for each drawdown:
1. What regime was the market in when the drawdown started?
2. Were there BH signals firing? (Look at the BH state timeline)
3. How many losing trades were in the drawdown? Were they all in one instrument?
4. Did any risk circuit breakers fire?

Use `Q16_worst_drawdown_events` from `05_queries.sql` for the top drawdown events.

---

## Monte Carlo Bands

### Reading the fan chart

```
$300k ─────────────────────────────── P95 (good luck)
$200k ──────────────────────────── P75
$150k ──────────────────────── P50 (base case)
$120k ──────────────────── P25
 $90k ──────────────── P5 (bad luck)
 $50k ────────── BLOWUP THRESHOLD
```

### Decision rules from MC

| MC Output | Decision |
|-----------|---------|
| P5 terminal > initial equity | Strategy is robust to bad luck |
| P5 terminal < initial equity | Reduce position sizes or raise min_tf_score |
| P50 < historical backtest | Good -- backtest was above average; plan for P50 |
| Blowup prob > 1% | Reduce leverage immediately |
| P95 / P50 > 5x | Very high variance; strategy is a lottery ticket |

---

## Sensitivity Surface

The sensitivity surface shows how Sharpe ratio (or another metric) changes as you vary two parameters simultaneously. Key interpretations:

### Flat regions (good)

A flat region means the parameter does not matter much -- small changes don't hurt performance. If bh_form is in a flat region from 1.3 to 1.8, the parameter choice is robust.

### Sharp peaks (bad)

If performance spikes only at a narrow parameter value (e.g., bh_form = 1.47 gives Sharpe 2.1 but 1.50 gives 1.6), the strategy is overfit to that specific value. The parameter is unstable and should not be used.

### Monotonic slopes (informative)

If performance increases monotonically as you raise min_tf_score, that means: only high-conviction trades are profitable. Consider raising min_tf_score to 3 or 4 as the entry filter, at the cost of fewer trades.

---

## Trade Statistics

### Win rate by tf_score

Expected progression (from `trade_stats_by_tf` view):

| tf_score | Typical win rate | Typical avg return |
|---------|-----------------|-------------------|
| 0 | N/A (no trades taken) | - |
| 1 | 40-45% | -0.1% to +0.2% |
| 2 | 48-54% | +0.2% to +0.5% |
| 3 | 52-58% | +0.4% to +0.8% |
| 4 | 54-61% | +0.5% to +1.0% |
| 5 | 57-64% | +0.6% to +1.2% |
| 6 | 58-66% | +0.7% to +1.5% |
| 7 | 62-70% | +0.8% to +2.0% |

If your win rate for tf_score = 7 is below 55%, the BH signal may be miscalibrated or CF values need adjustment.

### Profit factor

Profit factor = total winning P&L / |total losing P&L|. Interpretation:

| Profit factor | Interpretation |
|--------------|----------------|
| < 1.0 | Losing strategy |
| 1.0 - 1.5 | Marginal (transaction costs likely eat the edge) |
| 1.5 - 2.0 | Acceptable |
| 2.0 - 3.0 | Good |
| > 3.0 | Excellent (or suspiciously overfit) |

### MFE/MAE profile

The MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion) tables tell you:

- **MFE median**: The typical peak unrealized gain. If MFE median for tf_score 7 is 2.5% but avg return is 0.8%, you are leaving money on the table with early exits.
- **MAE median**: The typical peak unrealized loss. If MAE is large relative to final P&L, your stop-loss logic may be too tight (stopping out at the worst point) or too loose (not stopping losses that continued).
- **Implied R multiple**: `MFE_p50 / |MAE_p50|`. Should be > 1.5 for a healthy risk-reward profile.

---

## Regime Performance

From `regime_performance` view:

Expected by regime:

| Regime | Strategy behavior | Expected trades |
|--------|------------------|-----------------|
| BULL | Long BH formations fire frequently | High win rate (60%+) long trades |
| BEAR | Short BH formations; harder to get right | Moderate win rate (55%) short trades |
| SIDEWAYS | Gear 2 mean reversion; small frequent wins | High win rate (65%) small returns |
| HIGH_VOL | Frequent BH signals but shorter duration | Mixed results |

If SIDEWAYS regime shows negative P&L, Gear 2 harvesting is failing. Check:
1. Z-score entry threshold (default 1.5)
2. Z-score exit threshold (default 0.3)
3. Lookback period (default 20 bars)
