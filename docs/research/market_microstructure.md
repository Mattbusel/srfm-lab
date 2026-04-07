# Market Microstructure Considerations

## Slippage Model

The BH engine uses a simplified slippage model for backtesting. Real slippage has three components:

### 1. Bid-ask spread

The bid-ask spread at execution time is the minimum slippage for any market order. Typical spreads:

| Instrument | Typical spread | Spread as % of price |
|-----------|---------------|---------------------|
| ES (via SPY) | $0.01 | 0.002% |
| NQ (via QQQ) | $0.01 | 0.003% |
| CL (via USO) | $0.01 | 0.013% |
| BTC/USD | $10–$50 | 0.015–0.080% |
| ETH/USD | $0.5–$2 | 0.015–0.060% |

### 2. Market impact

For larger orders relative to daily volume, the act of entering pushes the price. At our position sizes ($50K–$400K notional), market impact is negligible for liquid ETFs like SPY/QQQ but can be meaningful for VIXY, UNG.

### 3. Timing slippage

BH entry signals trigger at bar close. The actual execution fills on the next bar's open, which may gap from the close. This gap-at-open slippage is the primary source of live vs backtest divergence for overnight events.

**Calibrating slippage from live data** (Q09 in `05_queries.sql`):

```sql
SELECT symbol, entry_hour, avg_slippage_bps, p90_slippage_bps
FROM slippage_analysis
ORDER BY symbol, entry_hour;
```

Compare the `avg_slippage_bps` from live trades to the assumed slippage in backtests. If live slippage is 2x the backtest assumption, increase the backtest slippage parameter and recompute performance.

---

## Volume Profile and VWAP Considerations

The BH signal fires at bar close regardless of where within the bar the extreme move occurred. This can cause:

**Early formation**: The bar makes a large move in the first 15 minutes, mass crosses bh_form, but by close the price reverts 50% of the move. Entry at close may be at the halfway point rather than the extremum.

**Mitigation**: Consider using a VWAP-based entry:
- If bar's VWAP is within 0.5% of close, standard entry
- If close is significantly above VWAP (extended into close), wait for next open

This logic is available in the strategy but disabled by default (reduces trade frequency by ~15%).

---

## Overnight and Weekend Gaps

Gap events can cause BH mass to spike to the maximum cap (20.0) in a single bar. This is intentional -- a large gap IS a momentum event. However, gaps also cause:

1. **Entry at gap price**: If you can't enter before the gap (it occurs after hours), you enter at a worse price than the bar's close. This gap slippage can be 1–5% for commodity futures.

2. **False formation on mean-reverting gaps**: Some gaps (e.g., earnings-driven) quickly reverse. The BH forms, you enter, and the stock reverses the gap over the next 3 bars.

**Countermeasure**: Require at least 2 bars of BH persistence (mass >= bh_form for 2 consecutive bars) before entry on crypto (24/7 market). For stock ETFs, weekend gaps from Sunday open to Monday open are real momentum events.

---

## Trading Hours Considerations

Stock ETF positions:
- Entries only during regular market hours (9:30 AM – 4:00 PM ET)
- 15-minute bar closes are clean during market hours
- After-hours bars are excluded from BH calculation (no data in Alpaca free tier)

Crypto positions:
- 24/7 trading, no gap risk
- Weekend BH formations are valid
- Liquidity is slightly lower on weekends; increase slippage assumption by 30%

BH formation timing by hour (from Q27 in `05_queries.sql`):

```
Most formations occur during:
  - Pre-market: 8-9 AM ET (CL, NG: energy inventory reports)
  - US open: 9:30-10:30 AM ET (ES, NQ, YM: most volatile hour)
  - London/NY overlap: 8-11 AM ET (GC, ZB)
  - US close: 3:30-4 PM ET (ES, NQ: late-day momentum)
  - Bitcoin: No clear intraday pattern (24/7)
```
