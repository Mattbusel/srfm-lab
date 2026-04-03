# LARSA v4 — "Logical Brown Caribou" QC Backtest Analysis
Generated: 2026-04-03

## Headline Comparison

| Metric | v1 (274%) | v2 (175%) | v3 (200%) | v4 (204%) | v4 vs v1 | v4 vs v3 |
|--------|-----------|-----------|-----------|-----------|----------|----------|
| Net Return | 274% | 175.1% | 200.2% | 204.4% | -69.6pp | +4.2pp |
| Net P&L | $2,747k | ~$1,751k | $2,002k | $2,043k | -$704k | +$41k |
| Max Drawdown | 29.9% | 48.1% | 40.2% | 40.0% | +10.1pp | -0.2pp |
| Sharpe | 4.289 | — | 0.516 | 0.537 | -3.752 | +0.021 |
| Win Rate | 54.9% | 49.9% | 52.4% | 52.9% | -2.0pp | +0.5pp |
| Trades | 377 | 513 | 485 | 486 | +109 | +1 |
| CAR | — | — | — | 17.2% | — | — |
| Total Fees | — | — | — | $95,127 | — | — |

Note: v1 Sharpe of 4.289 was computed differently (per-trade Sharpe). QC portfolio Sharpe for v1 was ~0.83 in its final rolling window. The v4 portfolio Sharpe of 0.537 is the full-period QC value.

---

## Annual P&L vs All Versions

| Year | v1 | v3 | v4 | v4-v1 | v4-v3 | Notes |
|------|----|----|-----|-------|-------|-------|
| 2018 | +$94k | -$153k | -$140k | -$234k | +$13k | BEAR regime — both v3/v4 badly negative |
| 2019 | +$93k | -$47k | -$29k | -$122k | +$18k | BEAR regime improvement but still negative |
| 2020 | +$629k | +$629k | +$340k | -$289k | -$289k | Big drop vs v1/v3 — fewer momentum entries |
| 2021 | +$109k | +$456k | +$116k | +$7k | -$340k | v3 benefited from stop_loss relaxation here |
| 2022 | +$171k | +$250k | +$182k | +$11k | -$68k | Only 12 trades — gate very selective |
| 2023 | +$690k | +$863k | +$939k | +$249k | +$76k | Best v4 year, outperforms both predecessors |
| 2024 | +$1,116k | +$112k | +$730k | -$386k | +$618k | NQ cap rescued 2024 vs v3's stop_loss disaster |

**2024 recovery:** v3 was destroyed by a cascading NQ stop_loss in 2024 (+$112k). v4's NQ notional cap limits single-instrument risk, recovering +$618k vs v3 in 2024.

**2020 regression:** v4 earned only $340k vs v1/v3's ~$629k in the COVID rally. The notional cap on NQ reduced size on the biggest single-direction run of the backtest.

---

## NQ Notional Cap Verification

**v3 worst NQ losses (from prior analysis):** -$265k, -$177k, -$121k  
**v4 largest NQ single-trade losses:**

| Date | Direction | P&L | Qty | Notes |
|------|-----------|-----|-----|-------|
| 2024-10-14 | Buy | -$88,400 | 16 | Largest NQ loss in v4 |
| 2023-05-02 | Buy | -$69,615 | 21 | |
| 2024-06-18 | Buy | -$69,600 | 20 | |
| 2019-02-06 | Buy | -$53,400 | 40 | Pre-cap era sizing |
| 2023-05-30 | Buy | -$37,760 | 21 | |

**Verdict: Cap is working.** Largest NQ loss dropped from -$265k (v3) to -$88k (v4) — a 67% reduction in worst-case NQ single-trade exposure. The -$40 quantity in 2019 suggests the cap was less effective in early years when ES/NQ prices were lower (notional per contract was smaller), but the 2024 losses (qty 16-20) confirm the cap is binding at current price levels.

**NQ total P&L in 2024:** -$27k across 33 trades (WR 57%). The cap prevented the large individual losses but NQ still marginally negative for the year.

---

## BEAR Gate Verification (2018-2019)

**Blame report: 2018-2019 — 174 trades, Net -$168,721, WR 43.7%**

| Year | Trades | Buys | Sells | P&L |
|------|--------|------|-------|-----|
| 2018 | 68 | 46 | 22 | -$140,192 |
| 2019 | 106 | 83 | 23 | -$28,529 |

**Key observation:** The BEAR gate did NOT block long (Buy) trades in 2018-2019. There are 129 Buy trades across the two years. The gate appears to weight/reduce size rather than gate outright, OR the regime classifier still allowed buys during the 2018-2019 bear period (these years were not a full-year bear — S&P had significant rallies in H1 2018 and H1 2019).

**What the gate did accomplish:** v3 lost -$153k in 2018 vs v4's -$140k (+$13k improvement). v3 lost -$47k in 2019 vs v4's -$29k (+$18k improvement). Marginal improvement, not a structural fix.

**Worst 2018-2019 trades (losses not blocked):**
- 2018-05-14: ES Buy -$97,500, YM Buy -$89,790 (same day — correlated crash event)
- 2019-07-04: ES Buy -$65,025
- 2019-02-06: NQ Buy -$53,400, ES Buy -$49,088

The May 2018 and Feb 2019 drawdowns are the primary pain points. The gate is not blocking these entries — the regime signal still reads BULL or NEUTRAL during these periods.

**Instrument breakdown 2018-2019:**
- ES: -$94k (42% WR)
- NQ: +$132k (50% WR) — NQ was the only profitable instrument in the bear period
- YM: -$206k (39% WR) — YM is the primary bear-period bleeder

---

## Detailed P&L Summary

### By Instrument (full backtest)
| Instrument | P&L | Trades | Notes |
|------------|-----|--------|-------|
| ES | +$989k | 204 | Backbone of returns |
| NQ | +$605k | 159 | Cap reduced upside but also losses |
| YM | +$545k | 123 | Weakest per-trade avg |

### By Direction
| Direction | P&L | Trades | Notes |
|-----------|-----|--------|-------|
| Buy | +$1,804k | 379 | Long bias dominates |
| Sell | +$335k | 107 | Shorts additive but small |

### Top 5 Winning Trades
| Date | Symbol | Direction | P&L |
|------|--------|-----------|-----|
| 2023-12-13 | YM15Z23 | Buy | +$227,045 |
| 2023-11-13 | ES15Z23 | Buy | +$201,250 |
| 2024-11-22 | YM20Z24 | Buy | +$195,360 |
| 2020-11-06 | YM18Z20 | Buy | +$192,610 |
| 2023-11-13 | YM15Z23 | Buy | +$169,920 |

### Top 5 Losing Trades
| Date | Symbol | Direction | P&L |
|------|--------|-----------|-----|
| 2024-06-18 | ES21M24 | Buy | -$114,800 |
| 2024-11-29 | YM20Z24 | Buy | -$99,735 |
| 2018-05-14 | ES15M18 | Buy | -$97,500 |
| 2024-10-17 | YM20Z24 | Buy | -$96,900 |
| 2024-04-09 | YM21M24 | Sell | -$96,330 |

**Average trade duration:** 13h 15m (median 2h — bimodal: short scalps and overnight holds)

---

## edge.py: v3 vs v4 Comparison

```
v3 trades: 485  net: +$2,101,549
v4 trades: 486  net: +$2,138,586

New wins  (in v4, not v3):  +$92,115  (4 trades)
New losses(in v4, not v3):  -$32,000  (2 trades)
Lost wins (in v3, not v4):  +$27,205  (4 trades)  [missed]
Fixed losses (v3 loss -> v4 smaller): +$844,040

Net edge: +$37,038  v4 BETTER by $37,038
Biggest regressor: 2022  v3 +$251,950  v4 +$182,380  (-$69,570)
```

The NQ cap "fixed" $844k in losses (mostly 2024 NQ blowup) but the net improvement is only +$37k because 2022 regressed -$69.5k and some winning trades were missed.

---

## Key Findings

### 1. NQ Notional Cap Works — But 2020 Upside Was Sacrificed
The cap successfully reduced the worst-case NQ loss from -$265k to -$88k (67% reduction) and rescued 2024 from v3's stop_loss disaster (+$618k improvement that year). However, 2020's COVID recovery rally — where NQ ran harder than ES/YM — was trimmed: v4 earned $340k vs v1/v3's $629k. The cap is correctly managing tail risk but cuts winners symmetrically with losers.

### 2. BEAR Gate Is Ineffective for 2018-2019
Despite having a BEAR regime gate, v4 still placed 129 Buy trades in 2018-2019, losing -$168k. The gate appears to be a soft weight adjustment rather than a hard block, or the regime classifier is not reading 2018-2019 as a bear market (reasonable — both years had extended bull phases). YM was the primary bleeder (-$206k over 2 years). The gate did marginally help (+$13k in 2018, +$18k in 2019 vs v3) but not structurally.

### 3. 2023 Breakout Exceeds All Prior Versions
v4's 2023 P&L of +$939k is the best of any version for that year (v1: +$690k, v3: +$863k), driven by two enormous YM/ES November-December momentum trades. This suggests the regime and sizing logic is well-calibrated for high-volatility trend years. However, this creates dependency: remove those two trades and 2023 would be ~$500k.

---

## Remaining Failure Modes

1. **2018-2019 bear regime longs not blocked** — YM -$206k in 2 years suggests YM is particularly trend-following unfriendly in sideways/bear markets
2. **2020 underperformance vs v1** — NQ cap cost ~$289k in the best single-year momentum event
3. **Large single-day correlated losses** — May 2018 (-$187k in 1 day across ES+YM) and Oct 2024 (-$96k) suggest intra-day portfolio correlation is not managed
4. **2024 NQ still net negative** (-$27k on 33 trades, 57% WR) — even with the cap, NQ is not adding value in 2024

---

## v5 Hypotheses

### H1: Hard YM gate in BEAR regime
YM was -$206k in 2018-2019 and is the least liquid of the three. A hard block on YM longs when regime = BEAR could save ~$150-200k without missing the same trades in ES/NQ.

### H2: Correlated same-day loss limit (portfolio heat)
The May 2018 event (-$187k on a single day across ES+YM) and Nov 2024 could be mitigated by a daily portfolio loss circuit-breaker (e.g., stop all new entries after -$75k intraday).

### H3: NQ cap tiered by regime
Current cap is flat. In BULL regimes, allow full NQ size (capture 2020-style rallies). In NEUTRAL/BEAR, apply the cap. This would let 2020 run while still protecting 2024.

### H4: Improve 2022 performance
v4 only made 12 trades in 2022 (+$182k). The strategy went very quiet. Investigate whether the regime filter was overly restrictive — 2022 was a persistent bear with strong short setups that might have been gateable.

---

## Run Configuration
- **QC Run Name:** Logical Brown Caribou
- **Version:** larsa-v4
- **Backtest Period:** 2018-02-07 to 2024-12-23
- **Starting Equity:** $1,000,000
- **Ending Equity:** $3,043,460
- **Total Fees:** $95,127
