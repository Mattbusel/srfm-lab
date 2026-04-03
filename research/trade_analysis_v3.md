# LARSA v3 — "Measured Red Anguilline" QC Backtest Analysis
Generated: 2026-04-03

---

## Headline Comparison

| Metric | v1 (274%) | v2 (175%) | v3 (200%) | v3 vs v1 |
|--------|-----------|-----------|-----------|----------|
| Net Return | 274% | 175% | 200.2% | -73.8pp |
| Max Drawdown | 29.9% | 48.1% | 40.2% | +10.3pp worse |
| Sharpe (portfolio) | 4.289 | ~2.5 | 0.516 | -3.773 |
| Win Rate | 54.9% | 49.9% | 52.4% | -2.5pp |
| Trades | 377 | 513 | 485 | +108 |
| Net P&L | $2,747,046 | ~$1,751,000 | $2,001,585 | -$745,461 |
| Total Fees | — | — | $99,964 | — |
| Sortino (portfolio) | — | — | 0.337 | — |

> **Note:** v1 Sharpe of 4.289 appears to be a trade-level Sharpe from QC's tradeStatistics, not the portfolio-level Sharpe. v3's trade-level Sharpe is 0.104; portfolio-level Sharpe is 0.516. These are likely apples-to-oranges across runs — flag for reconciliation.

---

## Annual P&L vs v1

| Year | v1 P&L | v3 P&L (net) | Delta | v3 Trade Ct | v3 WR | Notes |
|------|--------|--------------|-------|-------------|-------|-------|
| 2018 | +$93,848 | -$153,126 | -$247k | 68 | 46% | Severe regression — see below |
| 2019 | +$93,202 | -$46,711 | -$140k | 106 | 42% | Worst WR year; mass early trades firing |
| 2020 | +$628,838 | +$334,293 | -$295k | 37 | 59% | CONV_SIZE cap killing large wins |
| 2021 | +$108,965 | +$48,425 | -$61k | 92 | 51% | High trade count hurts quality |
| 2022 | +$171,055 | +$250,071 | +$79k | 12 | 58% | v3 BEST year vs v1 |
| 2023 | +$690,225 | +$863,547 | +$173k | 70 | 57% | v3 BEST year — strongest absolute |
| 2024 | +$1,115,855 | +$705,086 | -$411k | 100 | 62% | NQ late-2024 blowups destroyed gains |

**Total period (net):** v3 = $2,001,585 vs v1 = $2,747,046 (-$745k, -27%)

---

## Trade Distribution Overview

| Metric | Value |
|--------|-------|
| Total trades | 485 |
| Winning trades | 254 (52.4%) |
| Losing trades | 231 (47.6%) |
| Avg trade duration | 12h 44m |
| Median trade duration | 2h 00m |
| Avg winning trade | +$29,654 |
| Avg losing trade | -$23,509 |
| Largest single win | +$224,550 (YM Dec-23, Buy, qty=90) |
| Largest single loss | -$265,200 (NQ Oct-24, Buy, qty=48) |
| Total fees | $99,964 |

---

## P&L by Instrument (net of fees)

| Instrument | Net P&L | Share |
|------------|---------|-------|
| ES | $992,042 | 49.6% |
| NQ | $648,663 | 32.4% |
| YM | $360,880 | 18.0% |

---

## P&L by Direction (net of fees)

| Direction | Net P&L | Share |
|-----------|---------|-------|
| Buy | $1,586,571 | 79.3% |
| Sell | $415,013 | 20.7% |

The strategy is overwhelmingly long-biased. Sells only contribute 20.7% of profit.

---

## Top 5 Winning Trades

| Rank | Entry | Symbol | Dir | Qty | Net P&L |
|------|-------|--------|-----|-----|---------|
| 1 | 2023-12-13 | YM15Z23 | Buy | 90 | +$224,163 |
| 2 | 2024-11-22 | YM20Z24 | Buy | 98 | +$199,009 |
| 3 | 2023-11-13 | ES15Z23 | Buy | 45 | +$196,682 |
| 4 | 2020-11-06 | YM18Z20 | Buy | 34 | +$192,464 |
| 5 | 2022-12-21 | NQ17H23 | Sell | 28 | +$184,400 |

---

## Top 5 Losing Trades

| Rank | Entry | Symbol | Dir | Qty | Net P&L |
|------|-------|--------|-----|-----|---------|
| 1 | 2024-10-14 | NQ20Z24 | Buy | 48 | -$265,406 |
| 2 | 2024-06-18 | NQ21M24 | Buy | 56 | -$176,641 |
| 3 | 2024-12-09 | NQ20Z24 | Buy | 52 | -$121,384 |
| 4 | 2024-12-11 | NQ20Z24 | Buy | 51 | -$120,579 |
| 5 | 2024-11-29 | YM20Z24 | Buy | 112 | -$102,962 |

**Pattern:** Four of the five worst losses are in 2024, three are NQ Buys in Q4 2024. This is a concentrated failure mode.

---

## Well Analysis (8h clustering)

| Metric | Value |
|--------|-------|
| Total wells | 273 |
| Winning wells | 141 (51.6%) |
| Losing wells | 132 (48.4%) |
| Total net P&L check | $2,001,584 ✓ |

### Top 10 Wells

| Net P&L | Start Date | Trades |
|---------|------------|--------|
| +$349,597 | 2023-11-13 | 3 |
| +$328,087 | 2024-11-22 | 2 |
| +$244,955 | 2023-12-12 | 3 |
| +$192,464 | 2020-11-06 | 1 |
| +$184,806 | 2024-02-08 | 5 |
| +$176,986 | 2023-11-17 | 4 |
| +$154,550 | 2022-12-20 | 2 |
| +$143,131 | 2024-07-08 | 5 |
| +$134,862 | 2023-07-12 | 6 |
| +$126,781 | 2023-11-21 | 4 |

### Bottom 10 Wells

| Net P&L | Start Date | Trades |
|---------|------------|--------|
| -$98,333 | 2024-10-17 | 1 |
| -$98,519 | 2024-04-09 | 1 |
| -$102,879 | 2019-02-06 | 2 |
| -$106,568 | 2024-09-06 | 2 |
| -$115,796 | 2024-10-11 | 3 |
| -$116,695 | 2024-11-11 | 2 |
| -$120,579 | 2024-12-11 | 1 |
| -$121,384 | 2024-12-09 | 1 |
| -$170,491 | 2024-06-18 | 4 |
| -$189,064 | 2018-05-14 | 2 |

**8 of the 10 worst wells are in 2018 or 2024.** The bookend problem — same structural issue as v1/v2.

---

## CONV_SIZE Impact Analysis (solo-BH cap 0.65 → 0.40)

The CONV_SIZE reduction was meant to limit runaway position sizing on borderline setups. The evidence is mixed:

**Wins were NOT protected:**
- Largest win: +$224,550 at qty=90 (YM, well-sized). This trade was still allowed to be large because YM was in a strong regime.
- Top 5 wins still hit $192k–$224k — the cap did not materially clip the best trades.
- However, the NQ 2020 event (nov-06, +$192k at qty=34) shows NQ was already being constrained: qty=34 is small vs the YM trades at qty=90–98.

**Losses were NOT contained:**
- Largest loss: -$265,200 at qty=48 (NQ Oct-24). Under the old 0.65 cap this would have been larger — but the damage is still catastrophic.
- The worst losses cluster in 2024 NQ Buys. The CONV_SIZE cap helped (qty=48–56 rather than potentially 70+), but NQ's own multiplier ($20/pt) means the dollar damage is severe regardless.
- YM Nov-29 loss: qty=112 — this is the cap clearly allowing a very large YM position. YM at $5/pt is 4x less dangerous than NQ at $20/pt, but qty=112 indicates the per-contract cap was maxed out.

**Conclusion:** CONV_SIZE 0.40 partially clipped NQ losses but did not fix the core problem. The solo-BH cap needs to be instrument-aware (or notional-capped), not just contract-count-capped.

---

## pos_floor v3 Impact (ctl>=5 vs ctl>=3)

Raising the pos_floor trigger from ctl>=3 to ctl>=5 was designed to reduce noise trades.

**Evidence:**
- 2018: 68 trades at 46% WR — still firing heavily, early regime noise unchanged. The ctl>=5 threshold did not prevent bad early-2018 entries.
- 2019: 106 trades at 42% WR — this is the *worst* trade count year. Puzzling: the ctl>=5 gate was supposed to reduce count. This suggests BEAR_FAST (the 2019 regime) generates many medium-confidence signals that still clear ctl=5.
- 2022: Only 12 trades, 58% WR — the low count here is correct (range-bound year, rightly cautious). This is pos_floor working as intended.
- 2021: 92 trades is still very high. The ctl>=5 gate has not tamed 2021 volume.

**Conclusion:** pos_floor ctl>=5 meaningfully reduced trading in quiet/choppy regimes (2022) but did NOT reduce the flood in volatile/trending regimes (2019, 2021). This implies signals in volatile periods naturally clear ctl=5 easily. A higher regime-conditional threshold (e.g. ctl>=7 in BEAR_FAST) is needed.

---

## Key Findings

### Finding 1: 2024 NQ Long Concentration is the Largest P&L Drain
Three NQ Buy trades in Oct–Dec 2024 alone cost -$507k net. These are not stop-loss failures — they are maximum-conviction entries (qty=48–56) that simply caught strong adverse moves. The 2024 STOP_LOSS regression from v2 is partially addressed, but NQ Q4 2024 remains a structural vulnerability. If these three trades had been blocked, v3 would be at ~$2.5M net (closer to v1). This is a signal-quality problem, not a sizing problem.

### Finding 2: 2018–2019 Early Period Still Destroys Value (-$200k combined)
v1 made +$187k in 2018–2019; v3 lost -$200k — a $387k swing. This is the same BEAR_FAST regime problem visible in v2 (-$216k in 2018). The CONV_SIZE and pos_floor changes did not fix this. The algorithm is generating too many low-quality entries in strong bear regimes. A bear regime gate (halt new long entries when regime = BEAR_FAST) would directly address this.

### Finding 3: 2022–2023 is Genuinely Better Than v1
v3 outperforms v1 in both 2022 (+$79k delta) and 2023 (+$173k delta), contributing +$252k of outperformance in the mid-period. This confirms that the core LARSA logic has improved — the problem is entirely at the tails (early bear regime, late-2024 NQ). The 2023 well cluster (Nov 13, Dec 12, Nov 17, Nov 21) is exceptional, generating +$907k in 4 weeks of November–December 2023.

### Finding 4: Sharpe Ratio Collapse (4.289 → 0.516 portfolio)
The Sharpe ratio is dramatically lower than v1. This is partly definitional (v1 may be reporting trade-level Sharpe; v3 portfolio Sharpe is 0.516) but even so the risk-adjusted performance has degraded. High trade count years (2019: 106, 2021: 92) with poor WR are inflating volatility without proportional returns.

---

## Recommended v4 Changes

### 1. Bear Regime Long Gate (HIGH PRIORITY)
Add explicit regime gate: when BEAR_FAST is active, block new long (Buy) entries entirely or require ctl >= 9 (vs current 5). This directly targets the 2018/2019 drain. Expected impact: recover $200–400k of v1's edge in the early period.

### 2. NQ Notional Position Cap (HIGH PRIORITY)
Replace contract-count CONV_SIZE cap with a notional cap: max position notional = $X regardless of instrument. NQ at $20/pt means qty=48 × $20 × 100-pt move = $96k per 100-pt move. Cap all instruments at e.g. $5M notional. This would reduce NQ qty from 48–56 to ~25–30, halving the worst losses while preserving signal-direction benefit.

### 3. ctl Threshold: Regime-Conditional (MEDIUM PRIORITY)
Make pos_floor threshold regime-conditional: ctl>=5 in BULL/NEUTRAL, ctl>=7 in BEAR_FAST/HIGH_VOL. This targets the 2019 problem (106 trades, 42% WR) where medium-quality signals were still clearing ctl=5 in a bad regime.

### 4. 2024 Q4 NQ Review (MEDIUM PRIORITY)
Examine what signal triggered the Oct-14, Jun-18, Dec-09, Dec-11 NQ entries specifically. Are these large-timeframe momentum entries on FOMC/macro days? If so, an event-day filter (no new entries within 4h of scheduled macro events) may eliminate the worst losses.

### 5. Sell-Side Underutilization (LOW PRIORITY)
Sells generate only 20.7% of profits but presumably ~50% of opportunities. Investigate whether short entry criteria are materially more restrictive than long criteria. In bear regimes, sells should be the *primary* vehicle — the asymmetry here is suspicious.

---

## Summary

v3 is a partial improvement over v2 (200% vs 175%, drawdown 40% vs 48%) but has regressed vs v1 (274%). The 2022–2023 period shows genuine improvement. The remaining gap to v1 is almost entirely explained by two failure modes: (1) bear-regime 2018–2019 long entries, and (2) high-conviction NQ Buy entries in 2024 Q4. Both are addressable with targeted gates. The core signal quality has improved; the regime-awareness and instrument-specific risk management have not.
