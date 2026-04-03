# LARSA v3 Well Forensics — "Measured Red Anguilline"

> Generated 2026-04-03 | 485 trades | 273 wells | 2018-2024

---

## Well Statistics vs v1

| Metric | v1 | v3 | Delta |
|--------|----|----|-------|
| Total Wells | 263 | 273 | +10 |
| Well Win Rate | 54.4% | 52.0% | -2.4pp |
| Well Avg P&L | $11,034 | $7,698 | $-3,336 |
| Single-inst wells | 216 | 178 | -38 |
| Single-inst win rate | 50.0% | 50.0% | +0.0pp |
| Single-inst avg P&L | $2,496 | $2,457 | $-39 |
| Single-inst total P&L | $539,182 | $437,330 | $-101,852 |
| Multi-inst wells | 47 | 95 | +48 |
| Multi-inst win rate | 74.5% | 55.8% | -18.7pp |
| Multi-inst avg P&L | $50,272 | $17,518 | $-32,754 |
| Multi-inst total P&L | $2,362,805 | $1,664,219 | $-698,586 |
| Multi-inst % of gross P&L | 81.4% | 79.2% | -2.2pp |
| Gross P&L | $2,901,988 | $2,101,549 | $-800,439 |
| Net P&L | $2,747,046 | $2,001,584 | $-745,461 |
| Max Drawdown | 29.9% | 37.5% | +7.7pp |
| Sharpe | 4.289 | 4.760 | +0.471 |

**Key finding:** v3 has 95 multi-instrument wells vs v1's 47 (+48), but multi-inst win rate collapsed from 74.5% to 55.8% (-18.7pp). This is the primary driver of lower gross P&L. v3 also added 108 more trades (485 vs 377) while generating $800k less gross P&L — the extra activity is dilutive.

---

## Annual Well Counts and P&L

| Year | v1 Wells | v1 P&L | v3 Wells | v3 P&L | Delta P&L |
|------|----------|--------|----------|--------|-----------|
| 2018 | 37 | $93,848 | 39 | $-136,765 | $-230,612 |
| 2019 | 47 | $93,202 | 53 | $-27,914 | $-121,116 |
| 2020 | 24 | $628,838 | 24 | $340,115 | $-288,722 |
| 2021 | 46 | $108,965 | 48 | $60,805 | $-48,160 |
| 2022 | 10 | $171,055 | 11 | $251,950 | $+80,895 |
| 2023 | 42 | $690,225 | 39 | $878,072 | $+187,848 |
| 2024 | 57 | $1,115,855 | 59 | $735,285 | $-380,570 |

---

## NQ Notional Exposure Analysis

NQ futures multiplier: $20/point. All exposures are gross notional at entry.

### Top 10 NQ Trades by Notional

| Date | Dir | Qty | Entry | Notional | Portfolio | % Portfolio | P&L |
|------|-----|-----|-------|----------|-----------|-------------|-----|
| 2024-12-09 15:00:00 | Buy | 52 | 21,623.75 | $22,488,700 | $3,207,685 | 701.1% | $-121,160 |
| 2024-06-18 19:00:00 | Buy | 56 | 19,924.25 | $22,315,160 | $2,697,328 | 827.3% | $-176,400 |
| 2024-12-11 20:00:00 | Buy | 51 | 21,800.25 | $22,236,255 | $3,159,241 | 703.8% | $-120,360 |
| 2024-12-06 18:00:00 | Buy | 51 | 21,619.25 | $22,051,635 | $3,167,869 | 696.1% | $40,035 |
| 2024-12-05 15:00:00 | Buy | 51 | 21,510.75 | $21,940,965 | $3,148,198 | 696.9% | $19,890 |
| 2024-06-18 15:00:00 | Buy | 55 | 19,886.75 | $21,875,425 | $2,697,328 | 811.0% | $26,675 |
| 2024-12-04 19:00:00 | Buy | 50 | 21,472.25 | $21,472,250 | $3,108,663 | 690.7% | $39,750 |
| 2024-06-12 19:00:00 | Buy | 55 | 19,518.00 | $21,469,800 | $2,691,508 | 797.7% | $-28,875 |
| 2024-06-06 18:00:00 | Buy | 56 | 19,072.50 | $21,361,200 | $2,768,006 | 771.7% | $-45,080 |
| 2024-05-22 17:00:00 | Buy | 56 | 18,804.00 | $21,060,480 | $2,775,375 | 758.8% | $-36,120 |

**Peak NQ notional: $22,488,700 on 2024-12-09 (701.1% of $3,207,685 portfolio)**

### The 3 Killer NQ Trades

#### 2024-10-14 — NQ Buy, $-265,200
- **Entry:** 20,626.25  **Exit:** 20,350.00  **Move:** -276.25 pts (-1.34%)
- **Quantity:** 48 contracts
- **Notional:** $19,801,200
- **Portfolio at entry:** $2,814,930
- **NQ notional as % of portfolio:** 703.4%
- **P&L:** $-265,200
- **Point loss:** -276.25 pts x 48 contracts x $20 = $-265,200

#### 2024-06-18 — NQ Buy, $-176,400
- **Entry:** 19,924.25  **Exit:** 19,766.75  **Move:** -157.50 pts (-0.79%)
- **Quantity:** 56 contracts
- **Notional:** $22,315,160
- **Portfolio at entry:** $2,697,328
- **NQ notional as % of portfolio:** 827.3%
- **P&L:** $-176,400
- **Point loss:** -157.50 pts x 56 contracts x $20 = $-176,400

#### 2024-12-09 — NQ Buy, $-121,160
- **Entry:** 21,623.75  **Exit:** 21,507.25  **Move:** -116.50 pts (-0.54%)
- **Quantity:** 52 contracts
- **Notional:** $22,488,700
- **Portfolio at entry:** $3,207,685
- **NQ notional as % of portfolio:** 701.1%
- **P&L:** $-121,160
- **Point loss:** -116.50 pts x 52 contracts x $20 = $-121,160

**Root cause:** NQ sizing is proportional to portfolio equity. As the portfolio grew to $2.7-3.2M, the position sizer allocated 48-56 NQ contracts. At $20/point, a 276-point adverse move on 48 contracts = $-265,200. The position runs at 700-830% of portfolio NAV in gross notional — making single-trade drawdowns of 4-8% of equity routine and catastrophic tail events inevitable without a notional cap.

---

## BEAR Regime Analysis (2018-2019)

Regime labels from `results/regimes_ES.csv` (ES-based regime classifier). BEAR label = regime state classified as BEAR at trade entry hour.

### 2018-2019 Trade P&L by Direction

| Year | Buy Trades | Buy P&L | Sell Trades | Sell P&L | Total P&L |
|------|-----------|---------|------------|---------|-----------|
| 2018 | 46 | $-123,198 | 22 | $-13,568 | $-136,765 |
| 2019 | 83 | $-37,041 | 23 | $9,128 | $-27,914 |

### BEAR-Regime Long Gate Analysis

If the `rhb > 5` BEAR gate had been applied (blocking all Buy entries when regime = BEAR):

| Year | Blocked Longs | P&L of Blocked Trades | Recovery (savings) |
|------|--------------|----------------------|-------------------|
| 2018 | 2 | $7,060 | $-7,060 |
| 2019 | 21 | $-75,764 | $+75,764 |
| **Total** | **23** | **$-68,704** | **$+68,704** |

**2018 note:** The 2 blocked BEAR-regime longs in 2018 were winners (+$7,060). Blocking them would cost $7k. The 2018 bear losses came from SIDEWAYS-regime longs and Sell trades during the Q4 2018 crash, not from BEAR-labeled periods.

**Net regime-gate recovery: $68,704** (dominated by 21 blocked BEAR-regime longs in 2019 saving $75,764, offset by $7,060 cost in 2018).

**Full-backtest BEAR-regime longs:** 35 trades across all years, cumulative P&L = $-147,564. Blocking all would recover $147,564.

---

## Top 10 Winning Wells (v3)

| Start | End | Dur | Instruments | Dir | P&L | Trades |
|-------|-----|-----|-------------|-----|-----|--------|
| 2023-11-13 | 2023-11-14 | 20h | ES+NQ+YM | Buy | $350,190 | 3 |
| 2024-11-22 | 2024-11-25 | 67h | ES+YM | Buy | $328,805 | 2 |
| 2023-12-12 | 2023-12-13 | 30h | NQ+YM | Buy | $245,910 | 3 |
| 2020-11-06 | 2020-11-09 | 67h | YM | Buy | $192,610 | 1 |
| 2024-02-08 | 2024-02-12 | 97h | ES+NQ | Buy | $186,135 | 5 |
| 2023-11-17 | 2023-11-20 | 73h | ES+NQ | Buy | $177,808 | 4 |
| 2022-12-20 | 2022-12-22 | 43h | NQ | Sell | $154,795 | 2 |
| 2024-07-08 | 2024-07-10 | 46h | ES+NQ+YM | Buy | $144,538 | 5 |
| 2023-07-12 | 2023-07-18 | 139h | ES+YM | Buy | $136,195 | 6 |
| 2023-11-21 | 2023-11-24 | 69h | ES+YM | Buy | $128,075 | 4 |

---

## Top 10 Losing Wells (v3)

| Start | End | Dur | Instruments | Dir | P&L | Trades |
|-------|-----|-----|-------------|-----|-----|--------|
| 2018-05-14 | 2018-05-15 | 18h | ES+YM | Buy | $-188,385 | 2 |
| 2024-06-18 | 2024-06-20 | 51h | ES+NQ | Buy | $-169,300 | 4 |
| 2024-12-09 | 2024-12-09 | 1h | NQ | Buy | $-121,160 | 1 |
| 2024-12-11 | 2024-12-12 | 19h | NQ | Buy | $-120,360 | 1 |
| 2024-11-11 | 2024-11-12 | 19h | ES+YM | Buy | $-115,990 | 2 |
| 2024-10-11 | 2024-10-15 | 92h | ES+NQ | Buy | $-114,975 | 3 |
| 2024-09-06 | 2024-09-09 | 68h | ES | Sell | $-106,250 | 2 |
| 2019-02-06 | 2019-02-07 | 19h | ES+NQ | Buy | $-102,488 | 2 |
| 2024-04-09 | 2024-04-09 | 1h | YM | Sell | $-98,020 | 1 |
| 2024-10-17 | 2024-10-18 | 18h | YM | Buy | $-97,920 | 1 |

---

## v3 vs v1 Well Quality Comparison

| Dimension | v1 | v3 | Interpretation |
|-----------|----|----|----------------|
| Gross P&L | $2,901,988 | $2,101,549 | v3 generates $800k less despite 108 more trades |
| Multi-inst wells | 47 | 95 | v3 fires multi-inst wells 2x as often |
| Multi-inst win rate | 74.5% | 55.8% | Quality of multi-inst signals degraded severely |
| Avg multi-inst P&L | $50,272 | $17,518 | Each multi-inst well earns 65% less on average |
| Avg single-inst P&L | $2,496 | $2,457 | Single-inst wells essentially unchanged |
| Max drawdown | 29.9% | 37.5% | v3 has 7.7pp more drawdown |
| Sharpe | 4.289 | 4.760 | v3 Sharpe slightly higher (+0.47) despite lower P&L |
| Well avg win P&L | $70,463 | $45,447 | v3 wins are much smaller |
| Well avg loss P&L | $-59,785 | $-33,220 | v3 losses also smaller (more symmetric but lower edge) |

### Diagnosis

The v3 configuration has **over-fired multi-instrument coordination**: the simultaneous ES+NQ+YM signal logic triggers more often, but when wrong, three losing legs compound. In v1, 74.5% of multi-inst wells were winners, generating $2.36M from just 47 events. In v3, 55.8% win rate across 95 events generates only $1.66M. The correlated-entry threshold is too permissive in v3.

The NQ single-trade notional exposure (700-830% of portfolio) is the most dangerous structural flaw. A position sizer without a notional-cap-as-fraction-of-NAV will continue producing catastrophic single-day drawdowns as NAV grows.

---

*Report generated by trade forensics pipeline. Source: Measured Red Anguilline (v3 QC backtest).*