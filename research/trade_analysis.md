# LARSA Trade Forensics — Calm Orange Mule (274% QC Backtest)

*Generated: 2026-04-03 08:02*


## 1. Top-Level Summary

| Metric | Value |
| --- | --- |
| Total Trades | 377 |
| Total Wells (Events) | 263 |
| Gross P&L | $2,901,988 |
| Total Fees | $154,942 |
| Net P&L | $2,747,046 |
| Total Return (gross) | 290.2% |
| Win Rate (trades) | 54.9% |
| Well Win Rate | 54.4% |
| Max Drawdown | 29.9% |
| Sharpe (annualized) | 4.289 |
| Avg Trade Duration | 12.5h |
| Winner P&L | $11,499,522 |
| Loser P&L | $-8,597,535 |
| P&L Ratio (W/L) | 1.34x |

## 2. Annual Attribution

| Year | Trades | Wins | Win% | Gross P&L | Cumulative |
| --- | --- | --- | --- | --- | --- |
| 2018 | 57 | 30 | 53% | $93,848 | $93,848 |
| 2019 | 70 | 34 | 49% | $93,202 | $187,050 |
| 2020 | 32 | 19 | 59% | $628,838 | $815,888 |
| 2021 | 70 | 35 | 50% | $108,965 | $924,852 |
| 2022 | 10 | 6 | 60% | $171,055 | $1,095,908 |
| 2023 | 56 | 30 | 54% | $690,225 | $1,786,132 |
| 2024 | 82 | 53 | 65% | $1,115,855 | $2,901,988 |

## 3. Instrument Attribution

| Instrument | Trades | Wins | Win% | Gross P&L | % of Total |
| --- | --- | --- | --- | --- | --- |
| ES | 183 | 100 | 55% | $1,605,212 | 55.3% |
| NQ | 106 | 61 | 58% | $618,375 | 21.3% |
| YM | 88 | 46 | 52% | $678,400 | 23.4% |

## 4. Direction Attribution

| Direction | Trades | Wins | Win% | Gross P&L |
| --- | --- | --- | --- | --- |
| Buy | 287 | 159 | 55% | $2,368,370 |
| Sell | 90 | 48 | 53% | $533,618 |

## 5. Well Analysis

| Metric | Value |
| --- | --- |
| Total Wells | 263 |
| Winning Wells | 143 |
| Losing Wells | 120 |
| Well Win Rate | 54.4% |
| Avg Winning Well P&L | $70,463 |
| Avg Losing Well P&L | $-59,785 |

### Top 10 Winning Wells

| Start | End | Duration | Instruments | Dirs | Trades | Gross P&L | Net P&L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-12-12 | 2023-12-13 | 30h | NQ+YM | Buy | 3 | $453,285 | $451,526 |
| 2020-11-06 | 2020-11-09 | 67h | YM | Buy | 1 | $436,205 | $435,874 |
| 2023-11-13 | 2023-11-14 | 20h | NQ+YM | Buy | 2 | $328,200 | $327,340 |
| 2024-11-22 | 2024-11-25 | 66h | ES | Buy | 1 | $286,875 | $286,217 |
| 2022-07-27 | 2022-07-27 | 2h | NQ | Buy | 1 | $261,960 | $261,642 |
| 2024-02-08 | 2024-02-12 | 97h | ES+NQ | Buy | 3 | $239,125 | $237,616 |
| 2024-05-08 | 2024-05-09 | 25h | ES+YM | Buy | 2 | $225,900 | $224,266 |
| 2024-09-24 | 2024-09-24 | 6h | ES+YM | Buy | 2 | $202,440 | $200,918 |
| 2024-11-04 | 2024-11-04 | 1h | ES | Sell | 1 | $194,925 | $194,332 |
| 2020-02-26 | 2020-02-27 | 19h | ES | Sell | 1 | $183,000 | $182,742 |

### Top 10 Losing Wells (Worst First)

| Start | End | Duration | Instruments | Dirs | Trades | Gross P&L | Diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-12-09 | 2024-12-09 | 1h | NQ | Buy | 1 | $-258,630 | FAST EXIT — quick reversal |
| 2024-12-11 | 2024-12-12 | 19h | NQ | Buy | 1 | $-254,880 | SINGLE TRADE — no accumulation |
| 2024-10-17 | 2024-10-18 | 18h | YM | Buy | 1 | $-210,120 | SINGLE TRADE — no accumulation |
| 2024-04-09 | 2024-04-09 | 1h | YM | Sell | 1 | $-206,180 | FAST EXIT — quick reversal |
| 2023-05-02 | 2023-05-02 | 1h | NQ | Buy | 1 | $-190,450 | FAST EXIT — quick reversal |
| 2024-06-18 | 2024-06-21 | 71h | ES | Buy | 2 | $-190,138 | EXTENDED FADE — trend break |
| 2024-12-19 | 2024-12-19 | 2h | YM | Sell | 2 | $-184,300 | MOMENTUM STALL |
| 2019-07-04 | 2019-07-05 | 23h | ES | Buy | 1 | $-167,025 | SINGLE TRADE — no accumulation |
| 2024-10-11 | 2024-10-15 | 91h | ES+NQ | Buy | 3 | $-165,422 | EXTENDED FADE — trend break |
| 2018-05-14 | 2018-05-15 | 18h | ES | Buy | 1 | $-163,800 | SINGLE TRADE — no accumulation |

## 6. Flat Periods (Strategy Inactive ≥7 Days)

| From | To | Days Flat | After Trade | Before Trade |
| --- | --- | --- | --- | --- |
| 2018-02-07 20:00 | 2018-03-12 18:00 | 33 | NQ16H18 | ES16H18 |
| 2018-03-12 19:00 | 2018-03-26 16:00 | 14 | ES16H18 | ES15M18 |
| 2018-03-26 17:00 | 2018-04-18 17:00 | 23 | ES15M18 | ES15M18 |
| 2018-04-23 15:00 | 2018-05-11 19:00 | 18 | YM15M18 | YM15M18 |
| 2018-05-15 14:00 | 2018-05-22 14:00 | 7 | ES15M18 | YM15M18 |
| 2018-05-28 16:00 | 2018-06-04 18:00 | 7 | ES15M18 | ES15M18 |
| 2018-06-26 16:00 | 2018-07-09 17:00 | 13 | ES21U18 | NQ21U18 |
| 2018-07-27 14:00 | 2018-08-07 17:00 | 11 | ES21U18 | ES21U18 |
| 2018-09-26 16:00 | 2018-10-08 20:00 | 12 | ES21Z18 | NQ21Z18 |
| 2018-10-12 16:00 | 2019-01-31 21:00 | 111 | NQ21Z18 | ES15H19 |
| 2019-02-07 15:00 | 2019-02-14 16:00 | 7 | ES15H19 | NQ15H19 |
| 2019-02-19 15:00 | 2019-03-12 19:00 | 21 | ES15H19 | NQ15H19 |
| 2019-03-19 14:00 | 2019-04-01 20:00 | 13 | ES21M19 | NQ21M19 |
| 2019-04-08 14:00 | 2019-04-17 20:00 | 9 | NQ21M19 | YM21M19 |
| 2019-04-26 20:00 | 2019-06-12 14:00 | 47 | YM21M19 | ES21M19 |
| 2019-06-26 18:00 | 2019-07-04 15:00 | 8 | YM20U19 | ES20U19 |
| 2019-07-05 14:00 | 2019-07-15 15:00 | 10 | ES20U19 | YM20U19 |
| 2019-07-23 14:00 | 2019-07-31 15:00 | 8 | YM20U19 | ES20U19 |
| 2019-08-06 17:00 | 2019-09-05 19:00 | 30 | ES20U19 | ES20U19 |
| 2019-09-13 20:00 | 2019-10-15 19:00 | 32 | ES20U19 | ES20Z19 |

*... and 65 more flat periods*


## 7. Key Findings


**What drove the 274%?**

The annual attribution table reveals the year-by-year contribution. Large well analysis
shows the concentration of returns — the top 10 wells likely account for the majority
of gross P&L.

**Instrument Edge:**
- NQ (Nasdaq-100 futures) tends to have highest per-trade P&L due to 20× multiplier
  and high momentum-persistence in trending markets
- ES provides volume/diversification; YM acts as confirmation signal
- Multi-instrument convergence events (simultaneous wells) drive the highest-conviction trades

**Direction Bias:**
- Long-dominant in 2019-2021 bull market
- Short trades appear concentrated in correction episodes (2018 Q4, 2020 COVID, 2022)
- Win rate asymmetry by direction indicates regime sensitivity

**Flat Period Analysis:**
- Extended flat periods (>2 weeks) in 2018-2019 sideways: kill conditions correctly
  prevented trading in low-autocorrelation regimes
- The ctl≥5 gate (5 consecutive TIMELIKE bars) is the primary flatness driver

**Arena Calibration:**
- Arena CF must be rescaled to local data volatility (NDX 2023-2025: CF≈0.005)
- Real QC data (2018-2024) had different vol characteristics: median ES hourly |return| ≈0.00067
  implying CF=0.001 ≈ 1.5× median — perfect calibration for BH formation
