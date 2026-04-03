# LARSA Strategy Changelog

## v4 — 2026-04-03  [PENDING QC]
### Changes
- **NQ notional cap $400k**: Hard ceiling on NQ dollar exposure. At $3M portfolio, reduces NQ from 65% to 13% of equity.
- **BEAR long gate rhb>5**: Blocks new long entries after 5+ consecutive BEAR-regime bars.
### Expected Impact
- NQ cap: +~$507k recovery (Oct-14 -$265k, Jun-18 -$177k, Dec-09 -$121k)
- BEAR gate: +~$200k recovery (2018: -$153k, 2019: -$47k)
- Target: ~$2.7M net P&L (~285% return)

## v3 — 2026-04-03  [QC RESULT: 200.2%]
### Changes
- **CONV_SIZE**: Solo BH capped at 0.40, convergence (2+ BH) gets full 0.65
- **pos_floor v3**: Trigger ctl>=5 (was 3), retention 70% (was 90%), 5%/bar decay
### Result
- Net: $2,001,585 (+200.2%)  Max DD: 40.2%  Sharpe: 0.516  Trades: 485
- 2022: +$250k (+79k vs v1) ✓  2023: +$863k (+173k vs v1) ✓
- 2024: +$112k (-$1,004k vs v1) ✗  ← STOP_LOSS destroyed convergence events
- Failure: NQ Oct-14 -$265k, Jun-18 -$177k, Dec-09 -$121k (notional blowup)

## v2 — 2026-04-03  [QC RESULT: 175.1%]
### Changes
- BEAR_FAST: tl_req=2 in BEAR regime (was 3)
- STOP_LOSS: cut if well P&L < -3% portfolio
### Result
- Net: ~$1,751,000 (+175.1%)  Max DD: 48.1%  Trades: 513
- 2018: -$216k (BEAR_FAST whipsaw) ✗  2019: -$98k ✗  2024: +$112k ✗
- Root cause: STOP_LOSS fires during convergence dips, cutting profitable events

## v1 — 2026-04-01  [QC RESULT: 274% — BASELINE]
### Result
- Net: $2,747,046 (+274%)  Max DD: 29.9%  Sharpe: 4.289  Trades: 377
- 2020: +$629k  2023: +$690k  2024: +$1,116k
- Edge: 47 multi-instrument wells = 81.4% of gross P&L at 74.5% WR
