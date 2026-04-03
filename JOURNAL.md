# SRFM Lab Journal

---

## 2026-04-03  v4 QC result — Logical Brown Caribou: 204.4% return, Sharpe 0.537, DD 40.0%, 486 trades. NQ notional cap working (largest NQ loss -$88k vs v3's -$265k). BEAR gate in place but 2018-2019 still -$169k (longs not blocked, only regime-weighted). v4 vs v3: +4pp return, same DD, slightly better Sharpe.  [RESULT]
**Finding**: (edit in JOURNAL.md)
**Next**: (edit in JOURNAL.md)


## 2026-04-03  v3 -> QC  [SUBMITTED]
**Result**: 200.2% return, Sharpe 0.516, DD 38.4%, 485 trades (-73pp vs v1)
**Changes**: CONV_SIZE (solo 0.40) + pos_floor ctl>=5/70%/decay
**Finding**: NQ $20/pt makes 0.65 leverage = $2M notional at peak. Killer: Oct-14 -$265k.
**Next**: NQ notional cap $400k + BEAR long gate rhb>5

## 2026-04-03  v2 -> QC  [SUBMITTED]
**Result**: 175.1% return, DD 48.1%, 513 trades
**Changes**: BEAR_FAST (tl_req=2 in BEAR) + STOP_LOSS (-3% well cut)
**Finding**: BEAR_FAST adds whipsaw entries. STOP_LOSS fires during convergence events, cuts winners.
**Next**: Revert both. Keep ATR_SCALE + CONV_MASS only.

## 2026-04-03  v1 -> QC  [SUBMITTED]
**Result**: 290.2% return, Sharpe 4.29, DD 29.9%, 377 trades
**Changes**: Baseline LARSA -- BH physics, regime detection, 3-instrument (ES/NQ/YM)
**Finding**: Win rate 54.9%. NQ profitable but high variance ($20/pt multiplier). ES dominant contributor ($1.6M PnL).
**Next**: Add CONV_SIZE solo cap + pos_floor to reduce NQ variance.

---
*Last updated: 2026-04-03*
