# SRFM Strategy Hypotheses

Ranked by expected impact. Updated as experiments run.

| # | Hypothesis | Expected Impact | Status | Verdict |
|---|-----------|----------------|--------|---------|
| H1 | Multi-instrument convergence creates edge | +81.4% of P&L | ✓ TESTED | CONFIRMED — 74.5% WR vs 54.4% baseline |
| H2 | BEAR_FAST (tl_req=2 in BEAR) helps in bear markets | +$50k | ✗ TESTED | REJECTED — whipsaws in 2018-2019, -$310k |
| H3 | STOP_LOSS (-3% well cut) protects capital | -$50k DD | ✗ TESTED | REJECTED — fires on convergence dips, cuts winners |
| H4 | ATR_SCALE reduces vol-spike losses | -$20k DD | ~ NEUTRAL | Minimal effect on NDX data scale |
| H5 | CONV_SIZE (0.40 solo / 0.65 multi) improves Sharpe | -1.76pp DD | ~ PARTIAL | DD improvement, but NQ blowup still hurts |
| H6 | pos_floor ctl>=5 / 70% / decay improves synth | +0.2 synth Sh | ~ NEUTRAL | Neutral on real data, minor synthetic improvement |
| H7 | NQ notional cap $400k prevents blowup | +$507k | ⏳ PENDING | v4 — expected to recover Oct/Jun/Dec 2024 losses |
| H8 | BEAR long gate (rhb>5) saves 2018-2019 | +$200k | ⏳ PENDING | v4 — blocks sustained BEAR longs |
| H9 | SIDEWAYS as regime transition signal | unknown | 💡 IDEA | Regime graph: SIDEWAYS→BULL 8.0%, SIDEWAYS→BEAR 2.4% |
| H10 | Shorter pos_floor in BEAR avoids holding losses | +$50k? | 💡 IDEA | Asymmetric floor: 0% in BEAR, 70% in BULL only |
