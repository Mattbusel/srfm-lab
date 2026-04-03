# SRFM Lab — Results Summary

Last updated: 2025-04-03

---

## Phase 1 Status: COMPLETE

All deliverables complete. 129/129 tests passing.

---

## Key Findings

### 1. LARSA Physics (from code analysis)

| Finding | Detail |
|---------|--------|
| BH formation gate | `bh_mass > 1.5 AND ctl >= 5` — the ctl gate prevents false positives |
| SPACELIKE decay | `bh_mass *= 0.7` — one volatile bar wipes ~30% of accumulated mass |
| Bear well amplifier | `sb = min(2.0, 1+ctl*0.1)` — 10-bar TIMELIKE run doubles mass contribution |
| Reform memory | 15-bar window adds 50% of prior peak mass — rapid re-entry detection |
| Convergence multiplier | max_lev scales to 2.5x when all 3 instruments have active BHs |
| Hawking temperature | `ht = z*(z-prev_z)` — second-order BB Z-score, gates at ht > 1.8 |

---

### 2. Monte Carlo (50,000 simulations on synthetic LARSA-like trade distribution)

```
Win rate: 58%  |  Avg win: +1.8%  |  Avg loss: -0.9%
------------------------------------------------------
Median terminal return :  +236.4%
Mean terminal return   :  +243.9%
5th  percentile        :  +142.1%
95th percentile        :  +370.1%
P(ruin < 50%)          :   0.0%
P(return > 100%)       :  99.6%
P(return > 200%)       :  71.9%
P(return > 274%)       :  30.2%
Avg max drawdown       :   6.2%
P95 max drawdown       :   9.4%
```

**Interpretation:** The 274% actual result is at the ~70th percentile of achievable
outcomes with this trade distribution. Not an outlier. Zero ruin across 50k sims.

---

### 3. Well Detection (synthetic ES, 52,560 hourly bars)

```
Total wells   : 68
Long / Short  : 40 / 28
Avg duration  : 6.8 bars
CF used       : 0.001 (LARSA default)
```

Key: Only 2.7 wells/year at current CF on synthetic GBM data. Real ES data
(with autocorrelated returns) should generate significantly more wells.

---

### 4. CF Calibration — Critical Finding

**The BH formation mechanism requires autocorrelated price structure.**

Pure GBM breaks TIMELIKE runs too frequently for `ctl` to reach the 5-bar threshold.
LARSA's CF values were calibrated on real market data, not synthetic.

| Asset | Current CF | Wells/yr (synthetic) | Calibrated CF (25 wells/yr) |
|-------|-----------|---------------------|----------------------------|
| ES    | 0.001     | 2.7                 | 0.0012                     |
| NQ    | 0.0012    | 4.8                 | 0.0030                     |
| YM    | 0.0008    | 0.0                 | 0.0020                     |
| ZB    | 0.0004    | 0.0                 | 0.0012                     |
| GC    | 0.0006    | 0.0                 | 0.0012–0.0014              |

**Rule of thumb:** Set CF ≈ median hourly |return| of the asset.
This puts ~50% of bars TIMELIKE, enabling ctl to accumulate to 5.

---

### 5. Multi-Asset Correlation Structure

BH-active state correlation on synthetic ES/NQ/YM (rho_price=0.85):

```
ES-NQ: +0.149
ES-YM:  0.000 (YM has 0 wells at current CF — see above)
NQ-YM:  0.000
```

When wells do form on real data, the ES-NQ-YM trio should show strong BH
correlation since they are driven by the same equity market momentum.

Convergence bars (>=2 simultaneous BHs): 0.22% of total bars.
Directional agreement at convergence: 82.5% same direction — high signal coherence.

---

### 6. Regime Distribution (synthetic ES, 52,560 bars)

| Regime         | Bars   | % of Time |
|----------------|--------|-----------|
| BULL           | 33,543 | 63.8%     |
| SIDEWAYS       | 10,684 | 20.3%     |
| BEAR           |  8,140 | 15.5%     |
| HIGH_VOLATILITY|    193 |  0.4%     |

---

## Backtest Status

| Run | Method | Status | Result |
|-----|--------|--------|--------|
| LARSA v1 (original) | QC Cloud UI | DONE | 274% (2018–2024) |
| LARSA v1 (local) | LEAN CLI + Docker | BLOCKED: Docker not installed | — |
| LARSA v1 (cloud API) | lean cloud backtest | BLOCKED: requires paid QC plan | — |

**To run the local backtest:** Install Docker Desktop, then:
```bash
cd /c/Users/Matthew/lean-org
lean backtest larsa-v1
```

---

## Experiment Queue

| ID | Experiment | Status | Blocker |
|----|-----------|--------|---------|
| 2A | Parameter sensitivity sweep (CF, BH_FORM, BH_DECAY) | PENDING | LEAN/Docker |
| 2B | Resolution experiment (30min, 15min, daily) | PENDING | LEAN/Docker + data |
| 2C | Multi-asset well survey | DONE (synthetic) | — |
| 2D | CF calibration per asset | DONE (synthetic) | — |
| 2E | Convergence P&L attribution | DONE (synthetic) | — |
| 3A | Walk-forward validation (6 windows) | PENDING | LEAN/Docker |
| 3B | Monte Carlo on real trade data | PENDING | Actual backtest first |
| 3C | Regime-conditional performance | PENDING | LEAN/Docker |

---

## Open Questions

1. **Does LARSA's 274% replicate locally?** Needs Docker for LEAN or QC paid plan.

2. **What drove the 274% specifically?** 2020 COVID crash + recovery likely contributed
   60-100%. Per-year breakdown requires actual trade log.

3. **BH mechanism fundamentally requires momentum.** Real market autocorrelation is
   the signal source. The strategy is detecting regime persistence, not noise.

4. **CF values need real-data calibration.** Synthetic results are directional only.
   True calibration requires downloading hourly ES/NQ/YM data via QC or IB.

5. **The flat 2019-2021 sideways periods.** LARSA likely had few trades during the
   2019 grind-up. Investigate relaxing `tl_confirm` to 2 in SIDEWAYS regime.

---

## Tool Status

| Tool | Status | Notes |
|------|--------|-------|
| `tools/well_detector.py` | Working | Runs on any OHLCV CSV |
| `tools/monte_carlo.py` | Working | Bootstraps from trade returns list |
| `tools/regime_analyzer.py` | Working | Full EMA/ATR/ADX from raw prices |
| `tools/sensitivity.py` | Ready | Needs LEAN/Docker to run backtests |
| `tools/param_sweep.py` | Ready | Needs LEAN/Docker to run backtests |
| `tools/walk_forward.py` | Ready | Needs LEAN/Docker to run backtests |
| `tools/batch_runner.py` | Ready | Needs LEAN/Docker to run backtests |
| `tools/compare.py` | Ready | Needs LEAN result JSONs as input |

---

## File Map

```
lib/
  srfm_core.py     - MinkowskiClassifier, BlackHoleDetector, GeodesicAnalyzer,
                     GravitationalLens, HawkingMonitor (exact LARSA formulas)
  features.py      - 31-element feature vector, named index constants
  agents.py        - D3QN, DDQN, TD3QN ensemble + size_position
  regime.py        - RegimeDetector (exact detect_regime logic)
  risk.py          - PortfolioRiskManager, KillConditions

strategies/
  larsa-v1/        - Production LARSA code (274% verified on QC Cloud)

tests/             - 129 tests, all green
data/              - Synthetic OHLCV CSVs (ES, NQ, YM, ZB, GC)
results/
  larsa-v1/mc/     - Monte Carlo JSON
  survey/          - Multi-asset survey CSVs
  regimes_ES.csv   - Regime timeline
  experiments.md   - Detailed experiment log
```

---

## Phase 5: Real Trade Log Analysis (Calm Orange Mule)

**Source:** QC LARSA backtest trade log (2018-2024, ES/NQ/YM hourly)
**File:** `tools/trade_forensics.py`, `research/trade_analysis.md`, `research/notebooks/08_trade_forensics.py`

### Headline Numbers
| Metric | Value |
|--------|-------|
| Total Trades | 377 |
| Total Wells (Events) | 263 |
| Gross P&L | $2,901,988 |
| Gross Return | 290.2% |
| Win Rate (trades) | 54.9% |
| Max Drawdown | 29.9% |
| Sharpe (annualized) | 4.289 |
| P&L Ratio W/L | 1.34x |

### Annual Attribution (The Real Story)
| Year | Trades | Win% | Gross P&L | % of Total |
|------|--------|------|-----------|-----------|
| 2018 | 57 | 53% | $93,848 | 3.2% |
| 2019 | 70 | 49% | $93,202 | 3.2% |
| 2020 | 32 | 59% | $628,838 | 21.7% |
| 2021 | 70 | 50% | $108,965 | 3.8% |
| 2022 | 10 | 60% | $171,055 | 5.9% |
| 2023 | 56 | 54% | $690,225 | 23.8% |
| **2024** | **82** | **65%** | **$1,115,855** | **38.5%** |

**Key revision:** Prior hypothesis that "2020 drove 60-80% of returns" is WRONG.
- 2020+2023+2024 = 83.9% of all P&L
- 2024 alone = 111.6% of initial capital ($1M)
- Strategy compounds: larger capital base → larger absolute $ per % move

### The Convergence Finding (CONFIRMED)
| Metric | Multi-Instrument | Single-Instrument |
|--------|-----------------|------------------|
| # Wells | 47 (17.9%) | 216 (82.1%) |
| Gross P&L | $2,362,805 (81.4%) | $539,182 (18.6%) |
| Win Rate | **74.5%** | ~49% |

**Multi-instrument convergence wells are 17.9% of events but 81.4% of gross P&L.**
This definitively answers Open Question #3: the convergence multiplier IS the key.

### Instrument Attribution
- ES: 55.3% of P&L (183 trades, workhorse/signal)
- YM: 23.4% of P&L (88 trades, macro direction filter)
- NQ: 21.3% of P&L (106 trades, momentum alpha)

### Direction Bias
- Long: 81.6% of P&L ($2.37M, 287 trades, 55% WR)
- Short: 18.4% of P&L ($534k, 90 trades, 53% WR)
- Strategy is primarily a bull-trend follower with short capability

### Top Single Wells
| Date | Duration | Instruments | Gross P&L | Event |
|------|----------|-------------|-----------|-------|
| 2023-12-13 | 30h | NQ+YM | $453,285 | Dec Fed rally |
| 2020-11-06 | 67h | YM | $436,205 | US Election + Vaccine |
| 2023-11-13 | 20h | NQ+YM | $328,200 | Oct CPI relief rally |
| 2024-11-22 | 66h | ES | $286,875 | Post-election bull run |
| 2022-07-27 | 2h | NQ | $261,960 | Fed pivot expectations |
