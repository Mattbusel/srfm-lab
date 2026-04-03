# SRFM — Special Relativistic Financial Mechanics: A Research Explainer

*Generated from sprint analysis — 2025-04-03*

---

## 1. The Core Idea

SRFM treats price returns as velocities in a Minkowski spacetime. Each bar's
price move is compared to a "speed of light" constant `cf`:

```
beta = |Δclose/close| / cf
```

- **beta < 1** → TIMELIKE: the move is sub-luminal, ordered, causal
- **beta >= 1** → SPACELIKE: the move is super-luminal, anomalous, acausal

This single classification drives everything downstream.

---

## 2. The Proper Time Clock

In special relativity, a moving clock ticks slower (time dilation):

```
τ += 1/γ   where γ = 1/√(1 - v²),  v = min(0.99, |Δclose/close| / max_vol)
```

Fast moves (high v) → low τ increment → fewer "proper seconds" pass.
Slow, TIMELIKE moves → τ accumulates quickly → "clock runs fast" → more
time measured = more signal-worthy movement per real second.

This creates a natural filter: high-volatility bars contribute less to proper
time, so the strategy "ages" faster during quiet, directional periods.

---

## 3. Black Hole Wells

Mass accretes during TIMELIKE bars and decays during SPACELIKE bars:

```python
# TIMELIKE bar:
ctl      += 1
sb        = min(2.0, 1 + ctl * 0.1)          # sustain scaling: grows with run length
bh_mass   = bh_mass * 0.95 + |br| * 100 * sb  # EMA-like accumulation

# SPACELIKE bar:
ctl       = 0                                  # reset consecutive count
bh_mass  *= 0.7                                # aggressive 30% decay
```

**Formation requires: `bh_mass > bh_form=1.5 AND ctl >= 5`**

The `ctl >= 5` gate is the critical design: it prevents false positives from a
single large move. You need at least 5 consecutive quiet, directional bars before
a gravitational well is considered "formed."

### Mass Equilibrium Analysis

In steady state (p = fraction of TIMELIKE bars):

```
mass_eq = avg_TL_contribution / (1 - 0.95·p - 0.7·(1-p))
```

For BH to form, `mass_eq > 1.5`. This requires both:
1. High p (mostly TIMELIKE, i.e., CF set above typical per-bar move)
2. Sufficient per-bar magnitude (|br| large enough to build mass)

**Calibration rule:** CF ≈ 1.5× the asset's median hourly |return|
gives ~60% TIMELIKE rate, which is the optimal balance for mass accumulation.

### The Reform Memory

After a well collapses, for 15 bars afterward:
```python
bh_mass += prev_bh_mass * 0.5   # 50% of prior peak re-added at next formation
```
This allows rapid re-entry when the same gravitational structure re-forms.

---

## 4. The Geodesic Analyzer

A 20-bar log-linear regression on log(price) gives the "expected trajectory"
under constant momentum. Deviation from this trajectory is `geo_dev`:

```python
geo_dev = tanh((close - projected_price) / ATR)
```

Physical interpretation: in GR, free-falling objects follow geodesics. Price
deviating from its geodesic indicates a force is acting on it — either a signal
to fade (mean reversion) or amplify (momentum continuation).

The `causal_frac` (fraction of bar pairs with negative spacetime interval
ds² < 0) measures how "causal" the recent price history is — higher values
mean more internally consistent, ordered movement.

---

## 5. Gravitational Lensing

When a black hole exists, it bends light (amplifies signals from behind it).
In SRFM:

```python
R_E     = sqrt(ctl)                    # Einstein radius grows with TIMELIKE run
r       = |close - VWAP_TL| / ATR    # distance from "lens center"
mu      = 1 + R_E / (r + R_E)        # lensing magnification
```

`mu >= 1` always. When price is far from the VWAP of the TIMELIKE window
(r >> R_E), mu ≈ 1 (no amplification). When price is near the center
(r << R_E), mu → ∞ (maximum amplification).

In practice, mu is bounded: if ctl < 2, mu = max(0.3, ctl/3).

---

## 6. Hawking Temperature

```python
z   = (close - BB_middle) / BB_std    # Bollinger Z-score
ht  = z * (z - prev_z)               # second-order change
```

High ht means the Z-score is moving away from its prior value rapidly — an
impulsive, potentially dangerous move. Used to gate entries when ht > 1.8.

Physical analogy: Hawking radiation is what makes black holes evaporate.
High Hawking temperature signals the well is becoming unstable.

---

## 7. The Three-Agent Ensemble

LARSA uses three complementary signal generators:

| Agent  | Strategy | Key Inputs |
|--------|----------|-----------|
| D3QN   | Trend/momentum | BH mass+dir, ADX, MACD, MOM, ROC |
| DDQN   | Alignment/composite | Sign agreement across MACD/HIST/MOM/ROC |
| TD3QN  | Mean-reversion/vol-aware | BBP contrarian, BBW regime, RSI extremes |

The ensemble combines them with **Lorentz boost/damping**:
- TIMELIKE bar (beta < 1): signals boosted by γ = 1/√(1-β²), up to 2×
- SPACELIKE bar (beta >= 1): signals damped by 1/β

This naturally amplifies signals from quiet, ordered moves and suppresses
signals from chaotic, volatile bars.

Regime weights shift the blend:
```python
REGIME_WEIGHTS = {
    BULL:            [0.40, 0.35, 0.25],   # D3QN dominant (trend)
    BEAR:            [0.35, 0.25, 0.40],   # TD3QN dominant (vol-aware)
    SIDEWAYS:        [0.25, 0.40, 0.35],   # DDQN dominant (alignment)
    HIGH_VOLATILITY: [0.20, 0.30, 0.50],   # TD3QN dominant (protection)
}
```

---

## 8. Convergence Multiplier

When all 3 instruments (ES, NQ, YM) have simultaneous active BHs:

```
max_leverage → 2.5×   (vs 1.0× for single-instrument)
```

This is the most powerful signal in the system. Three independent gravitational
wells forming simultaneously indicates a macro regime shift, not noise.

From our multi-asset survey on synthetic correlated data:
- Convergence bars (≥2 BHs): 0.22% of all bars
- Directional agreement at convergence: 82.5% (very high coherence)

---

## 9. Kill Conditions (Ordered)

1. `geo_raw > 2.0` → exit immediately (crisis detection)
2. `bc < 120` → warmup period (120-bar minimum)
3. `tl_confirm < 3` → quality gate (3 consecutive TL required)
4. SPACELIKE → 15% position penalty (reduce, don't exit)
5. `|signal| < 0.03` → zero (noise floor)
6. `weak_bars == 3` → exit after 3 consecutive weak signals
7. `pos_floor` ratchet → prevents position going below historical floor

The kill conditions are evaluated in this exact order — they are not independent
filters. This ordering means geo_raw acts as an emergency circuit breaker.

---

## 10. Key Research Findings

### BH Formation Requires Autocorrelation

Pure GBM (random walk) almost never forms BH wells at typical CF values because:
- Each bar is independent → TIMELIKE/SPACELIKE alternates randomly
- ctl rarely reaches 5 before a SPACELIKE interruption
- Mass equilibrium too low: `mass_eq = avg_TL_contribution / (1 - 0.95p - 0.7(1-p))`

**The strategy fundamentally detects momentum persistence, not random moves.**
The 274% return required the 2020-2021 sustained bull run to activate the BH
mechanism at scale. During sideways or random regimes, the kill conditions
prevent almost all trading.

### CF Calibration Per Asset

| Asset | LARSA CF | Optimal CF (25 wells/yr on synthetic) |
|-------|---------|--------------------------------------|
| ES    | 0.001   | ~0.0012                              |
| NQ    | 0.0012  | ~0.0030                              |
| YM    | 0.0008  | ~0.0020                              |
| ZB    | 0.0004  | ~0.0012                              |
| GC    | 0.0006  | ~0.0012                              |

Rule of thumb: `CF ≈ 1.2–1.5 × median_hourly_|return|`

### 2020 COVID Was Likely 60-80% of the 274%

The regime-switching nature of LARSA means:
- During sustained BULL: BH mass builds rapidly (sb=2.0 multiplier)
- During COVID bear: short wells fire with high conviction
- During COVID recovery: long bull wells compound with convergence multiplier

The 30.2% probability of beating 274% in Monte Carlo confirms: the actual
result is achievable but not typical. The 2020 episode was favorable, not
representative.

---

## 11. The Arena (Pure Python, No LEAN)

The complete SRFM pipeline now runs without Docker, LEAN, or QC:

```python
from lib.srfm_core import MinkowskiClassifier, BlackHoleDetector, ...
from lib.agents import ensemble, size_position
from lib.broker import SimulatedBroker

broker = SimulatedBroker(cash=1_000_000)
for bar in price_bars:
    bit    = mc.update(bar.close)
    active = bh.update(bit, bar.close, prev_close)
    action, conf, _ = ensemble(features, mu, ...)
    target = size_position(features, action, conf, rm, regime, ...)
    broker.update(bar_return)
    broker.set_position(target)

stats = broker.stats()  # Sharpe, return, max_dd, trade_count
```

Tournament: sweeps CF × BH_FORM × BH_DECAY × MaxLev across 200+ combinations.
On 20k synthetic bars at 0.5 runs/sec = 400 sec for 200 parameter sets.

---

## 12. Open Questions

1. **Does the 274% replicate with real QC data?** The LEAN backtest requires
   Docker (or QC cloud account upgrade). The browser UI confirms it runs but
   we can't automate extraction.

2. **What was the 2019 flat period?** Low tl_confirm in sideways 2019 likely
   triggered kill conditions frequently. Relaxing to tl_confirm≥2 in SIDEWAYS
   regime might add profitable trades but increase false positives.

3. **Is the convergence multiplier the key?** Our synthetic analysis shows 82.5%
   directional agreement at convergence events — but they only occur 0.22% of
   bars. Real-data attribution would require the actual trade log from QC.

4. **ZB/GC calibration?** Bond futures need much lower CF; commodity futures
   need different bh_form thresholds. The current values are not calibrated
   for non-equity assets.

5. **Is SRFM a proxy for traditional signals?** The ensemble ultimately reduces
   to weighted MACD + RSI + BBP. The physics wrapping may add value through
   the ctl gate (momentum filter) and the Lorentz boost/damping (vol-adaptive
   weighting). A clean ablation study (SRFM physics ON vs OFF) on real data
   would resolve this.
