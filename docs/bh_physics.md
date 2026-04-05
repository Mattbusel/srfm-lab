# Black Hole Physics Trading Engine

## SRFM — Special Relativistic Financial Mechanics

> "Markets are not random walks. They are causal manifolds punctuated by
> phase transitions. The Black Hole engine detects when price momentum has
> become gravitationally bound — and bets accordingly."

---

## Table of Contents

1. [Theoretical Foundation — Minkowski Spacetime for Price Bars](#1-theoretical-foundation)
2. [MinkowskiClassifier — TIMELIKE vs SPACELIKE](#2-minkowskiclassifier)
3. [BlackHoleDetector — Mass Accumulation & Decay](#3-blackholedetector)
4. [BH Formation Threshold (BH_FORM)](#4-bh-formation-threshold)
5. [Hawking Temperature Monitor](#5-hawking-temperature-monitor)
6. [GravitationalLens — Lensing Amplification μ](#6-gravitationallens)
7. [GeodesicAnalyzer — 20-Bar Regression & Causal Fraction](#7-geodesicanalyzer)
8. [Delta Score — The Allocation Signal](#8-delta-score)
9. [BTC Lead Signal — Cross-Asset Momentum](#9-btc-lead-signal)
10. [Mayer Multiple Dampener](#10-mayer-multiple-dampener)
11. [OU Overlay — Mean Reversion Sleeve](#11-ou-overlay)
12. [GARCH(1,1) Volatility Scaling](#12-garch11-volatility-scaling)
13. [Three-Timeframe Fusion](#13-three-timeframe-fusion)
14. [Parameter Calibration Table](#14-parameter-calibration-table)
15. [Worked Example — Full BTC Signal Trace](#15-worked-example)
16. [Why It Works — Physical Intuition](#16-why-it-works)

---

## 1. Theoretical Foundation

### The Minkowski Metric Applied to Price Bars

In special relativity the spacetime interval between two events is:

```
ds² = c²dt² − dx²
```

where `c` is the speed of light, `dt` is elapsed time, and `dx` is spatial
displacement. An interval is **TIMELIKE** (`ds² > 0`) when the spatial
displacement is smaller than what light can cover in the time elapsed — the
two events can be causally connected. It is **SPACELIKE** (`ds² < 0`) when
displacement exceeds what any causal signal can achieve — the events are
causally disconnected.

SRFM maps this directly onto price bars:

```
ds² = CF²·dt² − (ΔP/P)²
```

| Symbol  | Financial Meaning                             |
|---------|-----------------------------------------------|
| `dt`    | One bar period (normalised to 1)              |
| `ΔP/P`  | Fractional price move: `|close_t − close_{t-1}| / close_{t-1}` |
| `CF`    | "Speed of light" for the instrument — maximum velocity of an *ordered* move |

Dividing both sides by `CF²dt²` and setting `dt = 1`:

```
β  =  (ΔP/P) / CF

TIMELIKE   ↔  β < 1   (sub-luminal: ordered, causal move)
SPACELIKE  ↔  β ≥ 1   (super-luminal: anomalous shock event)
```

This is the `beta` field computed by `MinkowskiClassifier`.

### What CF Means Per-Instrument

`CF` (the "financial speed of light") encodes how much a *normal* ordered
move looks like for a given instrument in a single bar period:

- **Low CF** (e.g. `EURUSD = 0.0005`): tight, well-regulated market — even a
  0.05% move per bar is "super-luminal."
- **High CF** (e.g. `SOL = 0.010`, `NG = 0.020`): volatile instrument — a 2%
  bar is still within the causal cone.

CF is calibrated so that roughly 80–90% of all bars are TIMELIKE in normal
market conditions. A SPACELIKE bar therefore marks a genuine anomaly: gap,
news shock, or liquidity vacuum.

---

## 2. MinkowskiClassifier

```python
class MinkowskiClassifier:
    def __init__(self, cf: float = 0.001, max_vol: float = 0.01): ...
    def update(self, close: float) -> str:  # returns "TIMELIKE" or "SPACELIKE"
```

### Algorithm (per bar)

```
1. price_diff = |close_t − close_{t-1}|
2. beta       = price_diff / close_{t-1} / CF
3. bit        = "TIMELIKE" if beta < 1.0 else "SPACELIKE"
4. if TIMELIKE:  tl_confirm = min(tl_confirm + 1, 3)
   else:         tl_confirm = 0
5. Proper time:
   v     = min(0.99, price_diff/close_{t-1} / max_vol)
   γ     = 1 / √(1 − v²)           # Lorentz factor
   τ    += 1/γ                       # proper time increment
```

`tl_confirm` is capped at 3 in the Minkowski classifier itself; the
`BlackHoleDetector` then allows this to grow further (unbounded) as its own
`ctl` counter.

### Proper Time

Proper time `τ` accumulates slower when price moves fast. A sequence of
violent bars produces less proper time than a slow grind — mirroring
relativistic time dilation. This provides a secondary measure of "how much
ordered time has elapsed" that discounts shock episodes.

---

## 3. BlackHoleDetector

```python
class BlackHoleDetector:
    def __init__(self,
                 bh_form:    float = 1.5,
                 bh_collapse: float = 1.0,
                 bh_decay:   float = 0.95): ...
    def update(self, bit: str, close: float, prev_close: float) -> bool:
```

### Mass Accumulation

On every TIMELIKE bar:

```
ctl      += 1
sb        = min(2.0, 1.0 + ctl × 0.1)        # "slingshot boost"
bh_mass   = bh_mass × BH_DECAY + |br| × 100 × sb
```

where `br = (close − prev_close) / prev_close` is the bar return.

On every SPACELIKE bar:

```
ctl      = 0
bh_mass *= 0.7                                 # hard noise penalty
```

### Why the EMA Asymptotes to 2.0

The slingshot boost `sb` converges to 2.0 as `ctl → 10`:

```
sb = min(2.0, 1.0 + ctl × 0.1)
   = 2.0   when ctl ≥ 10
```

The mass update is an exponentially weighted moving average with input
`|br| × 100 × sb`. In steady-state with constant bar return `r` and
`sb = 2`:

```
bh_mass* = (|r| × 100 × 2) / (1 − BH_DECAY)
         = 200|r| / 0.05          (BH_DECAY = 0.95)
         = 4000|r|
```

For a typical BTC hourly bar of `|r| ≈ 0.0008` (0.08%), the steady-state
mass is around 3.2 — comfortably above the formation threshold. This means
a sufficiently long streak of consistent moves will always form a black hole,
no matter how small each individual bar is.

### Mass Decay (BH_DECAY = 0.95)

Each bar, 5% of accumulated mass bleeds away:

```
bh_mass_new = bh_mass × 0.95 + new_input
```

This is a half-life of approximately `ln(2)/ln(1/0.95) ≈ 13.5 bars`. After
13–14 bars of silence (zero-return environment), mass halves. After 27 bars,
three-quarters is gone. This ensures that stale conviction cannot linger
indefinitely — it must be continuously refreshed by fresh TIMELIKE bars.

### Reform Memory

When a black hole collapses, it records `prev_bh_mass`. If it tries to form
again within the next 14 bars, it gets a 50% boost:

```
if 0 < reform_bars < 15:
    bh_mass += prev_bh_mass × 0.5
```

This models the physical phenomenon of a gravitational well temporarily
dissipating but re-forming rapidly because the underlying mass concentration
has not fully dispersed.

---

## 4. BH Formation Threshold

```
Formation  : bh_mass > BH_FORM  AND ctl ≥ 5
Active     : bh_mass > BH_COLLAPSE AND ctl ≥ 5
Collapse   : bh_mass ≤ BH_COLLAPSE  OR ctl < 5
```

The `ctl ≥ 5` gate ensures the detector requires at least 5 consecutive
TIMELIKE bars before declaring a gravitational well — single spikes cannot
trigger formation.

### Default vs Crypto Thresholds

| Parameter    | Default (engine) | Notes |
|--------------|-----------------|-------|
| `BH_FORM`    | 1.5             | Formation threshold (engine code default) |
| `BH_COLLAPSE`| 1.0             | Maintenance threshold |
| `BH_DECAY`   | 0.95            | Per-bar decay factor |

Note: The live strategy LARSA (production system) used `BH_FORM = 1.92` in
earlier iterations. The engine has standardised on 1.5 after calibration
showing this better balances signal frequency vs noise for multi-asset
portfolios. LARSA's 1.92 is referenced in legacy documentation.

The gap between formation (1.5) and collapse (1.0) creates **hysteresis**: a
BH that has formed will persist through mild disruptions, only collapsing when
mass falls 33% below where it started. This prevents rapid oscillation.

### ASCII Illustration — BH Formation

```
bh_mass
  2.0 |                          ┌──────────────────────
      |                        ╱ │   BLACK HOLE ACTIVE
BH_FORM─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╱  │
  1.5 |                    ╱    │
      |                  ╱      └──────── mass decay begins
BH_COL─ ─ ─ ─ ─ ─ ─ ─ ╱                ── ── collapse
  1.0 |              ╱              ──╱
      |            ╱          ──╱
  0.0 |──────────────────────────────────────────────────► bars
            ctl<5    ctl=5+    BH active      collapse
```

---

## 5. Hawking Temperature Monitor

Hawking radiation from a black hole of mass M has temperature:

```
T_H = ℏc³ / (8πGMk_B)  ∝  1 / (8πM)
```

As the black hole grows more massive (higher mass), it radiates less — it
becomes *colder*. Conversely a small or shrinking BH runs hot.

### SRFM Implementation

```python
class HawkingMonitor:
    def update(self, close: float, bb_middle: float, std: float) -> float:
        z       = (close - bb_middle) / std      # Bollinger Z-score
        ht      = z * (z - prev_z)               # second-order Z² derivative
        return ht
```

The Hawking Temperature proxy is a **second-order Z-score derivative** — it
measures how quickly the price is departing from its Bollinger mean relative
to where it was last bar.

```
T_H ~ z × (z − z_{prev})
```

| Reading     | Value     | Market Interpretation                     |
|-------------|-----------|-------------------------------------------|
| Cold well   | T_H < −1.5| Price accelerating back toward mean — mean reversion signal, add to long |
| Neutral     | −1.5 to 1.8| Normal operating regime                  |
| Hot well    | T_H > 1.8 | Price racing away from mean — BH evaporating, reduce longs |

A **hot well** warns that the gravitational field is losing coherence. The
physical analogy: a very light BH evaporates quickly via Hawking radiation.
A massive, cold BH is stable and persistent — exactly what the momentum
strategy wants.

```python
@property
def is_hot(self)      -> bool: return self.ht > 1.8
@property
def is_inverted(self) -> bool: return self.ht < -1.5
```

---

## 6. GravitationalLens

Gravitational lensing amplification μ measures how far price has drifted
from the TIMELIKE VWAP — the "Einstein radius" construct:

```
M     = ctl + (1 if TIMELIKE else 0)
R_E   = √M                               # Einstein radius
VWAP  = Σ(price_i × vol_i) / Σ(vol_i)   (TIMELIKE bars only)
r     = |price − VWAP| / ATR
μ     = 1 + R_E / (r + R_E)
```

When `M < 2`: `μ = max(0.3, M/3.0)` (weak field regime).

Interpretation:
- **μ ≈ 1.0**: price is far from VWAP — weak lensing, low amplification
- **μ → 2.0**: price sits exactly on top of the TIMELIKE VWAP — maximum
  lensing, strongest signal alignment
- μ acts as an additional confidence multiplier on the delta score

---

## 7. GeodesicAnalyzer

A 20-bar log-linear regression tracks the "geodesic" (shortest path through
spacetime) that price is following.

```
geo_slope = d(log P)/d(bar)  × 100         # annualised-bar slope
geo_dev   = tanh((close − P_geodesic) / ATR)  # deviation from geodesic
causal_frac = fraction of (P, P[k]) pairs where ds² < 0
```

### Causal Fraction

```python
for k in range(1, window):
    dp  = |close − closes[k]| / closes[k]
    ds2 = −(CF × k)² + dp²
    if ds2 < 0:   # TIMELIKE interval over k bars
        cc += 1
causal_frac = cc / (window − 1)
```

`causal_frac` measures what fraction of recent price history lies within
the causal light cone. A high causal fraction (> 0.75) means price has been
moving in an ordered, self-consistent manner — the trajectory is "smooth."
A low causal fraction signals the market has been jumping around randomly.

### Rapidity

Borrowed from relativistic kinematics:

```
rapidity = tanh(0.5 × ln((E + p_x) / (E − p_x)))
```

where `E = p_x = 19-bar return`. In the production formula the denominator
collapses to `1e-9` whenever returns are symmetric, giving rapidity ≈ 0.
This functions as a symmetry detector — non-zero rapidity signals a
directionally asymmetric momentum burst.

---

## 8. Delta Score — The Allocation Signal

The delta score is the primary output that flows into position sizing:

```
delta_score = tf_score × bh_mass × ATR
```

where:

| Component  | Description |
|------------|-------------|
| `tf_score` | Bitmask: bit 2 = daily BH active, bit 1 = hourly BH active, bit 0 = 15m BH active. Values 0–7. |
| `bh_mass`  | Current mass of the daily BH (or dominant timeframe) |
| `ATR`      | 14-bar Average True Range — scales signal by current volatility |

### TF Score → Position Cap Mapping

```python
TF_CAP = {7: 1.0, 6: 1.0, 4: 0.60, 3: 0.50, 2: 0.40, 1: 0.20, 0: 0.0}
```

| tf_score | Active Timeframes | Max Position Fraction |
|----------|-------------------|-----------------------|
| 7 (111)  | 1d + 1h + 15m     | 100%                  |
| 6 (110)  | 1d + 1h           | 100%                  |
| 5 (101)  | 1d + 15m          | 60%* (treated as 4)   |
| 4 (100)  | 1d only           | 60%                   |
| 3 (011)  | 1h + 15m          | 50%                   |
| 2 (010)  | 1h only           | 40%                   |
| 1 (001)  | 15m only          | 20%                   |
| 0 (000)  | None              | 0% (flat)             |

(*tf_score=5 not in TF_CAP, defaults to 0.0 via `.get(tf_score, 0.0)`;
effectively no trade unless daily is active.)

### Position Floor Logic

A position floor prevents premature exits during strong trends:

```python
if tf_score >= 6 and |target_frac| > 0.15 and bh_1h.ctl >= 5:
    pos_floor = max(pos_floor, 0.70 × |target_frac|)

if pos_floor > 0 and tf_score >= 4 and position not flat:
    target_frac = max(target_frac, pos_floor)   # prevents cutting below floor
    pos_floor  *= 0.95                           # gradual floor decay

if tf_score < 4 or target_frac == 0:
    pos_floor = 0.0                              # clear floor on exit signal
```

The floor decays at 5% per bar — similar to the mass half-life — so it
naturally fades if no fresh BH signal reinforces it.

---

## 9. BTC Lead Signal — Cross-Asset Momentum

For altcoins, BTC acts as the gravitational anchor of the crypto market:

```python
alt_score *= (1 + btc_active × 0.3)
```

where `btc_active = 1` if a BTC black hole is currently active (any
timeframe), else `0`.

This applies a **30% amplification** to any altcoin signal when BTC itself
is in a gravitational well. The intuition: when BTC momentum is strong and
directed, altcoins tend to follow with leverage. BTC black holes are leading
indicators for alt black holes.

The signal is asymmetric by design — BTC can only boost, not suppress, alt
signals. Suppression is handled separately by the Mayer Multiple dampener.

---

## 10. Mayer Multiple Dampener

The Mayer Multiple is `price / MA200`. When price greatly exceeds its
200-period moving average, the market is statistically overextended and
drawdown risk is elevated.

```python
scale = min(1.0, 2 × MA200 / price)
      = min(1.0, 2.0 / mayer_multiple)
```

| Mayer Multiple | Scale Factor | Effective Allocation |
|---------------|-------------|---------------------|
| 0.5 (deeply depressed) | 1.0 | Full signal |
| 1.0 (at MA200) | 1.0 | Full signal |
| 1.5 | 1.0 | Full signal (2/1.5 = 1.33, capped at 1.0) |
| 2.0 | 1.0 | Full signal (2/2.0 = 1.0, exactly at cap) |
| 2.5 | 0.80 | 20% reduction |
| 3.0 | 0.67 | 33% reduction |
| 4.0 | 0.50 | 50% reduction |

The dampener only activates when price is more than 2× the 200-bar moving
average. Below that level it has no effect. This is deliberately conservative
— it does not fight strong trends, it only scales back extreme parabolic moves
where historical crash risk is highest.

---

## 11. OU Overlay — Mean Reversion Sleeve

An Ornstein-Uhlenbeck process models mean-reverting dynamics:

```
dX = θ(μ − X)dt + σdW
```

| Parameter | Financial Meaning |
|-----------|-----------------|
| `θ` (kappa) | Mean reversion speed — higher = reverts faster |
| `μ`        | Long-run mean (typically 0 for log-returns) |
| `σ`        | Volatility of the noise term |
| `dW`       | Wiener process increment |

The OU overlay operates as a **separate 8% equity sleeve** dedicated purely
to mean reversion trades. It looks for situations where:

1. Price has deviated significantly from the OU mean (Hawking temperature
   is "inverted" — `T_H < −1.5`)
2. The BH is NOT active (no directional gravitational well)
3. The OU half-life `τ = ln(2)/θ` suggests reversion within a reasonable
   holding period

This keeps mean-reversion capital insulated from the momentum capital. The
two strategies are designed to be uncorrelated — momentum (BH engine) trades
with the trend, OU overlay fades extremes.

---

## 12. GARCH(1,1) Volatility Scaling

GARCH(1,1) models time-varying volatility:

```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

Typical calibrated values: `ω = 0.000001`, `α = 0.10`, `β = 0.89`
(these satisfy `α + β < 1` for stationarity).

### Position Sizing Formula

```
target_vol  = 0.90                         # 90% annualised vol target
position    = (target_vol × equity) / (σ_GARCH × √252 × price)
```

The GARCH forecast `σ_t` is updated each bar. When volatility spikes,
positions are automatically reduced. When volatility is below target, the
engine can scale up (subject to the TF_CAP ceiling).

This creates a **vol-targeting** overlay that sits on top of the BH
directional signal. Both must agree for maximum allocation:
- BH signal says direction + conviction
- GARCH scaling says how much to bet given current vol regime

---

## 13. Three-Timeframe Fusion

The engine runs three independent MinkowskiClassifier + BlackHoleDetector
pairs, each on a different time resolution:

```
cf_1d  = base_cf × 5.0     # daily: 5× base
cf_1h  = base_cf × 1.0     # hourly: base
cf_15m = base_cf × 0.35    # 15-minute: 35% of base
```

CF is scaled by timeframe because longer bars accumulate more price movement;
a 1.5% daily move is "normal" while a 1.5% 15-minute move is a shock.

### Synchronization Architecture

```
Daily loop ─────────────────────────────────────────────────────────►
  │  For each daily bar:
  │    - Update mc_1d, bh_1d
  │    - Update daily indicators (EMA12/26/50/200, ATR, ADX, BB)
  │    - Determine regime
  │
  │  Hourly loop ──────────────────────────────────────────────────►
  │    │  For each hourly bar within this day:
  │    │    - Update mc_1h, bh_1h
  │    │
  │    │  15m loop ────────────────────────────────────────────────►
  │    │    │  For each 15m bar within this hour:
  │    │    │    - Update mc_15m, bh_15m
  │    │    └──────────────────────────────────────────────────────
  │    │
  │    │  Compute tf_score (bitmask from 3 BH states)
  │    │  Compute position target from TF_CAP[tf_score]
  │    │  Execute trade logic
  │    └──────────────────────────────────────────────────────────
  │
  └──────────────────────────────────────────────────────────────
```

Direction is determined by the most reliable timeframe:
1. Daily BH direction (if active)
2. Hourly BH direction (fallback)
3. 15m BH direction (last resort)

For `long_only=True` mode, negative direction is treated as zero (flat).

---

## 14. Parameter Calibration Table

### CF Values by Instrument Class

| Symbol   | Asset Class     | CF      | Notes                                  |
|----------|-----------------|---------|----------------------------------------|
| BTC      | Crypto          | 0.005   | 0.5% bar = boundary of ordered move   |
| ETH      | Crypto          | 0.007   | More volatile than BTC                 |
| SOL      | Crypto          | 0.010   | High-beta alt                          |
| ES       | Equity Index    | 0.001   | S&P 500 futures — tight                |
| NQ       | Equity Index    | 0.0012  | Nasdaq slightly wider                  |
| QQQ      | Equity Index    | 0.0012  | ETF equivalent of NQ                   |
| CL       | Energy          | 0.015   | Crude oil — wide swings                |
| NG       | Energy          | 0.020   | Natural gas — widest                   |
| GC       | Metals          | 0.008   | Gold                                   |
| SI       | Metals          | 0.008   | Silver                                 |
| ZB       | Bond            | 0.003   | 30-year T-Bond                         |
| ZN       | Bond            | 0.003   | 10-year T-Note                         |
| EURUSD   | Forex           | 0.0005  | Major pair — very tight                |
| GBPUSD   | Forex           | 0.0005  | Major pair                             |
| USDJPY   | Forex           | 0.0005  | Major pair                             |
| VIX      | Volatility      | 0.025   | VIX is inherently spiky                |

### Timeframe CF Scaling

| Timeframe | CF Multiplier | Rationale |
|-----------|---------------|-----------|
| Daily     | 5.0×          | Daily bars accumulate ~5× typical hourly move |
| Hourly    | 1.0× (base)   | CF is calibrated at hourly resolution |
| 15-minute | 0.35×         | 15m moves are ~1/3 of hourly         |

### BH Parameters

| Parameter    | Value | Description |
|--------------|-------|-------------|
| BH_FORM      | 1.5   | Mass required to form a gravitational well |
| BH_COLLAPSE  | 1.0   | Mass below which well collapses |
| BH_DECAY     | 0.95  | Per-bar mass retention (5% bleeds away) |

---

## 15. Worked Example — Full BTC Signal Trace

### Setup

```
Symbol: BTC
CF (hourly): 0.005
BH_FORM: 1.5, BH_COLLAPSE: 1.0, BH_DECAY: 0.95
Starting bh_mass: 0.0, ctl: 0
```

### Bar-by-Bar Trace

```
Bar  Close    ΔP/P     beta    bit        ctl  sb     mass_input  bh_mass   active
─────────────────────────────────────────────────────────────────────────────────
 1   42000    +0.001   0.20    TIMELIKE   1    1.10   0.110       0.104     False
 2   42250    +0.006   1.19    SPACELIKE  0    —      decay×0.7   0.073     False
 3   42100    −0.004   0.71    TIMELIKE   1    1.10   0.377       0.444     False
 4   42400    +0.007   1.40    SPACELIKE  0    —      decay×0.7   0.311     False
 5   42600    +0.005   1.00    TIMELIKE   1    1.10   0.527       0.822     False
 6   42900    +0.007   1.40    SPACELIKE  0    —      decay×0.7   0.575     False
 7   43100    +0.005   0.94    TIMELIKE   1    1.10   0.527       1.073     False
 8   43350    +0.006   1.11    SPACELIKE  0    —      decay×0.7   0.751     False
 9   43600    +0.006   1.11    TIMELIKE   1    1.10   0.631       1.345     False
10   43900    +0.007   1.33    TIMELIKE   2    1.20   0.800       2.078     False
11   44200    +0.007   1.30    TIMELIKE   3    1.30   0.869       2.843     False
12   44500    +0.007   1.58    TIMELIKE   4    1.40   0.940       3.601     False
13   44800    +0.007   1.57    TIMELIKE   5    1.50   1.007       4.421    *TRUE*
                                          ctl≥5 AND mass>1.5 → BH FORMS
```

**Formation event at bar 13.**

### Delta Score Computation (bar 13, hourly timeframe only)

```
tf_score = 4 (daily BH active) | 2 (hourly BH active) | 0 (no 15m) = 6
ceiling  = TF_CAP[6] = 1.0

ATR (14-bar) ≈ 350 points
bh_mass_hourly = 4.421

delta_score = tf_score × bh_mass × ATR
            = 6 × 4.421 × 350
            = 9,284

position_frac = TF_CAP[6] × direction = 1.0 × (+1) = 1.0 (100% long)
```

### Hawking Temperature Check

```
Bollinger Middle (20-bar): 43,000
Bollinger Std: 800
Z = (44,800 − 43,000) / 800 = 2.25
Z_prev (bar 12) = (44,500 − 43,000) / 800 = 1.875
T_H = Z × (Z − Z_prev) = 2.25 × (2.25 − 1.875) = 0.844

T_H = 0.844 → NEUTRAL (not hot, not inverted) → proceed with long
```

### Mayer Multiple Check

```
MA200 ≈ 38,000
price = 44,800
mayer = 44,800 / 38,000 = 1.179

scale = min(1.0, 2 × 38,000 / 44,800) = min(1.0, 1.696) = 1.0
→ No dampening (mayer < 2.0)
```

### BTC Lead (already BTC, no cross-asset needed)

BTC is the lead itself, so `btc_active` amplification applies to alts only.

### Final Position

```
raw_fraction    = 1.0  (TF_CAP[6], long)
mayer_scale     = 1.0  (below 2× MA200)
garch_scale     = target_vol / (σ_GARCH × √252)
                ≈ 0.90 / (0.65 × √252)         (σ_GARCH ≈ 0.041/bar)
                ≈ 0.90 / 10.32
                ≈ 0.087  (8.7% of equity per point)

dollar_position = 0.087 × equity / price
                = 0.087 × $1,000,000 / 44,800
                ≈ 1.94 BTC
```

---

## 16. Why It Works — Physical Intuition

### Causal vs Non-Causal Price Moves

The core insight of SRFM is that financial markets generate two fundamentally
different types of price movement:

**TIMELIKE moves** (β < 1): These are *informed* moves. They propagate at a
rate consistent with information diffusion through a network of connected
participants. When a market trends in a consistent, sub-luminal manner, it
is signalling that there is an ongoing imbalance of information — someone
knows something and is gradually expressing it. These moves are *causal* in
the Granger sense: past prices predict future prices within the light cone.

**SPACELIKE moves** (β ≥ 1): These are *shocked* moves. They propagate faster
than any causal mechanism can explain — they are market microstructure
events: stop-hunting cascades, gap openings, news shocks, forced
liquidations. These moves have no predictive content; they are noise that
interrupts the signal.

### Why Consecutive TIMELIKE Bars Matter

The `ctl` (consecutive timelike) counter and the `ctl ≥ 5` gate are
statistically motivated. A single TIMELIKE bar is noise. Five consecutive
TIMELIKE bars with growing mass represents a directional flow that has
persisted through multiple bar transitions — it is unlikely to be
coincidental.

The slingshot boost `sb = min(2, 1 + ctl × 0.1)` gives extra weight to
conviction that has been sustained longer. This mirrors how in physics,
a massive object that has been accelerating consistently is harder to
deflect than one that has only recently begun moving.

### Why This Is Not Just a Momentum Indicator

Standard momentum indicators (RSI, MACD) measure the magnitude and direction
of price change. The BH engine measures something subtler: the **topological
character** of price movement — whether it is occurring within or outside the
causal light cone. Two price series can have identical magnitude of momentum
but radically different BH mass depending on whether the moves were smooth
(timelike) or spiky (spacelike).

This is why BH_DECAY is critical — it is not enough to have been trending.
The trend must be *currently active* and *continuously confirmed* by fresh
causal evidence.
