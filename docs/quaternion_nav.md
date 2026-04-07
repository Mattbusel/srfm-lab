# Quaternion Navigation Layer

A mathematical navigation system for moving through SRFM 4-space. Each bar's position is
represented as a unit quaternion, and successive bar quaternions define a trajectory whose
geometric properties become trading signals.

Status: **live in observability mode** as of April 2026. Signals are written to
`execution/live_trades.db` on every bar but are not yet wired into entry or exit logic.

---

## Background

The BH Physics engine classifies bars using a Minkowski metric and accumulates mass on
causal (timelike) sequences. It answers the question: *is this market in a structured
regime?* What it does not model is *how the market is moving through that regime* -- the
direction, velocity, and curvature of the trajectory in the 4-dimensional space the metric
defines.

The quaternion nav layer adds that missing piece. It represents each bar as a point in
4-space, tracks the rotation between consecutive bars, and measures how much the market
deviates from its predicted inertial path. Those three quantities feed directly into the
existing BH framework as additional observability signals.

---

## The 4-Space

Each bar maps to a point in the space `(t, P, V, MI)` where:

- `t` -- normalized elapsed time since the previous bar (1.0 per standard bar interval)
- `P` -- normalized close price (close / rolling max close, in `(0, 1]`)
- `V` -- normalized volume (volume / rolling max volume, in `(0, 1]`)
- `MI` -- market impact proxy: `|log_return| * sqrt(volume / vol_ema)`

Market impact approximates Kyle's lambda: it is large when a big price move is accompanied
by high volume relative to the recent average. Rolling max normalizers decay slowly
(`price_max *= 0.99995` per bar, `vol_max *= 0.9999`, `mi_max *= 0.9998`) so they adapt
to regime shifts without resetting discontinuously.

The raw 4-vector is then converted to a unit quaternion:

```
q_bar = normalize([t_norm, P_norm, V_norm, MI_norm])
```

All subsequent operations work on unit quaternions, so the magnitude constraint `|q| = 1`
is enforced at every step and acts as an early-warning check: any deviation from unit norm
indicates accumulated floating-point error in the pipeline.

---

## Five Components

### 1. QuaternionBar

Construction of `q_bar` for each incoming bar. The raw 4-vector is normalized so
`|q_bar| = 1.0`. A small epsilon (`1e-12`) is added to each component before normalization
so that the zero vector (which would produce a degenerate quaternion) is never possible
even on the first bar.

The bar quaternion represents where the market *is* in 4-space at this moment. It is
stored alongside the BH output in `SignalOutput` and written to `nav_state` in SQLite.

### 2. Rotation Tracking

The quaternion product `delta_q = q_new * q_prev^(-1)` gives the rotation that maps the
previous bar's position to the current one. For unit quaternions the inverse is the
conjugate (negate x, y, z), so this is numerically cheap.

`Q_current` is the running product of all delta rotations since the navigator was
initialized:

```
Q_current = delta_q * Q_current   (normalized after every step)
```

`Q_current` represents the market's current heading in price-space: the accumulated
rotational state of the regime. Two regimes that look identical in terms of BH mass but
have different `Q_current` orientations are heading in different directions.

Normalization is enforced after every product. Drift in `|Q_current|` compounds silently
and corrupts all downstream signals; the test suite checks `|Q_current| - 1.0 < 1e-11`
after every bar across a 500-bar stress sequence.

### 3. Geodesic Navigation

The shortest path between two quaternions on the unit 3-sphere is a great-circle arc,
computed via SLERP:

```
SLERP(q1, q2, t) = q1 * (q1^{-1} * q2)^t
```

The implementation chooses the shorter arc (negates q2 when `dot(q1, q2) < 0`) and falls
back to linear interpolation when the two quaternions are nearly identical (angle less
than `1e-10` radians), avoiding division by zero in the `sin(theta)` denominator.

For *extrapolation* -- predicting where the next bar should land on an inertial path --
the implementation applies the same rotation one more time rather than using SLERP with
`t = 2.0`. This is more numerically stable near `theta = pi`:

```
delta = q2 * q1^{-1}          # rotation from q1 to q2
q_predicted = delta * q2       # apply same rotation forward
```

**Geodesic deviation** is then:

```
deviation = geodesic_angle(q_predicted, q_actual)
```

where `geodesic_angle(a, b) = 2 * arccos(|dot(a, b)|)`.

Near BH mass concentrations the gravitational well bends the path away from its inertial
trajectory. The deviation is scaled by a curvature correction:

```
deviation_corrected = deviation * (1 + 0.15 * bh_mass)
```

The constant `0.15` is a principled prior, not a calibrated value. It should be treated
as a free parameter until the nav layer has enough live history to fit it properly.

**Interpretation:** A high geodesic deviation means the market took a significantly
different path than its recent momentum predicted. Near an active BH this is expected and
amplified; far from a BH it may indicate noise or an exogenous shock.

### 4. Regime Rotation Velocity

```
angular_velocity = 2 * arccos(|delta_q.w|)    (radians per bar)
```

This is the angle swept by `Q_current` in one bar -- the instantaneous rate of rotation
in price-space.

- **Low omega:** the market is moving in a stable, consistent direction. The regime
  heading is not changing. This is the BH formation regime.
- **High omega:** the market is rotating rapidly through orientations. Regime transitions,
  volatility spikes, and mean-reversion events all produce high omega.

The intended downstream use is a sizing multiplier: reduce position size when omega is
elevated, because the regime heading is unstable and the BH signal is less reliable.
This is not wired in yet.

### 5. Lorentz Boost as Quaternion Rotation

When the BH active flag transitions (false to true on formation, true to false on
collapse), the reference frame of the market changes. In the existing code this appears
as a hard parameter discontinuity. The nav layer instead expresses it as a quaternion
rotation in the time-price (w-x) plane:

```
rapidity = arctanh(min(0.99, bh_mass * 0.40))
q_boost = normalize([cos(rapidity/2), sin(rapidity/2), 0, 0])
Q_current = normalize(q_boost * Q_current)
```

The rapidity is proportional to BH mass at the moment of transition, so a high-mass BH
collapse produces a larger frame rotation than a marginal threshold crossing. The boost
angle approaches `pi/2` as mass approaches the cap.

This gives smooth, continuous transitions in `Q_current` instead of a reset or jump. Two
identical market sequences that differ only in the timing of a BH activation will have
different `Q_current` orientations afterward, which is the correct behavior: the
gravitational event changed the reference frame.

The Lorentz scale constant (`0.40`) is also a prior that should be calibrated on live
data.

---

## Implementation

### C++ (signal engine)

```
cpp/signal-engine/src/quaternion/quat_nav.hpp    -- QuatNav class + QuatNavOutput struct
cpp/signal-engine/src/quaternion/quat_nav.cpp    -- full implementation
cpp/signal-engine/tests/test_quat_nav.cpp        -- 15 test cases
```

`QuatNav` is a member of `InstrumentState` in `feed_processor.hpp`. It is updated in
`fill_signal_output()` immediately after the BH physics block. The six nav fields
(`nav_qw`, `nav_qx`, `nav_qy`, `nav_qz`, `nav_angular_vel`, `nav_geodesic_dev`) are
written to `SignalOutput` and available to any downstream consumer of the signal engine.

### Python (live trader)

```
bridge/quat_nav_bridge.py          -- QuatNavPy class + NavStateWriter
tests/test_quat_nav_bridge.py      -- 22 test cases
```

`QuatNavPy` mirrors the C++ implementation exactly. `NavStateWriter` creates the
`nav_state` table in `execution/live_trades.db` and writes one row per bar per timeframe.
Three `QuatNavPy` instances run per instrument (15m, 1h, 4h), updated in `_on_15m_bar()`,
`_flush_1h()`, and `_flush_4h()`.

Nav writes are suppressed during bootstrap. All nav code is wrapped in try/except so a
nav failure cannot affect trading logic.

### Database schema

```sql
CREATE TABLE IF NOT EXISTS nav_state (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol                  TEXT    NOT NULL,
    timeframe               TEXT    NOT NULL DEFAULT '15m',
    bar_time                TEXT    NOT NULL,
    timestamp_ns            INTEGER NOT NULL,
    bar_qw                  REAL    NOT NULL,
    bar_qx                  REAL    NOT NULL,
    bar_qy                  REAL    NOT NULL,
    bar_qz                  REAL    NOT NULL,
    qw                      REAL    NOT NULL,
    qx                      REAL    NOT NULL,
    qy                      REAL    NOT NULL,
    qz                      REAL    NOT NULL,
    angular_velocity        REAL    NOT NULL,
    geodesic_deviation      REAL    NOT NULL,
    bh_mass                 REAL    NOT NULL DEFAULT 0.0,
    bh_active               INTEGER NOT NULL DEFAULT 0,
    lorentz_boost_applied   INTEGER NOT NULL DEFAULT 0,
    lorentz_boost_rapidity  REAL    NOT NULL DEFAULT 0.0,
    strategy_version        TEXT    NOT NULL DEFAULT 'larsa_v17'
)
```

---

## Signals and Suggested Integration Points

These are not wired into entry or exit logic. They are listed here as candidates for
validation once sufficient live history exists.

| Signal | Query | Suggested use |
|---|---|---|
| `angular_velocity` | `SELECT AVG(angular_velocity) FROM nav_state WHERE symbol=? AND bar_time > ?` | Sizing multiplier: `1 / (1 + k * omega)`. Size down during rapid rotation. |
| `geodesic_deviation` | `SELECT geodesic_deviation FROM nav_state WHERE ...` | Conviction gate: reject entries when deviation is high relative to recent baseline. |
| `Q_current` orientation | `SELECT qw,qx,qy,qz FROM nav_state WHERE ...` | Regime fingerprint: cluster by orientation proximity to detect recurring states. |
| `lorentz_boost_applied` | `SELECT * FROM nav_state WHERE lorentz_boost_applied=1` | IAE pattern feature: correlate boost events with subsequent PnL. |
| `geodesic_deviation` near BH | Join with `bh_active=1` | Curvature-corrected deviation during active BH -- this is the signal most likely to carry independent information. |

### Validation query to run after 2-4 weeks of live data

```sql
SELECT
    n.symbol,
    CASE WHEN n.angular_velocity > 0.5 THEN 'high_omega' ELSE 'low_omega' END AS regime,
    AVG(p.pnl) AS avg_pnl,
    COUNT(*) AS n_trades
FROM nav_state n
JOIN trade_pnl p
    ON n.symbol = p.symbol
    AND n.bar_time <= p.entry_time
    AND n.timeframe = '1h'
GROUP BY n.symbol, regime
ORDER BY n.symbol, regime;
```

If `low_omega` trades outperform `high_omega` trades consistently across symbols, the
angular velocity signal is carrying real information. That is the trigger for wiring it
into sizing.

---

## Tests

### C++ (15 test cases in `tests/test_quat_nav.cpp`)

- `quat_mul`: `q * q^{-1} = identity` to `1e-12`
- `quat_normalize`: unit norm on arbitrary input, small input, and zero input (identity fallback)
- `slerp`: `t=0` returns `q1`, `t=1` returns `q2`, midpoint is equidistant, output is always unit norm
- `geodesic_angle`: identical quaternions give 0, orthogonal quaternions give `pi/2`
- `quat_extrapolate`: collinear sequence gives near-zero prediction error
- `QuatNav` 500-bar: `|Q_current| - 1.0 < 1e-12` at every bar
- `QuatNav` reset: restores identity
- Angular velocity: volatile regime has higher cumulative omega than stable
- Geodesic deviation: curvature correction increases with mass
- Lorentz boost: fires on BH activation, shifts orientation, both quaternions stay unit norm
- Stress (1000 bars with price shocks): all fields finite, all norms unit

### Python (22 test cases in `tests/test_quat_nav_bridge.py`)

Same coverage as C++ plus:
- DB round-trip: write 5 nav records, read back, verify every field to `1e-14`
- Silent error handling: write to closed connection logs warning, does not raise

Run with:

```bash
python tests/test_quat_nav_bridge.py
# 22 passed, 0 failed
```

---

## Caveats

**The curvature constant (0.15) and Lorentz scale (0.40) are priors.** They were chosen
to be order-of-magnitude reasonable given the BH mass range in live trading (0 to ~4).
They have not been fit to data and should be treated as hyperparameters once live history
accumulates.

**The 4-space representation may not be orthogonal to existing signals.** Price, volume,
and market impact are all inputs to BHState as well. The quaternion rotation captures a
different geometric property (the angle between consecutive positions rather than the
magnitude of any single coordinate), but correlation with existing signals has not been
measured.

**SLERP extrapolation assumes smooth rotation.** Markets can jump discontinuously. In
those cases geodesic deviation will spike regardless of BH mass, which is probably correct
behavior (a price gap is by definition a deviation from an inertial path) but the signal
will need a baseline filter to distinguish informative spikes from mechanical ones.
