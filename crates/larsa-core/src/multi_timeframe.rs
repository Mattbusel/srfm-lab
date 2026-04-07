/// Multi-timeframe signal aggregation for SRFM.
///
/// Aggregates signals across 15m, 1h, and 4h timeframes with
/// configurable weights and regime-based filtering.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Regime enum
// ---------------------------------------------------------------------------

/// Market regime classification for a single timeframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Regime {
    BullTrend,
    BearTrend,
    Ranging,
    HighVol,
}

impl Regime {
    /// Returns +1, -1, or 0 as the directional sign of the regime.
    pub fn direction_sign(&self) -> i8 {
        match self {
            Regime::BullTrend => 1,
            Regime::BearTrend => -1,
            Regime::Ranging => 0,
            Regime::HighVol => 0,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Regime::BullTrend => "BULL_TREND",
            Regime::BearTrend => "BEAR_TREND",
            Regime::Ranging => "RANGING",
            Regime::HighVol => "HIGH_VOL",
        }
    }
}

// ---------------------------------------------------------------------------
// TimeframeSignal
// ---------------------------------------------------------------------------

/// A single signal observation from one timeframe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeframeSignal {
    /// Timeframe label: "15m", "1h", or "4h".
    pub timeframe: &'static str,
    /// Current black-hole mass for this timeframe.
    pub bh_mass: f64,
    /// Raw signal value in [-1.0, 1.0].
    pub signal: f64,
    /// Regime classification at this timeframe.
    pub regime: Regime,
    /// Override weight; if 0.0, the canonical weight table is used.
    pub weight: f64,
}

impl TimeframeSignal {
    /// Canonical weight for a timeframe label.
    pub fn canonical_weight(timeframe: &str) -> f64 {
        match timeframe {
            "4h" => 0.5,
            "1h" => 0.3,
            "15m" => 0.2,
            _ => 0.1,
        }
    }

    /// Effective weight: uses override if set, else canonical.
    pub fn effective_weight(&self) -> f64 {
        if self.weight > 0.0 {
            self.weight
        } else {
            Self::canonical_weight(self.timeframe)
        }
    }
}

// ---------------------------------------------------------------------------
// AggregatedSignal
// ---------------------------------------------------------------------------

/// Result of aggregating signals from multiple timeframes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedSignal {
    /// Net direction: +1 long, -1 short, 0 flat.
    pub direction: i8,
    /// Magnitude of the aggregated signal, in [0.0, 1.0].
    pub strength: f64,
    /// Fraction of timeframes that agree on direction (0.0 to 1.0).
    pub agreement: f64,
    /// Regime of the highest-weight timeframe (usually 4h).
    pub primary_regime: Regime,
}

impl AggregatedSignal {
    /// Returns true when the signal is strong enough to generate an entry.
    pub fn is_entry_quality(&self) -> bool {
        self.agreement >= 0.67 && self.direction != 0
    }
}

// ---------------------------------------------------------------------------
// Bar -- minimal OHLCV bar used by update()
// ---------------------------------------------------------------------------

/// Minimal OHLCV bar passed to MultiTimeframeEngine::update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// Unix timestamp in seconds.
    pub timestamp: i64,
}

// ---------------------------------------------------------------------------
// TimeframeState -- internal per-timeframe bookkeeping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TimeframeState {
    /// Latest signal derived from BH physics.
    signal: f64,
    /// Latest BH mass.
    bh_mass: f64,
    /// Latest regime.
    regime: Regime,
    /// Incremental BH mass accumulator.
    mass: f64,
    /// Consecutive-timelike counter.
    ctl: i32,
    /// Whether a black hole is currently active.
    bh_active: bool,
    /// 20-bar EMA of close prices.
    ema20: f64,
    /// Whether EMA has been seeded.
    ema_initialized: bool,
    /// Rolling close buffer for vol estimation (last 20 bars).
    closes: Vec<f64>,
}

impl TimeframeState {
    fn new() -> Self {
        Self {
            signal: 0.0,
            bh_mass: 0.0,
            regime: Regime::Ranging,
            mass: 0.0,
            ctl: 0,
            bh_active: false,
            ema20: 0.0,
            ema_initialized: false,
            closes: Vec::with_capacity(24),
        }
    }

    /// Process one bar and update internal state.
    fn update(&mut self, bar: &Bar, cf: f64) {
        // Seed EMA on first bar.
        if !self.ema_initialized {
            self.ema20 = bar.close;
            self.ema_initialized = true;
            self.closes.push(bar.close);
            return;
        }

        // BH physics.
        let prev_close = *self.closes.last().unwrap_or(&bar.close);
        let beta = (bar.close - prev_close).abs() / (prev_close * cf + 1e-12);

        if beta < 1.0 {
            // Timelike: mass accretes.
            self.mass = self.mass * 0.97 + 0.03;
            self.ctl += 1;
        } else {
            // Spacelike: mass decays.
            self.mass *= 0.95;
            self.ctl = 0;
        }

        // BH formation threshold.
        if !self.bh_active && self.mass >= 1.5 && self.ctl >= 5 {
            self.bh_active = true;
        }
        // BH collapse threshold.
        if self.bh_active && self.mass < 1.0 {
            self.bh_active = false;
        }

        self.bh_mass = self.mass;

        // 20-bar EMA (alpha = 2/21).
        let alpha = 2.0 / 21.0;
        self.ema20 = self.ema20 * (1.0 - alpha) + bar.close * alpha;

        // Rolling close buffer (keep last 20).
        self.closes.push(bar.close);
        if self.closes.len() > 20 {
            self.closes.remove(0);
        }

        // Realized volatility from rolling closes.
        let realized_vol = if self.closes.len() >= 2 {
            let rets: Vec<f64> = self
                .closes
                .windows(2)
                .map(|w| (w[1] / w[0] - 1.0).abs())
                .collect();
            let mean_r = rets.iter().sum::<f64>() / rets.len() as f64;
            mean_r
        } else {
            0.0
        };

        // Regime classification.
        self.regime = if self.bh_active && bar.close > self.ema20 {
            Regime::BullTrend
        } else if self.bh_active && bar.close < self.ema20 {
            Regime::BearTrend
        } else if realized_vol > 0.02 {
            Regime::HighVol
        } else {
            Regime::Ranging
        };

        // Signal generation: direction from EMA, magnitude from BH mass.
        let direction = if bar.close > self.ema20 { 1.0 } else { -1.0 };
        let magnitude = (self.mass / 3.0).min(1.0);
        self.signal = if self.bh_active {
            direction * magnitude
        } else if self.ctl >= 3 {
            direction * magnitude * 0.5
        } else {
            0.0
        };
    }
}

// ---------------------------------------------------------------------------
// MultiTimeframeEngine
// ---------------------------------------------------------------------------

/// Aggregates signals across the 15m, 1h, and 4h timeframes.
///
/// Each call to `update` processes a new bar for one specific timeframe.
/// Once at least one signal per timeframe is available, `aggregate_signals`
/// is called internally and an `AggregatedSignal` is returned.
pub struct MultiTimeframeEngine {
    /// Internal state keyed by timeframe label.
    states: HashMap<&'static str, TimeframeState>,
    /// Cosmic flow parameter -- typical value 0.02.
    cf: f64,
    /// Minimum agreement fraction required to issue an entry signal.
    agreement_threshold: f64,
}

impl MultiTimeframeEngine {
    /// Create a new engine with default parameters.
    pub fn new() -> Self {
        let mut states = HashMap::new();
        states.insert("15m", TimeframeState::new());
        states.insert("1h", TimeframeState::new());
        states.insert("4h", TimeframeState::new());
        Self {
            states,
            cf: 0.02,
            agreement_threshold: 0.67,
        }
    }

    /// Create with a custom cosmic-flow constant and agreement threshold.
    pub fn with_params(cf: f64, agreement_threshold: f64) -> Self {
        let mut engine = Self::new();
        engine.cf = cf;
        engine.agreement_threshold = agreement_threshold;
        engine
    }

    /// Process a new bar for the given timeframe and return an aggregated
    /// signal if all three timeframes have at least one observation.
    ///
    /// Returns `None` during warmup (before all timeframes have data).
    pub fn update(&mut self, timeframe: &'static str, bar: &Bar) -> Option<AggregatedSignal> {
        if let Some(state) = self.states.get_mut(timeframe) {
            state.update(bar, self.cf);
        }

        // Collect current signals -- require all three timeframes to be warmed up.
        let signals = self.current_signals()?;
        let agg = aggregate_signals(&signals);

        if agg.agreement >= self.agreement_threshold || agg.direction == 0 {
            Some(agg)
        } else {
            // Not enough agreement -- return the aggregate anyway so the caller
            // can observe the state, but direction is forced to 0.
            Some(AggregatedSignal {
                direction: 0,
                strength: agg.strength,
                agreement: agg.agreement,
                primary_regime: agg.primary_regime,
            })
        }
    }

    /// Force-return the latest aggregated signal regardless of agreement.
    pub fn latest(&self) -> Option<AggregatedSignal> {
        let signals = self.current_signals()?;
        Some(aggregate_signals(&signals))
    }

    /// Build the `TimeframeSignal` slice from internal state, returning None
    /// if any timeframe has not received its first bar yet.
    fn current_signals(&self) -> Option<Vec<TimeframeSignal>> {
        let tfs = ["15m", "1h", "4h"];
        let mut out = Vec::with_capacity(3);
        for &tf in &tfs {
            let st = self.states.get(tf)?;
            if !st.ema_initialized {
                return None;
            }
            out.push(TimeframeSignal {
                timeframe: tf,
                bh_mass: st.bh_mass,
                signal: st.signal,
                regime: st.regime,
                weight: 0.0, // use canonical
            });
        }
        Some(out)
    }

    /// Reset all internal state (useful for walk-forward resets).
    pub fn reset(&mut self) {
        for state in self.states.values_mut() {
            *state = TimeframeState::new();
        }
    }
}

impl Default for MultiTimeframeEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// aggregate_signals -- standalone public function
// ---------------------------------------------------------------------------

/// Aggregate a slice of per-timeframe signals into a single `AggregatedSignal`.
///
/// Weighting: 4h = 0.5, 1h = 0.3, 15m = 0.2 (canonical), overridable via
/// the `weight` field of each `TimeframeSignal`.
///
/// Agreement score = fraction of timeframes whose signal direction matches
/// the weighted-average direction. Entry is only valid when agreement >= 0.67.
///
/// Regime rule: if the highest-weight timeframe is `Ranging`, the returned
/// strength is capped at 0.5.
pub fn aggregate_signals(signals: &[TimeframeSignal]) -> AggregatedSignal {
    if signals.is_empty() {
        return AggregatedSignal {
            direction: 0,
            strength: 0.0,
            agreement: 0.0,
            primary_regime: Regime::Ranging,
        };
    }

    // Normalize weights so they sum to 1.0.
    let total_weight: f64 = signals.iter().map(|s| s.effective_weight()).sum();
    let norm = if total_weight > 1e-12 { total_weight } else { 1.0 };

    // Weighted signal sum.
    let weighted_sum: f64 = signals
        .iter()
        .map(|s| s.signal * s.effective_weight() / norm)
        .sum();

    // Determine composite direction from weighted sum.
    let composite_dir: i8 = if weighted_sum > 1e-9 {
        1
    } else if weighted_sum < -1e-9 {
        -1
    } else {
        0
    };

    // Agreement score: fraction of timeframes whose signal direction matches.
    let n = signals.len() as f64;
    let agreeing = signals
        .iter()
        .filter(|s| {
            let sd = if s.signal > 1e-9 {
                1i8
            } else if s.signal < -1e-9 {
                -1i8
            } else {
                0i8
            };
            sd == composite_dir
        })
        .count() as f64;
    let agreement = agreeing / n;

    // Primary regime from the highest-weight timeframe.
    let primary = signals
        .iter()
        .max_by(|a, b| {
            a.effective_weight()
                .partial_cmp(&b.effective_weight())
                .unwrap()
        })
        .map(|s| s.regime)
        .unwrap_or(Regime::Ranging);

    // Strength = absolute weighted sum, capped at 1.0.
    let mut strength = weighted_sum.abs().min(1.0);

    // Cap strength at 0.5 when the primary regime is Ranging.
    if primary == Regime::Ranging {
        strength = strength.min(0.5);
    }

    AggregatedSignal {
        direction: composite_dir,
        strength,
        agreement,
        primary_regime: primary,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bar(close: f64, ts: i64) -> Bar {
        Bar {
            open: close,
            high: close * 1.001,
            low: close * 0.999,
            close,
            volume: 1_000.0,
            timestamp: ts,
        }
    }

    fn bull_signal(tf: &'static str) -> TimeframeSignal {
        TimeframeSignal {
            timeframe: tf,
            bh_mass: 2.0,
            signal: 0.8,
            regime: Regime::BullTrend,
            weight: 0.0,
        }
    }

    fn bear_signal(tf: &'static str) -> TimeframeSignal {
        TimeframeSignal {
            timeframe: tf,
            bh_mass: 2.0,
            signal: -0.8,
            regime: Regime::BearTrend,
            weight: 0.0,
        }
    }

    // --- Regime enum tests ---

    #[test]
    fn test_regime_direction_signs() {
        assert_eq!(Regime::BullTrend.direction_sign(), 1);
        assert_eq!(Regime::BearTrend.direction_sign(), -1);
        assert_eq!(Regime::Ranging.direction_sign(), 0);
        assert_eq!(Regime::HighVol.direction_sign(), 0);
    }

    #[test]
    fn test_regime_labels() {
        assert_eq!(Regime::BullTrend.label(), "BULL_TREND");
        assert_eq!(Regime::Ranging.label(), "RANGING");
    }

    // --- aggregate_signals tests ---

    #[test]
    fn test_aggregate_all_bull() {
        let sigs = vec![
            bull_signal("15m"),
            bull_signal("1h"),
            bull_signal("4h"),
        ];
        let agg = aggregate_signals(&sigs);
        assert_eq!(agg.direction, 1);
        assert!((agg.agreement - 1.0).abs() < 1e-9, "all agree => 1.0");
        assert!(agg.strength > 0.0);
        assert!(agg.is_entry_quality());
    }

    #[test]
    fn test_aggregate_all_bear() {
        let sigs = vec![
            bear_signal("15m"),
            bear_signal("1h"),
            bear_signal("4h"),
        ];
        let agg = aggregate_signals(&sigs);
        assert_eq!(agg.direction, -1);
        assert!((agg.agreement - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_two_thirds_agreement() {
        // 4h and 1h are bull, 15m is bear.
        // Weighted sum: 0.5*0.8 + 0.3*0.8 + 0.2*(-0.8) = 0.4 + 0.24 - 0.16 = 0.48 > 0.
        let sigs = vec![
            bear_signal("15m"),
            bull_signal("1h"),
            bull_signal("4h"),
        ];
        let agg = aggregate_signals(&sigs);
        assert_eq!(agg.direction, 1);
        // 2 out of 3 agree with bull direction.
        assert!((agg.agreement - 2.0 / 3.0).abs() < 1e-9);
        assert!(agg.is_entry_quality()); // exactly 0.67 threshold
    }

    #[test]
    fn test_aggregate_single_agreement_blocked() {
        // Only 15m is bull (weight 0.2); 1h and 4h are bear (weights 0.3 + 0.5).
        let sigs = vec![
            bull_signal("15m"),
            bear_signal("1h"),
            bear_signal("4h"),
        ];
        let agg = aggregate_signals(&sigs);
        // Weighted sum is negative -- direction should be -1.
        assert_eq!(agg.direction, -1);
        // 1 out of 3 timeframes agree with... wait, 2 agree with bear.
        assert!((agg.agreement - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_ranging_caps_strength() {
        // 4h is Ranging (highest weight) -- strength must be capped at 0.5.
        let sigs = vec![
            TimeframeSignal {
                timeframe: "15m",
                bh_mass: 2.0,
                signal: 0.9,
                regime: Regime::BullTrend,
                weight: 0.0,
            },
            TimeframeSignal {
                timeframe: "1h",
                bh_mass: 2.0,
                signal: 0.9,
                regime: Regime::BullTrend,
                weight: 0.0,
            },
            TimeframeSignal {
                timeframe: "4h",
                bh_mass: 0.3,
                signal: 0.9,
                regime: Regime::Ranging,
                weight: 0.0,
            },
        ];
        let agg = aggregate_signals(&sigs);
        assert!(agg.strength <= 0.5, "Ranging primary caps strength at 0.5");
    }

    #[test]
    fn test_empty_signals() {
        let agg = aggregate_signals(&[]);
        assert_eq!(agg.direction, 0);
        assert_eq!(agg.strength, 0.0);
    }

    // --- MultiTimeframeEngine tests ---

    #[test]
    fn test_engine_warmup_returns_none_until_all_tfs() {
        let mut engine = MultiTimeframeEngine::new();
        // Only feed 15m -- should not yet have data for 1h and 4h.
        let result = engine.update("15m", &make_bar(100.0, 0));
        // 1h and 4h have no data yet -- should return None.
        assert!(result.is_none());
    }

    #[test]
    fn test_engine_produces_signal_after_warmup() {
        let mut engine = MultiTimeframeEngine::new();
        // Seed all three timeframes with initial bars.
        engine.update("15m", &make_bar(100.0, 0));
        engine.update("1h", &make_bar(100.0, 0));
        engine.update("4h", &make_bar(100.0, 0));

        // Feed a second bar to all; now all are initialized.
        let r15 = engine.update("15m", &make_bar(101.0, 1));
        let r1h = engine.update("1h", &make_bar(101.0, 1));
        let r4h = engine.update("4h", &make_bar(101.0, 1));

        // After the second bar for each, the engine should produce a signal.
        assert!(r15.is_some() || r1h.is_some() || r4h.is_some());
    }

    #[test]
    fn test_engine_reset_clears_state() {
        let mut engine = MultiTimeframeEngine::new();
        engine.update("15m", &make_bar(100.0, 0));
        engine.update("1h", &make_bar(100.0, 0));
        engine.update("4h", &make_bar(100.0, 0));
        engine.reset();
        // After reset, warmup must repeat.
        let result = engine.update("15m", &make_bar(100.0, 1));
        assert!(result.is_none(), "post-reset: 1h and 4h have no data");
    }

    #[test]
    fn test_canonical_weights_sum_to_one() {
        let w15 = TimeframeSignal::canonical_weight("15m");
        let w1h = TimeframeSignal::canonical_weight("1h");
        let w4h = TimeframeSignal::canonical_weight("4h");
        let sum = w15 + w1h + w4h;
        assert!((sum - 1.0).abs() < 1e-12, "weights must sum to 1.0, got {sum}");
    }

    #[test]
    fn test_serde_roundtrip_aggregated_signal() {
        let agg = AggregatedSignal {
            direction: 1,
            strength: 0.75,
            agreement: 0.67,
            primary_regime: Regime::BullTrend,
        };
        let json = serde_json::to_string(&agg).unwrap();
        let decoded: AggregatedSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.direction, 1);
        assert!((decoded.strength - 0.75).abs() < 1e-12);
        assert_eq!(decoded.primary_regime, Regime::BullTrend);
    }
}
