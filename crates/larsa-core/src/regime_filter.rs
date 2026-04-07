/// Regime-based signal filtering for SRFM.
///
/// Applies a hierarchy of regime rules to raw signals before order generation.
/// Prevents entries in unfavorable conditions and inverts signals in
/// mean-reverting markets.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// RegimeContext
// ---------------------------------------------------------------------------

/// The full set of regime indicators needed to evaluate all filter rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeContext {
    /// Current black-hole mass from the 4h timeframe.
    pub bh_mass: f64,
    /// Hurst exponent for the recent price series (typically 0.30 to 0.70).
    pub hurst: f64,
    /// Ratio of realized vol to historical vol (>1.0 means elevated vol).
    pub vol_ratio: f64,
    /// HMM hidden state (0=CALM, 1=TRENDING, 2=VOLATILE).
    pub hmm_state: u8,
    /// Current drawdown as a fraction (0.0 to 1.0).
    pub drawdown: f64,
}

impl RegimeContext {
    /// Construct a neutral, "all clear" context for testing.
    pub fn neutral() -> Self {
        Self {
            bh_mass: 2.0,
            hurst: 0.55,
            vol_ratio: 1.0,
            hmm_state: 0,
            drawdown: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// RegimeSignal
// ---------------------------------------------------------------------------

/// Filter decision for a signal in the current regime context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegimeSignal {
    /// Allow entry; signal direction and magnitude are unchanged.
    Enter,
    /// In a position -- do not modify it, but do not add to it.
    Hold,
    /// Close the current position.
    Exit,
    /// Block entry entirely; if in a position, allow it to expire naturally.
    Blocked,
}

impl RegimeSignal {
    pub fn label(&self) -> &'static str {
        match self {
            RegimeSignal::Enter => "ENTER",
            RegimeSignal::Hold => "HOLD",
            RegimeSignal::Exit => "EXIT",
            RegimeSignal::Blocked => "BLOCKED",
        }
    }
}

// ---------------------------------------------------------------------------
// FilterResult -- richer output that includes the possibly-modified signal
// ---------------------------------------------------------------------------

/// Full result of applying regime filters to a raw signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterResult {
    /// Filter decision.
    pub action: RegimeSignal,
    /// Possibly modified signal value (inverted, scaled, or zeroed).
    pub adjusted_signal: f64,
    /// Human-readable explanation of which rule fired.
    pub reason: String,
}

// ---------------------------------------------------------------------------
// RegimeFilter
// ---------------------------------------------------------------------------

/// Stateless regime filter -- all methods take `&self` or are `pub fn` statics.
///
/// Rules evaluated in priority order:
///   1. Drawdown > 5%  --> block entries, allow exits.
///   2. BH mass < threshold  --> block new entries.
///   3. Realized vol > 2.5x historical  --> halve position sizes.
///   4. HMM state = VOLATILE (2)  --> require |signal| > 0.7 to enter.
///   5. Hurst < 0.42  --> invert signal direction.
pub struct RegimeFilter {
    /// Minimum BH mass required for new entry (default 0.5).
    pub bh_mass_threshold: f64,
    /// Vol ratio threshold above which sizes are halved (default 2.5).
    pub vol_ratio_threshold: f64,
    /// Signal strength gate required in volatile HMM state (default 0.7).
    pub volatile_signal_gate: f64,
    /// Hurst exponent below which signal is inverted (default 0.42).
    pub hurst_invert_threshold: f64,
    /// Drawdown fraction above which entries are blocked (default 0.05).
    pub drawdown_block_threshold: f64,
}

impl RegimeFilter {
    /// Create a filter with default thresholds.
    pub fn new() -> Self {
        Self {
            bh_mass_threshold: 0.5,
            vol_ratio_threshold: 2.5,
            volatile_signal_gate: 0.7,
            hurst_invert_threshold: 0.42,
            drawdown_block_threshold: 0.05,
        }
    }

    /// Apply all regime rules to `signal` and return a `FilterResult`.
    ///
    /// `signal` is expected to be in [-1.0, 1.0] (positive = long, negative = short).
    /// `in_position` indicates whether the strategy currently holds a position
    /// (used to distinguish HOLD from BLOCKED for the drawdown rule).
    pub fn filter_with_result(
        &self,
        signal: f64,
        ctx: &RegimeContext,
        in_position: bool,
    ) -> FilterResult {
        // Rule 1: Drawdown > threshold -- block entries, allow exits.
        if ctx.drawdown > self.drawdown_block_threshold {
            if in_position {
                return FilterResult {
                    action: RegimeSignal::Exit,
                    adjusted_signal: 0.0,
                    reason: format!(
                        "drawdown={:.2}% > {:.0}% threshold: force exit",
                        ctx.drawdown * 100.0,
                        self.drawdown_block_threshold * 100.0
                    ),
                };
            } else {
                return FilterResult {
                    action: RegimeSignal::Blocked,
                    adjusted_signal: 0.0,
                    reason: format!(
                        "drawdown={:.2}% > {:.0}% threshold: block entry",
                        ctx.drawdown * 100.0,
                        self.drawdown_block_threshold * 100.0
                    ),
                };
            }
        }

        // Rule 2: BH mass too low -- block new entries.
        if ctx.bh_mass < self.bh_mass_threshold {
            return FilterResult {
                action: RegimeSignal::Blocked,
                adjusted_signal: 0.0,
                reason: format!(
                    "bh_mass={:.3} < {:.3} threshold: insufficient mass",
                    ctx.bh_mass, self.bh_mass_threshold
                ),
            };
        }

        // From here on, signal may be modified in sequence.
        let mut adjusted = signal;

        // Rule 5: Hurst < threshold -- invert signal (mean-reverting regime).
        if ctx.hurst < self.hurst_invert_threshold {
            adjusted = -adjusted;
        }

        // Rule 3: Elevated vol -- halve signal magnitude.
        if ctx.vol_ratio > self.vol_ratio_threshold {
            adjusted *= 0.5;
        }

        // Rule 4: HMM VOLATILE state -- require |signal| > gate.
        if ctx.hmm_state == 2 && adjusted.abs() <= self.volatile_signal_gate {
            return FilterResult {
                action: RegimeSignal::Blocked,
                adjusted_signal: 0.0,
                reason: format!(
                    "HMM=VOLATILE: |signal|={:.3} <= gate={:.3}",
                    adjusted.abs(),
                    self.volatile_signal_gate
                ),
            };
        }

        // Signal survives all filters.
        let action = if adjusted.abs() < 1e-9 {
            RegimeSignal::Hold
        } else {
            RegimeSignal::Enter
        };

        FilterResult {
            action,
            adjusted_signal: adjusted,
            reason: "pass".to_string(),
        }
    }

    /// Simplified filter returning only the `RegimeSignal` action.
    pub fn filter(signal: f64, ctx: &RegimeContext) -> RegimeSignal {
        let f = RegimeFilter::new();
        f.filter_with_result(signal, ctx, false).action
    }

    /// Position-size scale factor based on current regime.
    /// Returns a value in (0.0, 1.0] that should be multiplied by the
    /// desired nominal position size.
    ///
    /// - 1.0 in calm conditions.
    /// - 0.5 when vol_ratio > threshold.
    /// - 0.0 when entry is blocked or exit is required.
    pub fn size_scale(&self, ctx: &RegimeContext) -> f64 {
        let result = self.filter_with_result(0.5, ctx, false);
        match result.action {
            RegimeSignal::Blocked | RegimeSignal::Exit => 0.0,
            RegimeSignal::Hold => 0.0,
            RegimeSignal::Enter => {
                // Vol scaling was already applied to adjusted_signal; derive scale.
                if ctx.vol_ratio > self.vol_ratio_threshold {
                    0.5
                } else {
                    1.0
                }
            }
        }
    }
}

impl Default for RegimeFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_context_allows_entry() {
        let ctx = RegimeContext::neutral();
        let action = RegimeFilter::filter(0.8, &ctx);
        assert_eq!(action, RegimeSignal::Enter);
    }

    #[test]
    fn test_low_bh_mass_blocks() {
        let mut ctx = RegimeContext::neutral();
        ctx.bh_mass = 0.3; // below 0.5 threshold
        let action = RegimeFilter::filter(0.8, &ctx);
        assert_eq!(action, RegimeSignal::Blocked);
    }

    #[test]
    fn test_drawdown_above_5pct_blocks_entry() {
        let mut ctx = RegimeContext::neutral();
        ctx.drawdown = 0.06; // 6%
        let f = RegimeFilter::new();
        let result = f.filter_with_result(0.8, &ctx, false);
        assert_eq!(result.action, RegimeSignal::Blocked);
    }

    #[test]
    fn test_drawdown_above_5pct_exits_position() {
        let mut ctx = RegimeContext::neutral();
        ctx.drawdown = 0.06;
        let f = RegimeFilter::new();
        let result = f.filter_with_result(0.8, &ctx, true);
        assert_eq!(result.action, RegimeSignal::Exit);
    }

    #[test]
    fn test_hurst_below_threshold_inverts_signal() {
        let mut ctx = RegimeContext::neutral();
        ctx.hurst = 0.35; // mean-reverting
        let f = RegimeFilter::new();
        let result = f.filter_with_result(0.8, &ctx, false);
        assert_eq!(result.action, RegimeSignal::Enter);
        // Inverted signal: positive input -> negative output.
        assert!(result.adjusted_signal < 0.0, "signal must be inverted, got {}", result.adjusted_signal);
    }

    #[test]
    fn test_high_vol_halves_signal() {
        let mut ctx = RegimeContext::neutral();
        ctx.vol_ratio = 3.0; // above 2.5 threshold
        let f = RegimeFilter::new();
        let result = f.filter_with_result(0.8, &ctx, false);
        assert_eq!(result.action, RegimeSignal::Enter);
        assert!(
            (result.adjusted_signal - 0.4).abs() < 1e-9,
            "expected 0.4 (halved 0.8), got {}",
            result.adjusted_signal
        );
    }

    #[test]
    fn test_hmm_volatile_blocks_weak_signal() {
        let mut ctx = RegimeContext::neutral();
        ctx.hmm_state = 2; // VOLATILE
        let f = RegimeFilter::new();
        // Signal strength 0.5 <= gate 0.7 -- should be blocked.
        let result = f.filter_with_result(0.5, &ctx, false);
        assert_eq!(result.action, RegimeSignal::Blocked);
    }

    #[test]
    fn test_hmm_volatile_allows_strong_signal() {
        let mut ctx = RegimeContext::neutral();
        ctx.hmm_state = 2;
        let f = RegimeFilter::new();
        // Signal strength 0.9 > gate 0.7 -- should pass.
        let result = f.filter_with_result(0.9, &ctx, false);
        assert_eq!(result.action, RegimeSignal::Enter);
    }

    #[test]
    fn test_size_scale_one_in_calm() {
        let ctx = RegimeContext::neutral();
        let f = RegimeFilter::new();
        let scale = f.size_scale(&ctx);
        assert!((scale - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_size_scale_half_in_high_vol() {
        let mut ctx = RegimeContext::neutral();
        ctx.vol_ratio = 3.0;
        let f = RegimeFilter::new();
        let scale = f.size_scale(&ctx);
        assert!((scale - 0.5).abs() < 1e-9, "expected 0.5 scale, got {scale}");
    }

    #[test]
    fn test_serde_regime_context() {
        let ctx = RegimeContext::neutral();
        let json = serde_json::to_string(&ctx).unwrap();
        let decoded: RegimeContext = serde_json::from_str(&json).unwrap();
        assert!((decoded.bh_mass - 2.0).abs() < 1e-12);
        assert_eq!(decoded.hmm_state, 0);
    }

    #[test]
    fn test_filter_result_reason_nonempty() {
        let ctx = RegimeContext::neutral();
        let f = RegimeFilter::new();
        let result = f.filter_with_result(0.7, &ctx, false);
        assert!(!result.reason.is_empty());
    }
}
