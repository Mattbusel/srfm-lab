/// market_regime_signals.rs
/// ========================
/// Microstructure-based regime signals for position sizing and execution.
///
/// Each signal is a scalar in [0, 1] unless otherwise stated.
/// Signals close to 1 indicate stress / toxicity / illiquidity and should
/// reduce position size or trigger defensive execution.

// ---------------------------------------------------------------------------
// Microstructure regime enum
// ---------------------------------------------------------------------------

/// Categorical microstructure regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicroRegime {
    /// Conditions are normal -- no unusual stress detected.
    Normal,
    /// High informed trading activity (VPIN elevated, OFI extreme).
    HighToxicity,
    /// Wide spreads and thin depth -- reduce sizing and avoid market orders.
    Illiquid,
    /// Spreads are elevated and volatile -- execution costs unpredictable.
    VolatileSpread,
    /// Tight spreads, balanced flow, low VPIN -- optimal execution window.
    Optimal,
}

// ---------------------------------------------------------------------------
// MicrostructureContext -- input data container
// ---------------------------------------------------------------------------

/// Input bundle for microstructure signal computation.
#[derive(Debug, Clone)]
pub struct MicrostructureContext {
    /// VPIN estimate in [0, 1].
    pub vpin: f64,
    /// Current effective spread in basis points.
    pub spread_bps: f64,
    /// Order Flow Imbalance z-score (signed, typically in [-4, 4]).
    pub ofi_zscore: f64,
    /// Kyle lambda (price impact coefficient), in bps per unit volume.
    pub kyle_lambda: f64,
    /// Recent mid-price volatility in bps (e.g. rolling 5-min std).
    pub recent_vol_bps: f64,
}

// ---------------------------------------------------------------------------
// MicrostructureRegimeSignal
// ---------------------------------------------------------------------------

/// Computes microstructure-based trading signals from a `MicrostructureContext`.
///
/// All signal methods return a value in [0, 1] where higher values indicate
/// more stressed / toxic / illiquid conditions.  A value of 1 means maximal
/// stress and 0 means no stress.
///
/// The object is stateless -- all computation is derived from the context
/// supplied to `new`.
#[derive(Debug, Clone)]
pub struct MicrostructureRegimeSignal {
    ctx: MicrostructureContext,
    /// Historical VPIN percentile baseline (90th percentile of observed VPIN).
    vpin_p90: f64,
    /// Baseline normal spread in bps (used to detect stress).
    normal_spread_bps: f64,
    /// Baseline Kyle lambda (used to z-score the current value).
    baseline_kyle_lambda: f64,
}

impl MicrostructureRegimeSignal {
    /// Create a new signal computer.
    ///
    /// `vpin_p90`             : 90th-percentile VPIN from recent history.
    ///                          Used to compute the informed-trading signal.
    /// `normal_spread_bps`    : typical / median spread.  Signals stress when
    ///                          current spread exceeds 2x this value.
    /// `baseline_kyle_lambda` : baseline Kyle lambda for z-scoring.
    pub fn new(
        ctx: MicrostructureContext,
        vpin_p90: f64,
        normal_spread_bps: f64,
        baseline_kyle_lambda: f64,
    ) -> Self {
        Self {
            ctx,
            vpin_p90: vpin_p90.max(1e-10),
            normal_spread_bps: normal_spread_bps.max(1e-10),
            baseline_kyle_lambda: baseline_kyle_lambda.max(1e-10),
        }
    }

    /// Update the underlying context (allows re-use without reallocation).
    pub fn update_context(&mut self, ctx: MicrostructureContext) {
        self.ctx = ctx;
    }

    // -----------------------------------------------------------------------
    // Individual signals
    // -----------------------------------------------------------------------

    /// Informed-trading signal based on VPIN percentile.
    ///
    /// Returns the current VPIN normalised by the 90th-percentile baseline.
    /// Clamped to [0, 1].  High values indicate elevated informed order flow
    /// which typically warrants reducing position size.
    pub fn informed_trading_signal(&self) -> f64 {
        (self.ctx.vpin / self.vpin_p90).clamp(0.0, 1.0)
    }

    /// Liquidity stress signal based on spread width.
    ///
    /// Returns 0 when spread is at or below the normal level.
    /// Returns 1 when spread is >= 3x the normal spread.
    /// Linearly interpolated between normal and 3x normal.
    pub fn liquidity_stress_signal(&self) -> f64 {
        let ratio = self.ctx.spread_bps / self.normal_spread_bps;
        // Below 1x: no stress.  At 3x: maximum stress.
        ((ratio - 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Flow toxicity signal based on OFI z-score.
    ///
    /// Extreme imbalance (|z| >= 3) is interpreted as a potential mean-
    /// reversion setup (toxic for trend followers) or as informed-flow
    /// (toxic for liquidity providers).  The signal increases with |z|.
    ///
    /// Returns 0 for |z| <= 1, ramps to 1 at |z| = 3.
    pub fn flow_toxicity_signal(&self) -> f64 {
        let abs_z = self.ctx.ofi_zscore.abs();
        ((abs_z - 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Market impact signal based on Kyle lambda.
    ///
    /// A rising Kyle lambda indicates the market is thinning (each unit of
    /// volume moves prices more).  Signal = current_lambda / baseline_lambda,
    /// clamped to [0, 1].
    pub fn market_impact_signal(&self) -> f64 {
        (self.ctx.kyle_lambda / self.baseline_kyle_lambda).clamp(0.0, 1.0)
    }

    /// Composite microstructure stress signal.
    ///
    /// Weighted sum of all four component signals.
    ///
    /// Weights (sum to 1):
    ///   informed_trading : 0.30
    ///   liquidity_stress : 0.30
    ///   flow_toxicity    : 0.20
    ///   market_impact    : 0.20
    pub fn composite_microstructure_signal(&self) -> f64 {
        let w_informed = 0.30;
        let w_liquidity = 0.30;
        let w_flow = 0.20;
        let w_impact = 0.20;

        let composite = w_informed * self.informed_trading_signal()
            + w_liquidity * self.liquidity_stress_signal()
            + w_flow * self.flow_toxicity_signal()
            + w_impact * self.market_impact_signal();

        composite.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// classify_microstructure_regime
// ---------------------------------------------------------------------------

/// Classify the current microstructure state into a `MicroRegime` category.
///
/// Decision rules (priority order):
///  1. HighToxicity  : VPIN > 0.7 OR |OFI z| > 2.5
///  2. Illiquid      : spread_bps > 5 * normal_spread_bps (pass 1.0 as normal for raw test)
///                     OR kyle_lambda > 2 * baseline
///  3. VolatileSpread: recent_vol_bps > 20 AND spread_bps > 2 * normal (approximated via input)
///  4. Optimal       : VPIN < 0.3 AND spread is tight AND OFI balanced
///  5. Normal        : otherwise
///
/// This function uses only the `MicrostructureContext` fields and fixed
/// thresholds -- callers should normalise inputs to the conventions described
/// in `MicrostructureContext`.
pub fn classify_microstructure_regime(ctx: &MicrostructureContext) -> MicroRegime {
    let vpin = ctx.vpin;
    let ofi_abs = ctx.ofi_zscore.abs();
    let spread = ctx.spread_bps;
    let lambda = ctx.kyle_lambda;
    let vol = ctx.recent_vol_bps;

    // 1. High toxicity: strongly informed flow.
    if vpin > 0.7 || ofi_abs > 2.5 {
        return MicroRegime::HighToxicity;
    }

    // 2. Illiquid: wide spread or large price impact.
    //    Thresholds: spread > 20 bps (absolute rough threshold for many instruments)
    //    or kyle_lambda > 10 (normalised units where 1.0 = baseline).
    if spread > 20.0 || lambda > 10.0 {
        return MicroRegime::Illiquid;
    }

    // 3. Volatile spread: moderate spread widening combined with elevated vol.
    if spread > 8.0 && vol > 15.0 {
        return MicroRegime::VolatileSpread;
    }

    // 4. Optimal: calm, balanced, tight conditions.
    if vpin < 0.3 && spread < 3.0 && ofi_abs < 0.5 && vol < 5.0 {
        return MicroRegime::Optimal;
    }

    MicroRegime::Normal
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(vpin: f64, spread: f64, ofi: f64, lambda: f64, vol: f64) -> MicrostructureContext {
        MicrostructureContext {
            vpin,
            spread_bps: spread,
            ofi_zscore: ofi,
            kyle_lambda: lambda,
            recent_vol_bps: vol,
        }
    }

    fn make_signal(
        vpin: f64, spread: f64, ofi: f64, lambda: f64, vol: f64,
        vpin_p90: f64, normal_spread: f64, baseline_lambda: f64,
    ) -> MicrostructureRegimeSignal {
        let ctx = make_ctx(vpin, spread, ofi, lambda, vol);
        MicrostructureRegimeSignal::new(ctx, vpin_p90, normal_spread, baseline_lambda)
    }

    // 1. Informed trading signal = 0 when VPIN is at zero
    #[test]
    fn test_informed_signal_zero_vpin() {
        let s = make_signal(0.0, 5.0, 0.0, 1.0, 3.0, 0.8, 5.0, 2.0);
        assert_eq!(s.informed_trading_signal(), 0.0);
    }

    // 2. Informed trading signal = 1 when VPIN equals or exceeds p90 baseline
    #[test]
    fn test_informed_signal_at_p90() {
        let s = make_signal(0.8, 5.0, 0.0, 1.0, 3.0, 0.8, 5.0, 2.0);
        assert!((s.informed_trading_signal() - 1.0).abs() < 1e-10);
    }

    // 3. Liquidity stress signal = 0 at normal spread
    #[test]
    fn test_liquidity_stress_zero_at_normal() {
        let s = make_signal(0.2, 5.0, 0.0, 1.0, 3.0, 0.8, 5.0, 2.0);
        assert_eq!(s.liquidity_stress_signal(), 0.0);
    }

    // 4. Liquidity stress signal = 1 when spread is 3x normal
    #[test]
    fn test_liquidity_stress_max_at_3x() {
        let s = make_signal(0.2, 15.0, 0.0, 1.0, 3.0, 0.8, 5.0, 2.0);
        assert!((s.liquidity_stress_signal() - 1.0).abs() < 1e-10);
    }

    // 5. Flow toxicity signal = 0 for small OFI
    #[test]
    fn test_flow_toxicity_zero_for_small_ofi() {
        let s = make_signal(0.2, 5.0, 0.5, 1.0, 3.0, 0.8, 5.0, 2.0);
        assert_eq!(s.flow_toxicity_signal(), 0.0);
    }

    // 6. Flow toxicity signal = 1 for |OFI z| >= 3
    #[test]
    fn test_flow_toxicity_max_at_3sigma() {
        let s = make_signal(0.2, 5.0, 3.5, 1.0, 3.0, 0.8, 5.0, 2.0);
        assert!((s.flow_toxicity_signal() - 1.0).abs() < 1e-10);
    }

    // 7. Market impact signal clamped to 1 when lambda >= baseline
    #[test]
    fn test_market_impact_signal_clamped() {
        let s = make_signal(0.2, 5.0, 0.0, 4.0, 3.0, 0.8, 5.0, 2.0);
        assert_eq!(s.market_impact_signal(), 1.0);
    }

    // 8. Composite signal in [0, 1]
    #[test]
    fn test_composite_signal_range() {
        for (vpin, spread, ofi, lambda) in [
            (0.0, 1.0, 0.0, 0.5),
            (1.0, 20.0, 4.0, 5.0),
            (0.5, 8.0, 2.0, 2.0),
        ] {
            let s = make_signal(vpin, spread, ofi, lambda, 5.0, 0.8, 5.0, 2.0);
            let c = s.composite_microstructure_signal();
            assert!(c >= 0.0 && c <= 1.0, "composite out of range: {c}");
        }
    }

    // 9. Composite signal weights sum consistently
    #[test]
    fn test_composite_max_is_1() {
        // All component signals at 1.0.
        let s = make_signal(0.8, 15.0, 4.0, 4.0, 5.0, 0.8, 5.0, 2.0);
        let c = s.composite_microstructure_signal();
        assert!((c - 1.0).abs() < 1e-10, "all components at max should give 1.0, got {c}");
    }

    // 10. Regime: HighToxicity from elevated VPIN
    #[test]
    fn test_regime_high_toxicity_vpin() {
        let ctx = make_ctx(0.85, 5.0, 0.0, 1.0, 3.0);
        assert_eq!(classify_microstructure_regime(&ctx), MicroRegime::HighToxicity);
    }

    // 11. Regime: HighToxicity from extreme OFI
    #[test]
    fn test_regime_high_toxicity_ofi() {
        let ctx = make_ctx(0.3, 5.0, -3.0, 1.0, 3.0);
        assert_eq!(classify_microstructure_regime(&ctx), MicroRegime::HighToxicity);
    }

    // 12. Regime: Illiquid from wide spread
    #[test]
    fn test_regime_illiquid_wide_spread() {
        let ctx = make_ctx(0.2, 25.0, 0.5, 1.0, 5.0);
        assert_eq!(classify_microstructure_regime(&ctx), MicroRegime::Illiquid);
    }

    // 13. Regime: Optimal under calm conditions
    #[test]
    fn test_regime_optimal() {
        let ctx = make_ctx(0.1, 2.0, 0.1, 1.0, 2.0);
        assert_eq!(classify_microstructure_regime(&ctx), MicroRegime::Optimal);
    }

    // 14. Regime: Normal for moderate conditions
    #[test]
    fn test_regime_normal() {
        let ctx = make_ctx(0.4, 6.0, 1.0, 2.0, 8.0);
        assert_eq!(classify_microstructure_regime(&ctx), MicroRegime::Normal);
    }

    // 15. Regime: VolatileSpread
    #[test]
    fn test_regime_volatile_spread() {
        let ctx = make_ctx(0.3, 10.0, 0.5, 1.0, 20.0);
        assert_eq!(classify_microstructure_regime(&ctx), MicroRegime::VolatileSpread);
    }
}
