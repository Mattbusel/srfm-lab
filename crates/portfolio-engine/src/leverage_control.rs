// leverage_control.rs -- Dynamic leverage management
// Computes volatility-targeting leverage, regime adjustments,
// gross exposure, and margin safety checks.

/// Dynamic leverage controller.
pub struct LeverageController {
    /// Absolute maximum leverage allowed (e.g. 2.0 = 200%).
    pub max_leverage: f64,
    /// Annualized volatility target (e.g. 0.10 = 10% ann. vol).
    pub vol_target: f64,
}

impl LeverageController {
    /// Create a new controller with given targets.
    pub fn new(vol_target: f64, max_leverage: f64) -> Self {
        LeverageController {
            max_leverage,
            vol_target,
        }
    }

    // -----------------------------------------------------------------------
    // Core leverage computation
    // -----------------------------------------------------------------------

    /// Compute the volatility-targeting leverage.
    ///
    /// target_leverage = vol_target / vol
    ///
    /// Capped at max_leverage. If vol is zero or negative, returns max_leverage.
    pub fn compute_target_leverage(vol: f64, vol_target: f64, max_leverage: f64) -> f64 {
        if vol <= 0.0 {
            return max_leverage;
        }
        let raw = vol_target / vol;
        raw.min(max_leverage).max(0.0)
    }

    /// Instance method version of compute_target_leverage using stored parameters.
    pub fn target_leverage(&self, vol: f64) -> f64 {
        Self::compute_target_leverage(vol, self.vol_target, self.max_leverage)
    }

    /// Apply regime-based adjustments to a base leverage:
    ///
    /// - BH active (Bull-Hurst / momentum regime): multiply by 1.2
    /// - Mean-reverting regime (H < 0.42): multiply by 0.8
    /// - Both active simultaneously: apply BH boost first, then MR reduction
    ///   = base * 1.2 * 0.8 = base * 0.96
    ///
    /// Result is capped at max_leverage.
    pub fn regime_adjusted_leverage(
        base_leverage: f64,
        bh_active: bool,
        hurst_h: f64,
        max_leverage: f64,
    ) -> f64 {
        let mut lev = base_leverage;
        if bh_active {
            lev *= 1.2;
        }
        if hurst_h < 0.42 {
            // Mean-reverting -- reduce leverage.
            lev *= 0.8;
        }
        lev.min(max_leverage).max(0.0)
    }

    /// Instance method version of regime_adjusted_leverage.
    pub fn adjusted_leverage(&self, base_leverage: f64, bh_active: bool, hurst_h: f64) -> f64 {
        Self::regime_adjusted_leverage(base_leverage, bh_active, hurst_h, self.max_leverage)
    }

    // -----------------------------------------------------------------------
    // Gross exposure
    // -----------------------------------------------------------------------

    /// Compute gross market exposure in USD.
    ///
    /// gross_exposure = sum_i |w_i * equity|
    ///
    /// For a standard long-only portfolio weights represent fractional allocations
    /// so this equals equity * sum(|weights|).
    pub fn compute_gross_exposure(weights: &[f64], prices: &[f64], equity: f64) -> f64 {
        let _ = prices; // prices not needed when weights are dollar-fraction based
        if equity <= 0.0 {
            return 0.0;
        }
        let sum_abs: f64 = weights.iter().map(|w| w.abs()).sum();
        sum_abs * equity
    }

    /// Compute gross exposure directly from share counts and prices.
    pub fn compute_gross_exposure_shares(shares: &[f64], prices: &[f64]) -> f64 {
        assert_eq!(shares.len(), prices.len());
        shares
            .iter()
            .zip(prices.iter())
            .map(|(&s, &p)| s.abs() * p)
            .sum()
    }

    // -----------------------------------------------------------------------
    // Margin / safety checks
    // -----------------------------------------------------------------------

    /// Check whether the gross exposure is within the allowed leverage limit.
    ///
    /// Returns true (safe) if gross_exposure / equity <= max_leverage.
    pub fn margin_safety_check(gross_exposure: f64, equity: f64, max_leverage: f64) -> bool {
        if equity <= 0.0 {
            return false;
        }
        let current_leverage = gross_exposure / equity;
        current_leverage <= max_leverage
    }

    /// Compute the current leverage ratio.
    pub fn current_leverage_ratio(gross_exposure: f64, equity: f64) -> f64 {
        if equity <= 0.0 {
            return f64::INFINITY;
        }
        gross_exposure / equity
    }

    /// Return how much additional exposure can be added before hitting max_leverage.
    pub fn available_headroom(gross_exposure: f64, equity: f64, max_leverage: f64) -> f64 {
        let max_exposure = max_leverage * equity;
        (max_exposure - gross_exposure).max(0.0)
    }

    // -----------------------------------------------------------------------
    // Scale adjustments
    // -----------------------------------------------------------------------

    /// Scale a vector of weights to achieve a target leverage ratio.
    ///
    /// Returns scaled weights such that sum(|w|) == target_leverage.
    pub fn scale_weights_to_leverage(weights: &[f64], target_leverage: f64) -> Vec<f64> {
        let sum_abs: f64 = weights.iter().map(|w| w.abs()).sum();
        if sum_abs < 1e-12 {
            return weights.to_vec();
        }
        let scale = target_leverage / sum_abs;
        weights.iter().map(|&w| w * scale).collect()
    }

    /// Reduce weights proportionally to bring leverage within max_leverage.
    /// If already within limit, returns weights unchanged.
    pub fn enforce_leverage_limit(weights: &[f64], max_leverage: f64) -> Vec<f64> {
        let current: f64 = weights.iter().map(|w| w.abs()).sum();
        if current <= max_leverage + 1e-9 {
            return weights.to_vec();
        }
        let scale = max_leverage / current;
        weights.iter().map(|&w| w * scale).collect()
    }

    /// Compute net exposure (sum of weights, signed -- positive = net long).
    pub fn net_exposure(weights: &[f64]) -> f64 {
        weights.iter().sum()
    }

    /// Compute dollar-neutral deviation: |net_exposure|. Zero = perfectly dollar-neutral.
    pub fn dollar_neutral_deviation(weights: &[f64]) -> f64 {
        Self::net_exposure(weights).abs()
    }

    // -----------------------------------------------------------------------
    // Vol targeting helpers
    // -----------------------------------------------------------------------

    /// Compute the implied portfolio volatility given a leverage and instrument vol.
    pub fn implied_vol(leverage: f64, instrument_vol: f64) -> f64 {
        leverage * instrument_vol
    }

    /// Compute the number of units to trade to achieve a target notional exposure
    /// given the price per unit.
    pub fn units_for_exposure(target_exposure_usd: f64, price: f64) -> f64 {
        if price <= 0.0 {
            return 0.0;
        }
        target_exposure_usd / price
    }

    /// Compute annualized dollar P&L volatility (equity * leverage * instrument_vol).
    pub fn dollar_vol(equity: f64, leverage: f64, instrument_vol: f64) -> f64 {
        equity * leverage * instrument_vol
    }

    /// Return leverage that targets a specific dollar P&L vol per day.
    pub fn leverage_for_dollar_vol(target_dollar_vol: f64, equity: f64, instrument_vol: f64) -> f64 {
        if equity <= 0.0 || instrument_vol <= 0.0 {
            return 0.0;
        }
        target_dollar_vol / (equity * instrument_vol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vol_targeting_basic() {
        let lev = LeverageController::compute_target_leverage(0.10, 0.10, 2.0);
        assert!((lev - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vol_targeting_capped() {
        let lev = LeverageController::compute_target_leverage(0.02, 0.10, 2.0);
        assert_eq!(lev, 2.0, "Should be capped at max_leverage");
    }

    #[test]
    fn test_bh_regime_boost() {
        let base = 1.0;
        let adj = LeverageController::regime_adjusted_leverage(base, true, 0.55, 3.0);
        assert!((adj - 1.2).abs() < 1e-9);
    }

    #[test]
    fn test_mean_revert_reduction() {
        let adj = LeverageController::regime_adjusted_leverage(1.0, false, 0.35, 3.0);
        assert!((adj - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_margin_safety_check_safe() {
        assert!(LeverageController::margin_safety_check(150_000.0, 100_000.0, 2.0));
    }

    #[test]
    fn test_margin_safety_check_breach() {
        assert!(!LeverageController::margin_safety_check(250_001.0, 100_000.0, 2.5));
    }

    #[test]
    fn test_gross_exposure() {
        let weights = vec![0.4, 0.3, 0.3];
        let prices = vec![100.0, 200.0, 300.0];
        let exp = LeverageController::compute_gross_exposure(&weights, &prices, 100_000.0);
        assert!((exp - 100_000.0).abs() < 1e-6);
    }
}
