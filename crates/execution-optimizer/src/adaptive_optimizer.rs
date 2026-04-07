/// adaptive_optimizer.rs -- adaptive execution that responds to live market conditions.
///
/// The AdaptiveExecutionOptimizer adjusts a base execution schedule in real time
/// based on observed spread, volatility, order-flow imbalance, and alpha decay.

// ── MarketConditions ──────────────────────────────────────────────────────────

/// Snapshot of current market microstructure conditions.
#[derive(Debug, Clone)]
pub struct MarketConditions {
    /// Observed bid-ask spread in basis points.
    pub spread_bps:  f64,
    /// Intraday realized volatility in basis points.
    pub vol_bps:     f64,
    /// Order-flow imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol), in [-1, 1].
    pub ofi:         f64,
    /// Kyle lambda: price-impact per unit of signed order flow.
    pub kyle_lambda: f64,
    /// Average daily volume in shares (or contracts).
    pub adv:         f64,
}

impl MarketConditions {
    /// Construct typical liquid equity conditions.
    pub fn typical_equity() -> Self {
        MarketConditions {
            spread_bps:  5.0,
            vol_bps:     50.0,
            ofi:         0.0,
            kyle_lambda: 1e-6,
            adv:         5_000_000.0,
        }
    }
}

// ── OptConfig ─────────────────────────────────────────────────────────────────

/// Configuration for the adaptive optimizer.
#[derive(Debug, Clone)]
pub struct OptConfig {
    /// Baseline (normal) spread in basis points for the instrument.
    pub baseline_spread_bps:  f64,
    /// Baseline (normal) volatility in basis points.
    pub baseline_vol_bps:     f64,
    /// Factor by which spread must exceed baseline to trigger pause (default 3.0).
    pub spread_pause_factor:  f64,
    /// Factor by which vol must exceed baseline to trigger pause (default 4.0).
    pub vol_pause_factor:     f64,
    /// Maximum fraction of ADV per interval for dynamic sizing.
    pub max_adv_pct:          f64,
    /// Minimum fraction of ADV per interval.
    pub min_adv_pct:          f64,
}

impl Default for OptConfig {
    fn default() -> Self {
        OptConfig {
            baseline_spread_bps: 5.0,
            baseline_vol_bps:    50.0,
            spread_pause_factor: 3.0,
            vol_pause_factor:    4.0,
            max_adv_pct:         0.20,
            min_adv_pct:         0.001,
        }
    }
}

// ── AdaptiveExecutionOptimizer ────────────────────────────────────────────────

/// Adaptive optimizer that modifies a base execution schedule based on
/// observed market conditions and alpha decay urgency.
pub struct AdaptiveExecutionOptimizer {
    pub config: OptConfig,
}

impl AdaptiveExecutionOptimizer {
    pub fn new(config: OptConfig) -> Self {
        AdaptiveExecutionOptimizer { config }
    }

    /// Compute execution urgency based on elapsed bars and alpha decay.
    ///
    /// Formula: urgency = 1 - exp(-bars_elapsed * ln(2) / alpha_decay_halflife)
    ///
    /// Returns value in [0, 1].  Urgency approaches 1 as more bars elapse.
    pub fn compute_urgency(
        &self,
        _conditions:          &MarketConditions,
        alpha_decay_halflife: f64,
        bars_elapsed:         f64,
    ) -> f64 {
        assert!(
            alpha_decay_halflife > 0.0,
            "alpha_decay_halflife must be positive"
        );
        let decay_rate = std::f64::consts::LN_2 / alpha_decay_halflife;
        let urgency = 1.0 - (-bars_elapsed * decay_rate).exp();
        urgency.clamp(0.0, 1.0)
    }

    /// Adjust a base schedule given current market conditions and urgency.
    ///
    /// Strategy:
    ///   - Low spread + low vol + high urgency: shift quantity toward earlier intervals.
    ///   - High spread + high vol: shift quantity toward later intervals.
    ///
    /// The adjustment preserves the total quantity.
    pub fn adjust_schedule(
        &self,
        base_schedule: &[f64],
        conditions:    &MarketConditions,
        urgency:       f64,
    ) -> Vec<f64> {
        if base_schedule.is_empty() {
            return Vec::new();
        }

        let n = base_schedule.len();
        let total: f64 = base_schedule.iter().sum();

        if total == 0.0 {
            return base_schedule.to_vec();
        }

        // Compute market quality score: high = good conditions (cheap to trade).
        // spread_score in [0,1]: 1 = tight spread.
        let spread_score = (self.config.baseline_spread_bps
            / conditions.spread_bps.max(1e-6))
        .clamp(0.0, 1.0);

        // vol_score in [0,1]: 1 = low vol.
        let vol_score = (self.config.baseline_vol_bps
            / conditions.vol_bps.max(1e-6))
        .clamp(0.0, 1.0);

        // Market quality in [0,1].
        let market_quality = 0.5 * spread_score + 0.5 * vol_score;

        // Combined acceleration factor: positive means accelerate, negative = decelerate.
        // Ranges from about -1 (bad conditions) to +1 (good conditions + high urgency).
        let accel = market_quality * urgency - (1.0 - market_quality) * (1.0 - urgency);
        let accel = accel.clamp(-1.0, 1.0);

        // Build weight vector: front-weighted if accel > 0, back-weighted if accel < 0.
        // Weight for interval i (0-indexed): w_i = base + accel * skew_factor(i).
        let mut weights: Vec<f64> = (0..n)
            .map(|i| {
                // skew_factor: +1 for first interval, -1 for last, linear in between.
                let skew = if n == 1 {
                    0.0
                } else {
                    1.0 - 2.0 * i as f64 / (n - 1) as f64
                };
                1.0 + accel * skew
            })
            .collect();

        // Ensure no negative weights.
        for w in &mut weights {
            if *w < 0.0 { *w = 0.0; }
        }

        let weight_sum: f64 = weights.iter().sum();
        if weight_sum == 0.0 {
            // Fallback: TWAP.
            let slice = total / n as f64;
            return vec![slice; n];
        }

        // Scale weights to preserve total quantity.
        weights
            .iter()
            .zip(base_schedule.iter())
            .map(|(&w, &_b)| w / weight_sum * total)
            .collect()
    }

    /// Determine whether execution should be paused.
    ///
    /// Returns true if:
    ///   - spread > spread_pause_factor * baseline_spread_bps, OR
    ///   - vol    > vol_pause_factor    * baseline_vol_bps
    pub fn should_pause(&self, conditions: &MarketConditions) -> bool {
        let spread_threshold =
            self.config.spread_pause_factor * self.config.baseline_spread_bps;
        let vol_threshold =
            self.config.vol_pause_factor * self.config.baseline_vol_bps;

        conditions.spread_bps > spread_threshold || conditions.vol_bps > vol_threshold
    }

    /// Compute a dynamic sizing factor for individual orders.
    ///
    /// Factor is reduced when:
    ///   - Spread is wider than baseline.
    ///   - Volatility is higher than baseline.
    ///   - OFI is adverse to direction (negative for a buy order).
    ///
    /// Returns value in [min_adv_pct, max_adv_pct].
    pub fn dynamic_sizing_factor(
        conditions: &MarketConditions,
        config:     &OptConfig,
    ) -> f64 {
        let spread_ratio = config.baseline_spread_bps
            / conditions.spread_bps.max(1e-6);
        let vol_ratio = config.baseline_vol_bps
            / conditions.vol_bps.max(1e-6);

        // OFI adjustment: if OFI is positive (more buys) it's cheaper to buy.
        let ofi_adj = 1.0 + 0.2 * conditions.ofi.clamp(-1.0, 1.0);

        let raw = config.max_adv_pct
            * spread_ratio.sqrt()
            * vol_ratio.sqrt()
            * ofi_adj;

        raw.clamp(config.min_adv_pct, config.max_adv_pct)
    }
}

impl Default for AdaptiveExecutionOptimizer {
    fn default() -> Self { Self::new(OptConfig::default()) }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_optimizer() -> AdaptiveExecutionOptimizer {
        AdaptiveExecutionOptimizer::default()
    }

    fn good_conditions() -> MarketConditions {
        MarketConditions {
            spread_bps:  2.0,
            vol_bps:     20.0,
            ofi:         0.1,
            kyle_lambda: 1e-7,
            adv:         10_000_000.0,
        }
    }

    fn bad_conditions() -> MarketConditions {
        MarketConditions {
            spread_bps:  50.0,
            vol_bps:     500.0,
            ofi:         -0.5,
            kyle_lambda: 1e-5,
            adv:         1_000_000.0,
        }
    }

    #[test]
    fn test_urgency_zero_at_start() {
        let opt = default_optimizer();
        let cond = good_conditions();
        let u = opt.compute_urgency(&cond, 10.0, 0.0);
        assert!((u - 0.0).abs() < 1e-12, "urgency at t=0 should be 0, got {}", u);
    }

    #[test]
    fn test_urgency_approaches_one() {
        let opt = default_optimizer();
        let cond = good_conditions();
        let u = opt.compute_urgency(&cond, 10.0, 1000.0);
        assert!(u > 0.999, "urgency should be near 1 after many bars, got {}", u);
    }

    #[test]
    fn test_urgency_half_at_halflife() {
        let opt = default_optimizer();
        let cond = good_conditions();
        let halflife = 20.0;
        let u = opt.compute_urgency(&cond, halflife, halflife);
        assert!(
            (u - 0.5).abs() < 1e-9,
            "urgency at halflife should be 0.5, got {}",
            u
        );
    }

    #[test]
    fn test_adjust_schedule_preserves_total() {
        let opt = default_optimizer();
        let base = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let total: f64 = base.iter().sum();
        let cond = good_conditions();
        let adjusted = opt.adjust_schedule(&base, &cond, 0.8);
        let adj_total: f64 = adjusted.iter().sum();
        assert!(
            (adj_total - total).abs() < 1e-6,
            "adjusted total {} != base total {}",
            adj_total,
            total
        );
    }

    #[test]
    fn test_adjust_schedule_good_conditions_front_loaded() {
        let opt = default_optimizer();
        let base = vec![100.0; 10];
        let cond = good_conditions();
        let adjusted = opt.adjust_schedule(&base, &cond, 1.0);
        // First interval should be larger than last.
        assert!(
            adjusted[0] >= adjusted[adjusted.len() - 1],
            "good conditions + high urgency should front-load: first={} last={}",
            adjusted[0],
            adjusted[adjusted.len() - 1]
        );
    }

    #[test]
    fn test_adjust_schedule_bad_conditions_back_loaded() {
        let opt = default_optimizer();
        let base = vec![100.0; 10];
        let cond = bad_conditions();
        let adjusted = opt.adjust_schedule(&base, &cond, 0.0);
        // Last interval should be >= first.
        assert!(
            adjusted[adjusted.len() - 1] >= adjusted[0],
            "bad conditions should back-load: first={} last={}",
            adjusted[0],
            adjusted[adjusted.len() - 1]
        );
    }

    #[test]
    fn test_should_pause_bad_spread() {
        let opt = default_optimizer();
        let mut cond = MarketConditions::typical_equity();
        cond.spread_bps = opt.config.baseline_spread_bps * 4.0; // 4x baseline
        assert!(opt.should_pause(&cond), "should pause on wide spread");
    }

    #[test]
    fn test_should_pause_bad_vol() {
        let opt = default_optimizer();
        let mut cond = MarketConditions::typical_equity();
        cond.vol_bps = opt.config.baseline_vol_bps * 5.0; // 5x baseline
        assert!(opt.should_pause(&cond), "should pause on high vol");
    }

    #[test]
    fn test_should_not_pause_normal_conditions() {
        let opt = default_optimizer();
        let cond = MarketConditions::typical_equity();
        assert!(!opt.should_pause(&cond), "should not pause under normal conditions");
    }

    #[test]
    fn test_dynamic_sizing_factor_in_bounds() {
        let config = OptConfig::default();
        let cond = MarketConditions::typical_equity();
        let factor = AdaptiveExecutionOptimizer::dynamic_sizing_factor(&cond, &config);
        assert!(
            factor >= config.min_adv_pct && factor <= config.max_adv_pct,
            "factor {} out of bounds [{}, {}]",
            factor,
            config.min_adv_pct,
            config.max_adv_pct
        );
    }

    #[test]
    fn test_dynamic_sizing_factor_reduces_in_bad_conditions() {
        let config = OptConfig::default();
        let good = good_conditions();
        let bad = bad_conditions();
        let f_good = AdaptiveExecutionOptimizer::dynamic_sizing_factor(&good, &config);
        let f_bad = AdaptiveExecutionOptimizer::dynamic_sizing_factor(&bad, &config);
        assert!(
            f_good >= f_bad,
            "sizing factor should be lower in bad conditions: good={} bad={}",
            f_good,
            f_bad
        );
    }

    #[test]
    fn test_adjust_schedule_single_interval() {
        let opt = default_optimizer();
        let base = vec![500.0];
        let cond = good_conditions();
        let adjusted = opt.adjust_schedule(&base, &cond, 0.5);
        assert_eq!(adjusted.len(), 1);
        assert!((adjusted[0] - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_adjust_schedule_empty() {
        let opt = default_optimizer();
        let adjusted = opt.adjust_schedule(&[], &good_conditions(), 0.5);
        assert!(adjusted.is_empty());
    }
}
