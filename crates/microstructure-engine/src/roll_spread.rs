/// Roll (1984) bid-ask spread estimator.
///
/// Infers the effective spread from the serial covariance of price changes:
///
///   Cov(Δp_t, Δp_{t-1}) = -(S/2)²
///   ⟹  S = 2 · √( -Cov(Δp_t, Δp_{t-1}) )   if Cov < 0
///
/// An online accumulator tracks Σ Δp_t · Δp_{t-1} without storing the
/// full price series.

use crate::streaming_stats::StreamingStats;

/// Online Roll spread estimator.
///
/// Feed mid-price observations sequentially. The estimator maintains
/// a two-lag running sum to compute serial covariance incrementally.
#[derive(Debug, Clone)]
pub struct RollSpreadEstimator {
    n:        u64,
    prev_dp:  Option<f64>,  // Δp_{t-1}
    prev_p:   Option<f64>,  // p_{t-1} for computing Δp_t

    // Running moments for serial-cov calculation
    sum_x:    f64,   // Σ Δp_t
    sum_y:    f64,   // Σ Δp_{t-1}
    sum_xy:   f64,   // Σ Δp_t · Δp_{t-1}
    sum_x2:   f64,
    sum_y2:   f64,

    /// Diagnostic: track spread estimates
    spread_stats: StreamingStats,
}

impl RollSpreadEstimator {
    pub fn new() -> Self {
        Self {
            n:        0,
            prev_dp:  None,
            prev_p:   None,
            sum_x:    0.0,
            sum_y:    0.0,
            sum_xy:   0.0,
            sum_x2:   0.0,
            sum_y2:   0.0,
            spread_stats: StreamingStats::new(),
        }
    }

    /// Push the next mid-price observation.
    pub fn update(&mut self, price: f64) {
        let Some(prev_p) = self.prev_p else {
            self.prev_p = Some(price);
            return;
        };
        let dp = price - prev_p;
        self.prev_p = Some(price);

        let Some(prev_dp) = self.prev_dp else {
            self.prev_dp = Some(dp);
            return;
        };

        // Now we have (dp, prev_dp) pair
        self.n     += 1;
        self.sum_x  += dp;
        self.sum_y  += prev_dp;
        self.sum_xy += dp * prev_dp;
        self.sum_x2 += dp * dp;
        self.sum_y2 += prev_dp * prev_dp;

        self.prev_dp = Some(dp);

        if let Some(s) = self.spread() {
            self.spread_stats.update(s);
        }
    }

    /// Estimated bid-ask spread. Returns `None` if covariance is non-negative
    /// (no spread signal) or insufficient data.
    pub fn spread(&self) -> Option<f64> {
        if self.n < 5 { return None; }
        let n   = self.n as f64;
        let cov = (self.sum_xy - self.sum_x * self.sum_y / n) / (n - 1.0);
        if cov >= 0.0 { return None; }   // positive serial cov → no Roll spread signal
        Some(2.0 * (-cov).sqrt())
    }

    /// Spread as a fraction of current price (requires last price argument).
    pub fn spread_bps(&self, price: f64) -> Option<f64> {
        if price <= 0.0 { return None; }
        self.spread().map(|s| s / price * 10_000.0)
    }

    /// Serial covariance Cov(Δp_t, Δp_{t-1}).
    pub fn serial_cov(&self) -> Option<f64> {
        if self.n < 2 { return None; }
        let n = self.n as f64;
        Some((self.sum_xy - self.sum_x * self.sum_y / n) / (n - 1.0))
    }

    pub fn count(&self) -> u64 { self.n }

    pub fn spread_mean(&self) -> f64  { self.spread_stats.mean() }
    pub fn spread_std(&self)  -> f64  { self.spread_stats.std() }

    pub fn reset(&mut self) { *self = Self::new(); }
}

impl Default for RollSpreadEstimator {
    fn default() -> Self { Self::new() }
}

/// Hasbrouck (2004) Gibbs sampling spread estimator (simplified analytical version).
///
/// Uses the identity:  c² = Var(Δp) / 2 + Cov(Δp_t, Δp_{t-1})
/// where c is the half-spread.
#[derive(Debug, Clone, Default)]
pub struct HasbrouckSpread {
    roll: RollSpreadEstimator,
    var_dp: f64,
    n_var:  u64,
    prev_dp: Option<f64>,
    prev_p:  Option<f64>,
    sum_dp:  f64,
    sum_dp2: f64,
}

impl HasbrouckSpread {
    pub fn new() -> Self { Self::default() }

    pub fn update(&mut self, price: f64) {
        self.roll.update(price);
        if let Some(pp) = self.prev_p {
            let dp = price - pp;
            self.n_var   += 1;
            self.sum_dp  += dp;
            self.sum_dp2 += dp * dp;
            let n = self.n_var as f64;
            self.var_dp  = (self.sum_dp2 - self.sum_dp * self.sum_dp / n) / n.max(2.0);
        }
        self.prev_p = Some(price);
    }

    /// Half-spread estimate via Hasbrouck method.
    pub fn half_spread(&self) -> Option<f64> {
        if self.n_var < 5 { return None; }
        let cov = self.roll.serial_cov()?;
        let val = self.var_dp / 2.0 + cov;
        if val < 0.0 { Some((-val).sqrt()) } else { None }
    }

    pub fn spread(&self) -> Option<f64> { self.half_spread().map(|h| 2.0 * h) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulate_prices_with_spread(spread: f64, n: usize) -> Vec<f64> {
        // Simulate: mid + random noise, trade prices bounce at ±spread/2
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut prices = Vec::with_capacity(n);
        let mut mid = 100.0_f64;
        for i in 0..n {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            let r = (h.finish() % 1000) as f64 / 1000.0 - 0.5;
            mid += r * 0.1;
            // Alternate buy/sell trades
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            prices.push(mid + sign * spread / 2.0);
        }
        prices
    }

    #[test]
    fn roll_spread_detects_negative_cov() {
        // With alternating buy/sell, serial cov should be negative
        let prices = simulate_prices_with_spread(0.10, 200);
        let mut r = RollSpreadEstimator::new();
        for p in prices { r.update(p); }
        // Serial cov should be negative (alternating sign pattern)
        if let Some(cov) = r.serial_cov() {
            assert!(cov < 0.0, "expected negative serial cov, got {}", cov);
        }
    }

    #[test]
    fn roll_spread_zero_spread() {
        // Random walk with no spread — serial cov near zero
        let mut r = RollSpreadEstimator::new();
        let mut p = 100.0_f64;
        for i in 0..100 {
            p += (i as f64 * 7.3).sin() * 0.01;
            r.update(p);
        }
        // May or may not have a spread signal — just check no panic
        let _ = r.spread();
    }
}
