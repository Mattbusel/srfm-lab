/// Kyle's Lambda -- market impact estimation.
///
/// Lambda is estimated via OLS regression of price changes on signed order flow:
///
///   delta_p_t = lambda * x_t + epsilon_t
///
/// where x_t is signed volume (positive = net buy, negative = net sell).
/// A higher lambda means larger price impact per unit of order flow -- i.e. lower
/// liquidity.
///
/// Two variants are provided:
///   1. Plain OLS over the last N trades.
///   2. EWMA-weighted OLS with configurable half-life (exponential decay on older obs).
///
/// References:
///   Kyle, A.S. (1985). "Continuous Auctions and Insider Trading."
///   Econometrica 53(6): 1315-1335.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Trade record
// ---------------------------------------------------------------------------

/// A single trade observation for lambda estimation.
#[derive(Debug, Clone, Copy)]
pub struct Trade {
    /// Trade price.
    pub price: f64,
    /// Signed volume: positive = buyer-initiated, negative = seller-initiated.
    pub signed_volume: f64,
    /// Unix timestamp in milliseconds (informational; not used in regression).
    pub timestamp_ms: u64,
}

impl Trade {
    pub fn new(price: f64, signed_volume: f64) -> Self {
        Trade { price, signed_volume, timestamp_ms: 0 }
    }

    pub fn with_ts(price: f64, signed_volume: f64, timestamp_ms: u64) -> Self {
        Trade { price, signed_volume, timestamp_ms }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Kyle Lambda estimator.
#[derive(Debug, Clone)]
pub struct KyleLambdaConfig {
    /// Rolling window size (number of trades).
    pub window_size: usize,
    /// Minimum observations required before emitting a lambda estimate.
    pub min_obs: usize,
    /// EWMA half-life in units of trades (for the weighted variant).
    /// Decay weight for trade i steps back = exp(-ln(2) * i / half_life).
    pub ewma_half_life: f64,
    /// If true, use EWMA weighting; if false, use equal weights (plain OLS).
    pub use_ewma: bool,
}

impl Default for KyleLambdaConfig {
    fn default() -> Self {
        KyleLambdaConfig {
            window_size: 100,
            min_obs: 10,
            ewma_half_life: 50.0,
            use_ewma: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Estimator
// ---------------------------------------------------------------------------

/// Kyle Lambda market-impact estimator.
///
/// Maintains a rolling buffer of recent trades and fits a regression each time
/// `push_trade` is called (or lazily via `lambda()`).
pub struct KyleLambdaEstimator {
    config: KyleLambdaConfig,
    /// Circular buffer of (delta_price, signed_volume) pairs.
    /// delta_price is computed as price[t] - price[t-1].
    buffer: VecDeque<(f64, f64)>, // (dp, x)
    /// Last observed price (to compute delta_price for next trade).
    last_price: Option<f64>,
    /// Cached lambda (invalidated on each new push).
    cached_lambda: Option<f64>,
    /// Count of all trades ever pushed.
    pub trade_count: u64,
}

impl KyleLambdaEstimator {
    /// Create a new estimator with given config.
    pub fn new(config: KyleLambdaConfig) -> Self {
        KyleLambdaEstimator {
            config,
            buffer: VecDeque::new(),
            last_price: None,
            cached_lambda: None,
            trade_count: 0,
        }
    }

    /// Create with default config.
    pub fn default_config() -> Self {
        Self::new(KyleLambdaConfig::default())
    }

    /// Push a new trade.
    ///
    /// Returns the updated lambda estimate once enough observations exist,
    /// otherwise `None`.
    pub fn push_trade(&mut self, price: f64, signed_volume: f64) -> Option<f64> {
        self.push_trade_ts(price, signed_volume, 0)
    }

    /// Push a new trade with explicit timestamp.
    pub fn push_trade_ts(
        &mut self,
        price: f64,
        signed_volume: f64,
        _timestamp_ms: u64,
    ) -> Option<f64> {
        self.trade_count += 1;
        self.cached_lambda = None;

        if let Some(prev) = self.last_price {
            let dp = price - prev;
            self.buffer.push_back((dp, signed_volume));
            if self.buffer.len() > self.config.window_size {
                self.buffer.pop_front();
            }
        }
        self.last_price = Some(price);

        if self.buffer.len() >= self.config.min_obs {
            Some(self.compute_lambda())
        } else {
            None
        }
    }

    /// Current lambda estimate.
    ///
    /// Returns the cached value if available, otherwise recomputes.
    /// Returns 0.0 if fewer than `min_obs` observations are available.
    pub fn lambda(&mut self) -> f64 {
        if let Some(l) = self.cached_lambda {
            return l;
        }
        if self.buffer.len() < self.config.min_obs {
            return 0.0;
        }
        let l = self.compute_lambda();
        self.cached_lambda = Some(l);
        l
    }

    /// Expected price impact for a given order size.
    ///
    /// impact = lambda * order_size
    ///
    /// A positive order_size corresponds to a buy order.
    /// Returns a signed price change estimate.
    pub fn impact_estimate(&mut self, order_size: f64) -> f64 {
        self.lambda() * order_size
    }

    /// Number of (dp, x) pairs currently in the regression buffer.
    pub fn obs_count(&self) -> usize {
        self.buffer.len()
    }

    // -----------------------------------------------------------------------
    // Internal regression
    // -----------------------------------------------------------------------

    fn compute_lambda(&self) -> f64 {
        if self.config.use_ewma {
            self.ols_ewma()
        } else {
            self.ols_equal()
        }
    }

    /// Plain OLS: lambda = cov(dp, x) / var(x)
    fn ols_equal(&self) -> f64 {
        let n = self.buffer.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = self.buffer.iter().map(|&(_, x)| x).sum::<f64>() / n;
        let mean_dp = self.buffer.iter().map(|&(dp, _)| dp).sum::<f64>() / n;

        let cov_num = self.buffer.iter().map(|&(dp, x)| (x - mean_x) * (dp - mean_dp)).sum::<f64>();
        let var_x = self.buffer.iter().map(|&(_, x)| (x - mean_x).powi(2)).sum::<f64>();

        if var_x.abs() < 1e-30 {
            return 0.0;
        }
        (cov_num / var_x).max(0.0) // Lambda must be non-negative
    }

    /// EWMA-weighted OLS.
    /// Weights decay exponentially: w_i = exp(-ln(2) * (n-1-i) / half_life)
    /// for i = 0..n (oldest..newest).
    fn ols_ewma(&self) -> f64 {
        let n = self.buffer.len();
        if n < 2 {
            return 0.0;
        }

        let decay = std::f64::consts::LN_2 / self.config.ewma_half_life;
        let weights: Vec<f64> = (0..n)
            .map(|i| (-(((n - 1 - i) as f64) * decay)).exp())
            .collect();

        let w_sum: f64 = weights.iter().sum();
        if w_sum < 1e-30 {
            return 0.0;
        }

        let pairs: Vec<(f64, f64)> = self.buffer.iter().copied().collect();

        let wmean_x = pairs.iter().zip(&weights).map(|(&(_, x), &w)| w * x).sum::<f64>() / w_sum;
        let wmean_dp =
            pairs.iter().zip(&weights).map(|(&(dp, _), &w)| w * dp).sum::<f64>() / w_sum;

        let cov_num = pairs
            .iter()
            .zip(&weights)
            .map(|(&(dp, x), &w)| w * (x - wmean_x) * (dp - wmean_dp))
            .sum::<f64>();

        let var_x = pairs
            .iter()
            .zip(&weights)
            .map(|(&(_, x), &w)| w * (x - wmean_x).powi(2))
            .sum::<f64>();

        if var_x.abs() < 1e-30 {
            return 0.0;
        }

        (cov_num / var_x).max(0.0)
    }

    /// R-squared of the most recent OLS fit (equal-weight).
    /// Returns 0.0 if insufficient data.
    pub fn r_squared(&self) -> f64 {
        let n = self.buffer.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let lambda = if self.config.use_ewma { self.ols_ewma() } else { self.ols_equal() };
        let mean_dp = self.buffer.iter().map(|&(dp, _)| dp).sum::<f64>() / n;
        let ss_tot = self.buffer.iter().map(|&(dp, _)| (dp - mean_dp).powi(2)).sum::<f64>();
        if ss_tot < 1e-30 {
            return 0.0;
        }
        let ss_res = self.buffer.iter().map(|&(dp, x)| (dp - lambda * x).powi(2)).sum::<f64>();
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    }
}

impl Default for KyleLambdaEstimator {
    fn default() -> Self {
        Self::default_config()
    }
}

// ---------------------------------------------------------------------------
// Free function helper
// ---------------------------------------------------------------------------

/// Estimate Kyle's lambda from a slice of `(price, signed_volume)` pairs.
///
/// Returns the final lambda value, or 0.0 if insufficient data.
pub fn estimate_kyle_lambda(trades: &[(f64, f64)], ewma: bool) -> f64 {
    let mut cfg = KyleLambdaConfig::default();
    cfg.use_ewma = ewma;
    let mut est = KyleLambdaEstimator::new(cfg);
    let mut last = 0.0;
    for &(p, x) in trades {
        if let Some(l) = est.push_trade(p, x) {
            last = l;
        }
    }
    last
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic trade series where price moves in direct proportion to
    /// signed volume (true lambda = expected_lambda), plus small noise.
    fn synthetic_trades(n: usize, lambda: f64, noise_scale: f64) -> Vec<(f64, f64)> {
        let mut trades = Vec::with_capacity(n);
        let mut price = 100.0f64;
        // Simple LCG for reproducibility
        let mut rng: u64 = 0xDEAD_BEEF;
        let lcg = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (*s >> 33) as f64 / (u32::MAX as f64)
        };

        for _ in 0..n {
            let x = lcg(&mut rng) * 200.0 - 100.0; // signed volume in [-100, 100]
            let noise = (lcg(&mut rng) - 0.5) * noise_scale;
            price += lambda * x + noise;
            trades.push((price, x));
        }
        trades
    }

    #[test]
    fn test_kyle_lambda_positive() {
        let trades = synthetic_trades(200, 0.05, 0.001);
        let lambda = estimate_kyle_lambda(&trades, false);
        assert!(lambda > 0.0, "lambda should be positive, got {}", lambda);
    }

    #[test]
    fn test_kyle_lambda_positive_ewma() {
        let trades = synthetic_trades(200, 0.05, 0.001);
        let lambda = estimate_kyle_lambda(&trades, true);
        assert!(lambda > 0.0, "EWMA lambda should be positive, got {}", lambda);
    }

    #[test]
    fn test_push_returns_none_until_min_obs() {
        let mut est = KyleLambdaEstimator::default();
        let mut price = 100.0;
        // Push min_obs - 1 trades (first trade does not emit dp, so need min_obs+1 pushes)
        for i in 0..9 {
            price += 0.01;
            let result = est.push_trade(price, if i % 2 == 0 { 10.0 } else { -10.0 });
            assert!(result.is_none(), "should not have enough data yet at trade {}", i);
        }
    }

    #[test]
    fn test_impact_estimate_proportional() {
        let trades = synthetic_trades(300, 0.10, 0.0001);
        let mut est = KyleLambdaEstimator::new(KyleLambdaConfig {
            use_ewma: false,
            min_obs: 20,
            ..Default::default()
        });
        for &(p, x) in &trades {
            est.push_trade(p, x);
        }
        let impact_100 = est.impact_estimate(100.0);
        let impact_200 = est.impact_estimate(200.0);
        // Impact should double when order size doubles
        assert!(
            (impact_200 - 2.0 * impact_100).abs() < 1e-9,
            "impact should be proportional: 2*{} != {}",
            impact_100,
            impact_200
        );
    }

    #[test]
    fn test_window_size_bounded() {
        let mut est = KyleLambdaEstimator::default();
        for i in 0..500 {
            est.push_trade(100.0 + i as f64 * 0.001, 10.0);
        }
        assert!(est.obs_count() <= 100, "buffer exceeded window_size");
    }

    #[test]
    fn test_lambda_zero_for_no_order_flow_variance() {
        // Constant signed volume => zero variance in x => lambda = 0
        let mut est = KyleLambdaEstimator::new(KyleLambdaConfig {
            use_ewma: false,
            min_obs: 5,
            ..Default::default()
        });
        let mut price = 100.0;
        for _ in 0..50 {
            price += 0.01;
            est.push_trade(price, 100.0); // constant signed volume
        }
        let l = est.lambda();
        assert!(l.abs() < 1e-6, "constant signed volume -> zero lambda, got {}", l);
    }

    #[test]
    fn test_r_squared_in_range() {
        let trades = synthetic_trades(200, 0.05, 0.001);
        let mut est = KyleLambdaEstimator::default();
        for &(p, x) in &trades {
            est.push_trade(p, x);
        }
        let r2 = est.r_squared();
        assert!(r2 >= 0.0 && r2 <= 1.0, "R2 out of [0,1]: {}", r2);
    }

    #[test]
    fn test_ewma_weights_sum_nonzero() {
        // Verify EWMA produces a valid lambda when data has variance
        let trades = synthetic_trades(150, 0.03, 0.0005);
        let mut est = KyleLambdaEstimator::new(KyleLambdaConfig {
            use_ewma: true,
            min_obs: 15,
            ewma_half_life: 30.0,
            window_size: 100,
        });
        let mut last: Option<f64> = None;
        for &(p, x) in &trades {
            if let Some(l) = est.push_trade(p, x) {
                last = Some(l);
            }
        }
        assert!(last.is_some(), "EWMA should produce a lambda estimate");
        assert!(last.unwrap() >= 0.0);
    }
}
