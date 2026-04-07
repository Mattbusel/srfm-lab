// analytics/microstructure.rs -- Real-time microstructure analytics.
//
// Implements: VPIN, Kyle lambda, Amihud illiquidity, Corwin-Schultz spread,
// and a composite Toxicity Meter.

use std::collections::VecDeque;
use crate::market_microstructure::{TickSide, tick_rule};

// ─── VPINEstimator ────────────────────────────────────────────────────────────

/// Volume-Synchronized Probability of Informed Trading (VPIN).
///
/// Algorithm (Easley, Lopez de Prado, O'Hara 2012):
///   1. Set bucket_size = total_volume / num_buckets.
///   2. Fill buckets; within each bucket classify buy/sell via tick rule.
///   3. VPIN = (1/50) * sum_buckets |V_buy - V_sell| / bucket_size
///
/// VPIN in [0, 1]; values near 1 indicate toxic order flow.
#[derive(Debug)]
pub struct VPINEstimator {
    num_buckets: usize,
    bucket_size: f64,
    /// Completed buckets: (buy_vol, sell_vol).
    buckets: VecDeque<(f64, f64)>,
    /// Current filling bucket.
    current_buy: f64,
    current_sell: f64,
    current_volume: f64,
    /// Total volume seen so far (for initial bucket_size calibration).
    total_volume: f64,
    num_volume_obs: u64,
    prev_price: f64,
    prev_side: TickSide,
}

impl VPINEstimator {
    /// Create a new VPIN estimator.
    ///
    /// - `bucket_size` -- volume per bucket; use total_daily_volume / 50 as a guide.
    ///   Pass 0.0 to auto-calibrate from first 100 ticks.
    pub fn new(bucket_size: f64) -> Self {
        Self {
            num_buckets: 50,
            bucket_size: if bucket_size > 0.0 { bucket_size } else { 0.0 },
            buckets: VecDeque::with_capacity(51),
            current_buy: 0.0,
            current_sell: 0.0,
            current_volume: 0.0,
            total_volume: 0.0,
            num_volume_obs: 0,
            prev_price: f64::NAN,
            prev_side: TickSide::Unknown,
        }
    }

    /// Feed a trade tick. Price and volume are used for tick rule classification.
    pub fn update(&mut self, price: f64, volume: f64) {
        self.total_volume += volume;
        self.num_volume_obs += 1;

        // Auto-calibrate bucket_size after 100 ticks
        if self.bucket_size == 0.0 && self.num_volume_obs == 100 {
            self.bucket_size = self.total_volume / self.num_buckets as f64;
        }
        if self.bucket_size == 0.0 {
            return; // still calibrating
        }

        // Classify via tick rule
        let side = if self.prev_price.is_nan() {
            TickSide::Unknown
        } else {
            tick_rule(price, self.prev_price, self.prev_side)
        };
        self.prev_price = price;
        self.prev_side = side;

        // Distribute volume to buy/sell
        let (buy_vol, sell_vol) = match side {
            TickSide::Buy => (volume, 0.0),
            TickSide::Sell => (0.0, volume),
            TickSide::Unknown => (volume * 0.5, volume * 0.5),
        };

        self.current_buy += buy_vol;
        self.current_sell += sell_vol;
        self.current_volume += volume;

        // Check if bucket is full
        while self.current_volume >= self.bucket_size {
            let excess = self.current_volume - self.bucket_size;
            let fill_ratio = (self.bucket_size - (self.current_volume - volume).max(0.0))
                / volume.max(1e-12);
            let fill_ratio = fill_ratio.min(1.0).max(0.0);

            let bucket_buy = self.current_buy - buy_vol * (1.0 - fill_ratio);
            let bucket_sell = self.current_sell - sell_vol * (1.0 - fill_ratio);

            if self.buckets.len() == self.num_buckets {
                self.buckets.pop_front();
            }
            self.buckets.push_back((bucket_buy.max(0.0), bucket_sell.max(0.0)));

            // Start new bucket with overflow volume
            self.current_buy = buy_vol * (1.0 - fill_ratio);
            self.current_sell = sell_vol * (1.0 - fill_ratio);
            self.current_volume = excess;
        }
    }

    /// Compute VPIN over completed buckets. Returns None if < 2 buckets available.
    pub fn vpin(&self) -> Option<f64> {
        if self.buckets.len() < 2 || self.bucket_size < 1e-12 {
            return None;
        }
        let n = self.buckets.len() as f64;
        let sum_imbalance: f64 = self
            .buckets
            .iter()
            .map(|(b, s)| (b - s).abs())
            .sum::<f64>();
        Some((sum_imbalance / n / self.bucket_size).min(1.0))
    }

    pub fn num_buckets_filled(&self) -> usize {
        self.buckets.len()
    }
}

// ─── KyleLambdaEstimator ─────────────────────────────────────────────────────

/// Rolling estimate of Kyle's lambda via OLS regression of price change on
/// signed volume. Uses exponential decay weighting -- older trades count less.
///
/// lambda = Cov(dp, signed_vol) / Var(signed_vol)
///
/// Positive lambda means price impact is positive (normal market).
#[derive(Debug)]
pub struct KyleLambdaEstimator {
    window: usize,
    decay: f64,
    /// (signed_volume, dp) pairs.
    obs: VecDeque<(f64, f64)>,
    prev_price: Option<f64>,
}

impl KyleLambdaEstimator {
    /// Create a new Kyle lambda estimator.
    ///
    /// - `window` -- number of trade observations (default 100)
    /// - `decay`  -- exponential decay per observation (0.99 typical)
    pub fn new(window: usize, decay: f64) -> Self {
        Self {
            window,
            decay,
            obs: VecDeque::with_capacity(window + 1),
            prev_price: None,
        }
    }

    /// Feed a trade. `signed_volume` is positive for buys, negative for sells.
    pub fn update(&mut self, price: f64, signed_volume: f64) {
        if let Some(pp) = self.prev_price {
            let dp = price - pp;
            if self.obs.len() == self.window {
                self.obs.pop_front();
            }
            self.obs.push_back((signed_volume, dp));
        }
        self.prev_price = Some(price);
    }

    /// Estimate lambda via weighted OLS. Returns None if insufficient data.
    pub fn lambda(&self) -> Option<f64> {
        if self.obs.len() < 10 {
            return None;
        }
        let n = self.obs.len();
        let mut w_sum = 0.0_f64;
        let mut wx_sum = 0.0_f64;
        let mut wy_sum = 0.0_f64;
        let mut wxx_sum = 0.0_f64;
        let mut wxy_sum = 0.0_f64;

        for (i, &(x, y)) in self.obs.iter().enumerate() {
            // Newer observations get higher weight
            let age = (n - 1 - i) as f64;
            let w = self.decay.powf(age);
            w_sum += w;
            wx_sum += w * x;
            wy_sum += w * y;
            wxx_sum += w * x * x;
            wxy_sum += w * x * y;
        }

        let denom = w_sum * wxx_sum - wx_sum * wx_sum;
        if denom.abs() < 1e-15 {
            return None;
        }

        let lambda = (w_sum * wxy_sum - wx_sum * wy_sum) / denom;
        Some(lambda)
    }
}

// ─── AmihudEstimator ─────────────────────────────────────────────────────────

/// Rolling Amihud (2002) illiquidity ratio.
///
/// Illiquidity_t = |r_t| / Volume_t
/// Rolling estimate = (1/n) * sum_{i=t-n+1}^{t} illiquidity_i
///
/// Higher values indicate less liquid markets.
#[derive(Debug)]
pub struct AmihudEstimator {
    window: usize,
    buf: VecDeque<f64>,
    prev_price: Option<f64>,
}

impl AmihudEstimator {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            buf: VecDeque::with_capacity(window + 1),
            prev_price: None,
        }
    }

    /// Feed a bar close price and bar volume.
    pub fn update(&mut self, close: f64, volume: f64) {
        if let Some(pp) = self.prev_price {
            if volume > 1e-12 && pp > 1e-12 {
                let ret = ((close - pp) / pp).abs();
                let illiq = ret / volume;
                if self.buf.len() == self.window {
                    self.buf.pop_front();
                }
                self.buf.push_back(illiq);
            }
        }
        self.prev_price = Some(close);
    }

    /// Return the rolling average illiquidity. None if insufficient data.
    pub fn illiquidity(&self) -> Option<f64> {
        if self.buf.is_empty() {
            return None;
        }
        Some(self.buf.iter().sum::<f64>() / self.buf.len() as f64)
    }
}

// ─── SpreadEstimator (Corwin-Schultz) ─────────────────────────────────────────

/// Corwin-Schultz (2012) two-day high-low spread estimator.
///
/// Uses the ratio of two-day to one-day high-low ranges to identify spread
/// component in observed prices.
///
/// spread = 2*(exp(alpha) - 1) / (1 + exp(alpha))
/// where alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
/// and beta = E[ln(H_t/L_t)^2 + ln(H_{t+1}/L_{t+1})^2]
///     gamma = ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1}))^2
#[derive(Debug)]
pub struct SpreadEstimator {
    window: usize,
    /// Each entry is (high, low) for a bar.
    bar_buf: VecDeque<(f64, f64)>,
    spread_buf: VecDeque<f64>,
}

impl SpreadEstimator {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            bar_buf: VecDeque::with_capacity(window + 2),
            spread_buf: VecDeque::with_capacity(window + 1),
        }
    }

    /// Feed a bar's high and low.
    pub fn update(&mut self, high: f64, low: f64) {
        if high <= low || low <= 0.0 {
            return;
        }
        self.bar_buf.push_back((high, low));

        // Compute pairwise spread estimate for consecutive bars
        if self.bar_buf.len() >= 2 {
            let n = self.bar_buf.len();
            let (h0, l0) = self.bar_buf[n - 2];
            let (h1, l1) = self.bar_buf[n - 1];

            let beta = (h0 / l0).ln().powi(2) + (h1 / l1).ln().powi(2);
            let h_max = h0.max(h1);
            let l_min = l0.min(l1);
            if l_min <= 0.0 || h_max <= l_min {
                return;
            }
            let gamma = (h_max / l_min).ln().powi(2);

            let k = 3.0 - 2.0 * 2.0_f64.sqrt();
            if k.abs() < 1e-12 {
                return;
            }
            let alpha = (2.0_f64 * beta).sqrt() - beta.sqrt();
            let alpha = alpha / k - ((2.0 * gamma / k).sqrt()).max(0.0);

            // alpha can be negative for some bars -- clamp
            let spread_raw = 2.0 * (alpha.max(0.0).exp() - 1.0)
                / (1.0 + alpha.max(0.0).exp());

            // Convert to basis points
            let mid = (h0 + l0) / 2.0;
            let spread_bps = (spread_raw * mid * 10_000.0).max(0.0);

            if self.spread_buf.len() == self.window {
                self.spread_buf.pop_front();
            }
            self.spread_buf.push_back(spread_bps);
        }

        if self.bar_buf.len() > self.window + 2 {
            self.bar_buf.pop_front();
        }
    }

    /// Return rolling average spread in basis points. None if insufficient data.
    pub fn spread_bps(&self) -> Option<f64> {
        if self.spread_buf.is_empty() {
            return None;
        }
        Some(self.spread_buf.iter().sum::<f64>() / self.spread_buf.len() as f64)
    }
}

// ─── ToxicityMeter ────────────────────────────────────────────────────────────

/// Composite market toxicity score normalized to [0, 1].
///
/// Combines:
///   - VPIN (0-1)
///   - Kyle lambda (normalized)
///   - Order imbalance (|buy_vol - sell_vol| / total_vol)
///
/// Weights: VPIN=0.4, lambda=0.35, imbalance=0.25
#[derive(Debug)]
pub struct ToxicityMeter {
    pub vpin: VPINEstimator,
    pub kyle: KyleLambdaEstimator,
    /// Rolling buy/sell volumes for order imbalance.
    buy_vol_buf: VecDeque<f64>,
    sell_vol_buf: VecDeque<f64>,
    /// Rolling lambda observations for normalization.
    lambda_buf: VecDeque<f64>,
    window: usize,
}

impl ToxicityMeter {
    pub fn new(bucket_size: f64) -> Self {
        Self {
            vpin: VPINEstimator::new(bucket_size),
            kyle: KyleLambdaEstimator::new(100, 0.99),
            buy_vol_buf: VecDeque::with_capacity(101),
            sell_vol_buf: VecDeque::with_capacity(101),
            lambda_buf: VecDeque::with_capacity(101),
            window: 100,
        }
    }

    /// Feed a trade. `is_buy` indicates buyer aggressor.
    pub fn update(&mut self, price: f64, volume: f64, is_buy: bool) {
        let signed_vol = if is_buy { volume } else { -volume };
        self.vpin.update(price, volume);
        self.kyle.update(price, signed_vol);

        if self.buy_vol_buf.len() == self.window {
            self.buy_vol_buf.pop_front();
            self.sell_vol_buf.pop_front();
        }
        if is_buy {
            self.buy_vol_buf.push_back(volume);
            self.sell_vol_buf.push_back(0.0);
        } else {
            self.buy_vol_buf.push_back(0.0);
            self.sell_vol_buf.push_back(volume);
        }

        if let Some(lam) = self.kyle.lambda() {
            if self.lambda_buf.len() == self.window {
                self.lambda_buf.pop_front();
            }
            self.lambda_buf.push_back(lam.abs());
        }
    }

    /// Return composite toxicity in [0, 1]. None if insufficient data.
    pub fn toxicity(&self) -> Option<f64> {
        let vpin_score = self.vpin.vpin().unwrap_or(0.0);

        // Normalize kyle lambda: lambda / (max_lambda in window)
        let lambda_score = if self.lambda_buf.is_empty() {
            0.0
        } else {
            let max_lam = self.lambda_buf.iter().cloned().fold(0.0_f64, f64::max);
            if max_lam > 1e-15 {
                let current_lam = *self.lambda_buf.back().unwrap_or(&0.0);
                (current_lam / max_lam).min(1.0)
            } else {
                0.0
            }
        };

        // Order imbalance
        let total_buy: f64 = self.buy_vol_buf.iter().sum();
        let total_sell: f64 = self.sell_vol_buf.iter().sum();
        let total = total_buy + total_sell;
        let imbalance_score = if total > 1e-12 {
            (total_buy - total_sell).abs() / total
        } else {
            0.0
        };

        if self.vpin.num_buckets_filled() < 2 {
            return None;
        }

        let score = 0.4 * vpin_score + 0.35 * lambda_score + 0.25 * imbalance_score;
        Some(score.min(1.0).max(0.0))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amihud_positive() {
        let mut est = AmihudEstimator::new(22);
        let prices = [100.0, 101.0, 99.5, 102.0, 100.5];
        let vols = [1000.0, 1200.0, 800.0, 1500.0, 900.0];
        for (&p, &v) in prices.iter().zip(vols.iter()) {
            est.update(p, v);
        }
        let illiq = est.illiquidity().unwrap();
        assert!(illiq >= 0.0, "illiquidity must be non-negative");
    }

    #[test]
    fn vpin_range() {
        let mut est = VPINEstimator::new(1000.0);
        let mut price = 100.0;
        for i in 0..5000 {
            price += if i % 3 == 0 { 0.1 } else { -0.05 };
            let vol = 500.0 + (i % 10) as f64 * 50.0;
            est.update(price, vol);
        }
        if let Some(v) = est.vpin() {
            assert!(v >= 0.0 && v <= 1.0, "VPIN must be in [0,1], got {}", v);
        }
    }
}
