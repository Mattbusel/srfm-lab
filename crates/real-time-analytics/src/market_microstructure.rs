// market_microstructure.rs — Market microstructure analytics.
//
// Implements: order flow imbalance, Kyle's λ, Amihud illiquidity,
// Roll bid-ask spread, information ratio, and the PIN model (EM).

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use anyhow::{Result, anyhow};

// ─── Tick Classification ──────────────────────────────────────────────────────

/// Direction of a trade relative to the quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TickSide {
    Buy,
    Sell,
    Unknown,
}

/// Classify a trade using the tick rule (Lee & Ready 1991).
///
/// Returns Buy if price > previous price, Sell if price < previous price,
/// and carries forward the last known side on a zero tick.
pub fn tick_rule(current_price: f64, prev_price: f64, prev_side: TickSide) -> TickSide {
    if current_price > prev_price {
        TickSide::Buy
    } else if current_price < prev_price {
        TickSide::Sell
    } else {
        prev_side // zero tick — carry forward
    }
}

// ─── OrderFlowImbalance ───────────────────────────────────────────────────────

/// Rolling order flow imbalance (OFI) tracker.
///
/// OFI = (buy_volume - sell_volume) / total_volume  ∈ [-1, 1]
///
/// Uses tick-rule classification when explicit side is unavailable.
pub struct OrderFlowImbalance {
    window: usize,
    buy_vols: VecDeque<f64>,
    sell_vols: VecDeque<f64>,
    total_buy: f64,
    total_sell: f64,
    last_price: f64,
    last_side: TickSide,
    tick_count: u64,
}

impl OrderFlowImbalance {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            buy_vols: VecDeque::with_capacity(window),
            sell_vols: VecDeque::with_capacity(window),
            total_buy: 0.0,
            total_sell: 0.0,
            last_price: 0.0,
            last_side: TickSide::Unknown,
            tick_count: 0,
        }
    }

    /// Update with a trade where side is already known.
    pub fn update_with_side(&mut self, volume: f64, side: TickSide) {
        let (buy_vol, sell_vol) = match side {
            TickSide::Buy => (volume, 0.0),
            TickSide::Sell => (0.0, volume),
            TickSide::Unknown => (volume * 0.5, volume * 0.5),
        };
        self.push(buy_vol, sell_vol);
        self.last_side = side;
        self.tick_count += 1;
    }

    /// Update using the tick rule (infer side from price movement).
    pub fn update_tick_rule(&mut self, price: f64, volume: f64) {
        let side = if self.tick_count == 0 {
            TickSide::Unknown
        } else {
            tick_rule(price, self.last_price, self.last_side)
        };
        self.last_price = price;
        self.update_with_side(volume, side);
    }

    fn push(&mut self, buy: f64, sell: f64) {
        if self.buy_vols.len() == self.window {
            self.total_buy -= self.buy_vols.pop_front().unwrap_or(0.0);
            self.total_sell -= self.sell_vols.pop_front().unwrap_or(0.0);
        }
        self.buy_vols.push_back(buy);
        self.sell_vols.push_back(sell);
        self.total_buy += buy;
        self.total_sell += sell;
    }

    /// OFI ∈ [-1, 1]. Positive = net buying pressure.
    pub fn ofi(&self) -> f64 {
        let total = self.total_buy + self.total_sell;
        if total < 1e-9 { return 0.0; }
        (self.total_buy - self.total_sell) / total
    }

    pub fn signed_flow(&self) -> f64 { self.total_buy - self.total_sell }
    pub fn buy_volume(&self) -> f64 { self.total_buy }
    pub fn sell_volume(&self) -> f64 { self.total_sell }
    pub fn total_volume(&self) -> f64 { self.total_buy + self.total_sell }
    pub fn tick_count(&self) -> u64 { self.tick_count }

    pub fn reset(&mut self) {
        self.buy_vols.clear();
        self.sell_vols.clear();
        self.total_buy = 0.0;
        self.total_sell = 0.0;
        self.tick_count = 0;
        self.last_side = TickSide::Unknown;
    }

    pub fn snapshot(&self) -> OfiSnapshot {
        OfiSnapshot {
            ofi: self.ofi(),
            buy_volume: self.total_buy,
            sell_volume: self.total_sell,
            signed_flow: self.signed_flow(),
            tick_count: self.tick_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfiSnapshot {
    pub ofi: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub signed_flow: f64,
    pub tick_count: u64,
}

// ─── KyleLambda ───────────────────────────────────────────────────────────────

/// Kyle's lambda: price impact coefficient λ from trade sequence.
///
/// Estimated via OLS of:  ΔP_t = λ · Q_t + ε_t
/// where Q_t is signed order flow and ΔP_t is the price change.
///
/// λ > 0 means price moves up with net buying (illiquidity).
pub struct KyleLambda {
    window: usize,
    price_changes: VecDeque<f64>,
    signed_flows: VecDeque<f64>,
    // Online OLS accumulators.
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_xy: f64,
    n: u64,
}

impl KyleLambda {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            price_changes: VecDeque::with_capacity(window),
            signed_flows: VecDeque::with_capacity(window),
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_xy: 0.0,
            n: 0,
        }
    }

    /// Update with a price change and the signed order flow that caused it.
    pub fn update(&mut self, price_change: f64, signed_flow: f64) {
        if self.n as usize == self.window {
            let old_x = self.signed_flows.pop_front().unwrap_or(0.0);
            let old_y = self.price_changes.pop_front().unwrap_or(0.0);
            self.sum_x -= old_x;
            self.sum_y -= old_y;
            self.sum_xx -= old_x * old_x;
            self.sum_xy -= old_x * old_y;
            self.n -= 1;
        }
        self.price_changes.push_back(price_change);
        self.signed_flows.push_back(signed_flow);
        self.sum_x += signed_flow;
        self.sum_y += price_change;
        self.sum_xx += signed_flow * signed_flow;
        self.sum_xy += signed_flow * price_change;
        self.n += 1;
    }

    /// Kyle's lambda estimate. Returns None if insufficient data.
    pub fn lambda(&self) -> Option<f64> {
        if self.n < 2 { return None; }
        let n = self.n as f64;
        let denom = self.sum_xx - self.sum_x * self.sum_x / n;
        if denom.abs() < 1e-14 { return None; }
        Some((self.sum_xy - self.sum_x * self.sum_y / n) / denom)
    }

    pub fn reset(&mut self) {
        self.price_changes.clear();
        self.signed_flows.clear();
        self.sum_x = 0.0;
        self.sum_y = 0.0;
        self.sum_xx = 0.0;
        self.sum_xy = 0.0;
        self.n = 0;
    }
}

// ─── AmihudIlliquidity ────────────────────────────────────────────────────────

/// Rolling Amihud (2002) illiquidity ratio.
///
/// ILLIQ_t = (1/N) Σ |r_t| / DollarVolume_t
///
/// Higher = more illiquid (price moves more per dollar traded).
pub struct AmihudIlliquidity {
    window: usize,
    ratios: VecDeque<f64>,
    sum: f64,
}

impl AmihudIlliquidity {
    pub fn new(window: usize) -> Self {
        Self { window, ratios: VecDeque::with_capacity(window), sum: 0.0 }
    }

    /// Update with an absolute return and the dollar volume for that period.
    pub fn update(&mut self, abs_return: f64, dollar_volume: f64) {
        if dollar_volume < 1.0 { return; } // avoid divide-by-zero
        let ratio = abs_return / dollar_volume;
        if self.ratios.len() == self.window {
            self.sum -= self.ratios.pop_front().unwrap_or(0.0);
        }
        self.ratios.push_back(ratio);
        self.sum += ratio;
    }

    /// Rolling Amihud ratio (×10^6 for readability at typical scales).
    pub fn illiquidity(&self) -> f64 {
        if self.ratios.is_empty() { return 0.0; }
        self.sum / self.ratios.len() as f64
    }

    /// Scaled by 1e6 for basis-point-like units.
    pub fn illiquidity_bps(&self) -> f64 { self.illiquidity() * 1e6 }

    pub fn reset(&mut self) { self.ratios.clear(); self.sum = 0.0; }
    pub fn count(&self) -> usize { self.ratios.len() }
}

// ─── BidAskBounce (Roll Model) ────────────────────────────────────────────────

/// Roll (1984) model: estimate effective bid-ask spread from return autocorrelation.
///
/// Roll estimator: s = 2·√(−Cov(ΔP_t, ΔP_{t-1}))
///
/// Uses an online estimator for the lag-1 autocovariance.
pub struct BidAskBounce {
    window: usize,
    returns: VecDeque<f64>,
    // Lag-1 products for autocovariance.
    lag_products: VecDeque<f64>,
    sum_r: f64,
    sum_r_lag: f64,
    sum_prod: f64,
    n: u64,
}

impl BidAskBounce {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            returns: VecDeque::with_capacity(window + 1),
            lag_products: VecDeque::with_capacity(window),
            sum_r: 0.0,
            sum_r_lag: 0.0,
            sum_prod: 0.0,
            n: 0,
        }
    }

    pub fn update(&mut self, price_return: f64) {
        if let Some(&prev) = self.returns.back() {
            let prod = price_return * prev;
            if self.lag_products.len() == self.window {
                let old_prod = self.lag_products.pop_front().unwrap_or(0.0);
                let old_r = self.returns.pop_front().unwrap_or(0.0);
                self.sum_prod -= old_prod;
                self.sum_r -= old_r;
                // The lag series lags by one so we must track separately.
                self.n -= 1;
            }
            self.lag_products.push_back(prod);
            self.sum_prod += prod;
            self.sum_r += price_return;
            self.n += 1;
        }
        self.returns.push_back(price_return);
        if self.returns.len() > self.window + 1 {
            self.returns.pop_front();
        }
    }

    /// Lag-1 autocovariance of returns.
    pub fn autocovariance(&self) -> f64 {
        if self.n < 2 { return 0.0; }
        let n = self.n as f64;
        // Use unbiased estimator: (1/(n-1)) * (Σ r_t*r_{t-1} - (1/n)*Σr_t * Σr_{t-1})
        // Since our lag products are r_t * r_{t-1}, and we track them over the window:
        self.sum_prod / n
    }

    /// Roll spread estimate. Returns None if autocovariance is non-negative (model fails).
    pub fn spread(&self) -> Option<f64> {
        let cov = self.autocovariance();
        if cov >= 0.0 { return None; } // Roll requires negative autocov
        Some(2.0 * (-cov).sqrt())
    }

    pub fn reset(&mut self) {
        self.returns.clear();
        self.lag_products.clear();
        self.sum_r = 0.0;
        self.sum_r_lag = 0.0;
        self.sum_prod = 0.0;
        self.n = 0;
    }
}

// ─── InformationRatio ─────────────────────────────────────────────────────────

/// Entropy-based information content of price moves.
///
/// Uses Shannon entropy over a discretized return distribution to measure
/// the information carried by recent price action.
///
/// H = -Σ p_i * log2(p_i)  (bits)
///
/// High entropy → unpredictable / high information content.
/// Low entropy → predictable / low surprise moves.
pub struct InformationRatio {
    window: usize,
    returns: VecDeque<f64>,
    bins: usize,
}

impl InformationRatio {
    pub fn new(window: usize, bins: usize) -> Self {
        Self { window, returns: VecDeque::with_capacity(window), bins }
    }

    pub fn update(&mut self, price_return: f64) {
        if self.returns.len() == self.window {
            self.returns.pop_front();
        }
        self.returns.push_back(price_return);
    }

    pub fn reset(&mut self) { self.returns.clear(); }

    /// Shannon entropy of the return distribution in bits.
    /// Returns None if fewer than 5 returns available.
    pub fn entropy(&self) -> Option<f64> {
        if self.returns.len() < 5 { return None; }
        let vals: Vec<f64> = self.returns.iter().cloned().collect();
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range < 1e-14 { return Some(0.0); }

        let mut counts = vec![0u64; self.bins];
        for &v in &vals {
            let bin = ((v - min) / range * (self.bins - 1) as f64).floor() as usize;
            let bin = bin.min(self.bins - 1);
            counts[bin] += 1;
        }

        let n = vals.len() as f64;
        let entropy = counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| { let p = c as f64 / n; -p * p.log2() })
            .sum::<f64>();

        Some(entropy)
    }

    /// Normalized entropy ∈ [0, 1] relative to maximum (log2(bins)).
    pub fn normalized_entropy(&self) -> Option<f64> {
        let h = self.entropy()?;
        let max_h = (self.bins as f64).log2();
        if max_h < 1e-14 { return Some(0.0); }
        Some(h / max_h)
    }

    pub fn snapshot(&self) -> InformationRatioSnapshot {
        InformationRatioSnapshot {
            entropy_bits: self.entropy(),
            normalized_entropy: self.normalized_entropy(),
            sample_count: self.returns.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationRatioSnapshot {
    pub entropy_bits: Option<f64>,
    pub normalized_entropy: Option<f64>,
    pub sample_count: usize,
}

// ─── PinEstimator (Easley et al. PIN Model) ───────────────────────────────────

/// Probability of Informed Trading (PIN) model.
///
/// Structural model of Easley, Kiefer, O'Hara, and Paperman (1996).
///
/// # Parameters
/// - α: probability of information event per day
/// - δ: probability that news is bad (given event)
/// - ε: noise (uninformed) trade arrival rate (per day, per side)
/// - μ: informed trade arrival rate (per day)
///
/// PIN = αμ / (αμ + 2ε)
///
/// Estimated via EM algorithm on daily (buy, sell) trade count vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinParams {
    /// Prob. of information event
    pub alpha: f64,
    /// Prob. bad news | event
    pub delta: f64,
    /// Noise trade rate (per side, per day)
    pub epsilon: f64,
    /// Informed trade rate (per day)
    pub mu: f64,
}

impl PinParams {
    pub fn pin(&self) -> f64 {
        let denom = self.alpha * self.mu + 2.0 * self.epsilon;
        if denom < 1e-12 { return 0.0; }
        self.alpha * self.mu / denom
    }

    pub fn default_init() -> Self {
        Self { alpha: 0.4, delta: 0.5, epsilon: 80.0, mu: 100.0 }
    }
}

impl Default for PinParams {
    fn default() -> Self { Self::default_init() }
}

/// Fit the PIN model via EM on a sequence of (buy_count, sell_count) day observations.
pub struct PinEstimator {
    observations: Vec<(f64, f64)>, // (buy_count, sell_count) per day
}

impl PinEstimator {
    pub fn new() -> Self { Self { observations: Vec::new() } }

    /// Add a daily (buy_count, sell_count) observation.
    pub fn add_day(&mut self, buys: f64, sells: f64) {
        if buys >= 0.0 && sells >= 0.0 {
            self.observations.push((buys, sells));
        }
    }

    pub fn clear(&mut self) { self.observations.clear(); }
    pub fn observation_count(&self) -> usize { self.observations.len() }

    /// Run EM to convergence. Returns fitted parameters or error.
    pub fn fit(&self, max_iter: usize, tol: f64) -> Result<PinParams> {
        if self.observations.len() < 3 {
            return Err(anyhow!("PIN EM requires at least 3 daily observations"));
        }

        let mut p = PinParams::default_init();

        for iter in 0..max_iter {
            let p_new = self.em_step(&p)?;

            // Check convergence.
            let delta_alpha = (p_new.alpha - p.alpha).abs();
            let delta_delta = (p_new.delta - p.delta).abs();
            let delta_eps   = (p_new.epsilon - p.epsilon).abs() / p.epsilon.max(1e-9);
            let delta_mu    = (p_new.mu - p.mu).abs() / p.mu.max(1e-9);

            p = p_new;

            if delta_alpha < tol && delta_delta < tol && delta_eps < tol && delta_mu < tol {
                eprintln!("PIN EM converged after {iter} iterations");
                break;
            }
        }

        Ok(p)
    }

    fn em_step(&self, p: &PinParams) -> Result<PinParams> {
        let alpha = p.alpha.clamp(1e-6, 1.0 - 1e-6);
        let delta = p.delta.clamp(1e-6, 1.0 - 1e-6);
        let epsilon = p.epsilon.max(1e-6);
        let mu = p.mu.max(1e-6);

        // Likelihoods for three latent states:
        // L0: no event
        // Lb: bad news event
        // Lg: good news event
        let mut _sum_p0 = 0.0;
        let mut sum_pb = 0.0;
        let mut sum_pg = 0.0;

        for &(b, s) in &self.observations {
            // L0: Poisson(ε) buys, Poisson(ε) sells
            let l0 = poisson_ll(b, epsilon) + poisson_ll(s, epsilon);
            // Lb: bad news → Poisson(ε) buys, Poisson(μ+ε) sells
            let lb = poisson_ll(b, epsilon) + poisson_ll(s, mu + epsilon);
            // Lg: good news → Poisson(μ+ε) buys, Poisson(ε) sells
            let lg = poisson_ll(b, mu + epsilon) + poisson_ll(s, epsilon);

            // Weight by prior (mixture coefficients).
            let w0 = (1.0 - alpha) * l0;
            let wb = alpha * delta * lb;
            let wg = alpha * (1.0 - delta) * lg;
            let total = w0 + wb + wg + 1e-300;

            _sum_p0 += w0 / total;
            sum_pb += wb / total;
            sum_pg += wg / total;
        }

        let n = self.observations.len() as f64;
        let new_alpha = ((sum_pb + sum_pg) / n).clamp(1e-6, 1.0 - 1e-6);
        let new_delta = (sum_pb / (sum_pb + sum_pg + 1e-12)).clamp(1e-6, 1.0 - 1e-6);

        // Update ε: expected noise rate (buy side in no-event + bad event, sell side in no-event + good event)
        let mut num_eps = 0.0_f64;
        let mut denom_eps = 0.0_f64;
        let mut num_mu = 0.0_f64;
        let mut denom_mu = 0.0_f64;

        for &(b, s) in &self.observations {
            let l0 = poisson_ll(b, epsilon) + poisson_ll(s, epsilon);
            let lb = poisson_ll(b, epsilon) + poisson_ll(s, mu + epsilon);
            let lg = poisson_ll(b, mu + epsilon) + poisson_ll(s, epsilon);

            let w0 = (1.0 - alpha) * l0;
            let wb = alpha * delta * lb;
            let wg = alpha * (1.0 - delta) * lg;
            let total = w0 + wb + wg + 1e-300;

            let pp0 = w0 / total;
            let ppb = wb / total;
            let ppg = wg / total;

            // ε estimated from buys under no-event + bad event, sells under no-event + good event.
            num_eps += pp0 * (b + s) + ppb * b + ppg * s;
            denom_eps += 2.0 * pp0 + ppb + ppg;

            // μ estimated from the elevated side under event days.
            num_mu += ppb * s + ppg * b;
            denom_mu += ppb + ppg;
        }

        let new_epsilon = (num_eps / denom_eps.max(1e-9)).max(1e-6);
        let new_mu = (num_mu / denom_mu.max(1e-9) - new_epsilon).max(1e-6);

        Ok(PinParams {
            alpha: new_alpha,
            delta: new_delta,
            epsilon: new_epsilon,
            mu: new_mu,
        })
    }
}

impl Default for PinEstimator {
    fn default() -> Self { Self::new() }
}

/// Evaluate Poisson probability mass function in log space.
/// Clamped to avoid underflow.
fn poisson_ll(k: f64, lambda: f64) -> f64 {
    if lambda < 1e-9 { return if k < 0.5 { 1.0 } else { 0.0 }; }
    // Poisson PMF: λ^k * e^{-λ} / k!
    // In log space: k*ln(λ) - λ - ln(k!)
    // Use Stirling for k!: ln(k!) ≈ k*ln(k) - k + 0.5*ln(2πk) for k > 1
    let log_pmf = k * lambda.ln() - lambda - log_factorial(k);
    log_pmf.exp().max(1e-300)
}

fn log_factorial(n: f64) -> f64 {
    if n <= 1.0 { return 0.0; }
    // Lanczos approximation shortcut: use Stirling for large n.
    if n > 20.0 {
        // Stirling: ln(n!) ≈ n*ln(n) - n + 0.5*ln(2πn)
        n * n.ln() - n + 0.5 * (2.0 * std::f64::consts::PI * n).ln()
    } else {
        // Direct for small n.
        let mut acc = 0.0;
        let mut i = 2.0;
        while i <= n {
            acc += i.ln();
            i += 1.0;
        }
        acc
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_rule() {
        assert_eq!(tick_rule(101.0, 100.0, TickSide::Unknown), TickSide::Buy);
        assert_eq!(tick_rule(99.0, 100.0, TickSide::Unknown), TickSide::Sell);
        assert_eq!(tick_rule(100.0, 100.0, TickSide::Buy), TickSide::Buy);
    }

    #[test]
    fn test_ofi_pure_buy() {
        let mut ofi = OrderFlowImbalance::new(100);
        for _ in 0..10 {
            ofi.update_with_side(100.0, TickSide::Buy);
        }
        assert!((ofi.ofi() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ofi_mixed() {
        let mut ofi = OrderFlowImbalance::new(100);
        for _ in 0..5 { ofi.update_with_side(100.0, TickSide::Buy); }
        for _ in 0..5 { ofi.update_with_side(100.0, TickSide::Sell); }
        assert!(ofi.ofi().abs() < 1e-10);
    }

    #[test]
    fn test_kyle_lambda_positive() {
        let mut k = KyleLambda::new(50);
        // Positive flow → positive price change → λ > 0.
        for i in 0..30 {
            let flow = (i as f64) * 10.0;
            let dp = flow * 0.001 + 0.0001;
            k.update(dp, flow);
        }
        let lambda = k.lambda().unwrap();
        assert!(lambda > 0.0, "expected positive lambda, got {lambda}");
    }

    #[test]
    fn test_amihud() {
        let mut a = AmihudIlliquidity::new(20);
        for _ in 0..20 {
            a.update(0.01, 1_000_000.0); // 1% return on $1M volume
        }
        assert!(a.illiquidity() > 0.0);
    }

    #[test]
    fn test_information_entropy() {
        let mut ir = InformationRatio::new(100, 10);
        // Uniform distribution → high entropy.
        for i in 0..100 { ir.update(i as f64 * 0.001 - 0.05); }
        let h = ir.normalized_entropy().unwrap();
        assert!(h > 0.5, "expected high entropy for uniform input, got {h}");
    }

    #[test]
    fn test_pin_em_basic() {
        let mut est = PinEstimator::new();
        // Simulate mostly uninformed days: ~100 buys, ~100 sells.
        // Low PIN expected.
        for _ in 0..30 {
            est.add_day(100.0, 100.0);
        }
        // A few informed days with sell pressure.
        for _ in 0..5 {
            est.add_day(80.0, 200.0);
        }
        let params = est.fit(100, 1e-4).unwrap();
        assert!(params.pin() > 0.0);
        assert!(params.pin() < 1.0);
        println!("PIN = {:.4}, alpha={:.3}, delta={:.3}, epsilon={:.1}, mu={:.1}",
            params.pin(), params.alpha, params.delta, params.epsilon, params.mu);
    }
}
