use std::collections::VecDeque;

// ─── GARCH(1,1) ──────────────────────────────────────────────────────────────

/// GARCH(1,1) state for realised-volatility estimation.
#[derive(Debug, Clone)]
pub struct GARCHState {
    /// Long-run variance (omega).
    pub omega: f64,
    /// ARCH coefficient.
    pub alpha: f64,
    /// GARCH coefficient.
    pub beta: f64,
    /// Current conditional variance estimate.
    pub variance: f64,
    /// Number of updates so far (warmup tracking).
    pub n: usize,
}

impl GARCHState {
    pub fn new(omega: f64, alpha: f64, beta: f64, initial_var: f64) -> Self {
        assert!(
            alpha + beta < 1.0,
            "GARCH stationarity: alpha+beta must be < 1.0"
        );
        Self { omega, alpha, beta, variance: initial_var, n: 0 }
    }

    /// Default parametrisation tuned for 15-min crypto bars.
    pub fn default_crypto() -> Self {
        // omega=1e-6, alpha=0.10, beta=0.88  =>  persistence 0.98
        Self::new(1e-6, 0.10, 0.88, 0.0004)
    }

    /// Update with a new log return.
    pub fn update(&mut self, ret: f64) {
        self.variance = self.omega + self.alpha * ret * ret + self.beta * self.variance;
        self.n += 1;
    }

    /// Annualised volatility (annualise from 15-min bars: sqrt(365*24*4) ≈ 187.1).
    #[inline]
    pub fn vol_scale(&self) -> f64 {
        (self.variance * 35040.0_f64).sqrt() // 35040 = 365*24*4
    }

    /// Inverse vol scale: target_vol / current_vol (position sizing multiplier).
    pub fn inv_vol_scale(&self, target_vol: f64) -> f64 {
        let v = self.vol_scale();
        if v < 1e-8 {
            1.0
        } else {
            (target_vol / v).clamp(0.1, 5.0)
        }
    }
}

// ─── OU Mean-Reversion ────────────────────────────────────────────────────────

/// Ornstein-Uhlenbeck state estimated from a rolling window of prices.
#[derive(Debug, Clone)]
pub struct OUState {
    pub prices: VecDeque<f64>,
    pub window: usize,
}

impl OUState {
    pub fn new(window: usize) -> Self {
        Self { prices: VecDeque::with_capacity(window + 1), window }
    }

    pub fn push(&mut self, price: f64) {
        self.prices.push_back(price);
        if self.prices.len() > self.window {
            self.prices.pop_front();
        }
    }

    /// Estimate (theta, long_mean, sigma) via OLS on log-prices.
    /// Returns None if insufficient data.
    pub fn estimate_params(&self) -> Option<(f64, f64, f64)> {
        let n = self.prices.len();
        if n < 10 {
            return None;
        }

        let lp: Vec<f64> = self.prices.iter().map(|p| p.ln()).collect();
        // Regress lp[t] on lp[t-1] using OLS: lp[t] = a + b*lp[t-1] + eps
        let n_f = (n - 1) as f64;
        let sum_x: f64 = lp[..n - 1].iter().sum();
        let sum_y: f64 = lp[1..].iter().sum();
        let sum_xx: f64 = lp[..n - 1].iter().map(|x| x * x).sum();
        let sum_xy: f64 = lp[..n - 1].iter().zip(lp[1..].iter()).map(|(x, y)| x * y).sum();

        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return None;
        }

        let b = (n_f * sum_xy - sum_x * sum_y) / denom;
        let a = (sum_y - b * sum_x) / n_f;

        // theta = -ln(b), long_mean = a / (1 - b)
        let theta = -b.ln().max(0.0);
        let long_mean = if (1.0 - b).abs() > 1e-8 { a / (1.0 - b) } else { lp.iter().sum::<f64>() / lp.len() as f64 };

        // Residual std dev.
        let residuals: Vec<f64> = lp[..n - 1]
            .iter()
            .zip(lp[1..].iter())
            .map(|(x, y)| y - (a + b * x))
            .collect();
        let sigma = (residuals.iter().map(|r| r * r).sum::<f64>() / (n_f - 2.0).max(1.0)).sqrt();

        Some((theta, long_mean, sigma))
    }

    /// Z-score of current price relative to OU equilibrium.
    /// Positive => price above long-run mean (mean-revert down expected).
    pub fn z_score(&self, price: f64) -> Option<f64> {
        let (_, long_mean, sigma) = self.estimate_params()?;
        let lp = price.ln();
        if sigma < 1e-10 {
            return None;
        }
        Some((lp - long_mean) / sigma)
    }
}

// ─── BH State Machine ────────────────────────────────────────────────────────

/// Core Black-Hole physics state for a single instrument.
///
/// The BH metaphor maps price dynamics to spacetime geometry:
/// - `cf` (Chandrasekhar factor) sets the event-horizon radius in price-space.
/// - A move is *timelike* (inside the light-cone) when beta = |dp|/(p*cf) < 1.
/// - Timelike moves grow mass via EMA; spacelike moves trigger mass collapse.
/// - When mass exceeds a threshold the BH "activates" (signals a trade).
#[derive(Debug, Clone)]
pub struct BHState {
    /// Chandrasekhar factor (event-horizon scale). Tuned per instrument.
    pub cf: f64,
    /// Accumulated mass (0.0 → 1.0 scale typical).
    pub mass: f64,
    /// Whether the BH is currently active (i.e., a position is signalled).
    pub active: bool,
    /// Direction: +1 long, -1 short, 0 flat.
    pub bh_dir: i8,
    /// Continuous tracking level (support/resistance estimate).
    pub ctl: f64,
    /// Recent close prices for beta computation.
    pub prices: VecDeque<f64>,
    /// Scale factor for cf (e.g. GARCH vol multiplier).
    pub cf_scale: f64,

    // Hyper-parameters (fixed after init).
    mass_decay: f64,
    mass_collapse: f64,
    activation_threshold: f64,
    deactivation_threshold: f64,
    ema_alpha: f64,
}

impl BHState {
    pub fn new(cf: f64) -> Self {
        Self {
            cf,
            mass: 0.0,
            active: false,
            bh_dir: 0,
            ctl: 0.0,
            prices: VecDeque::with_capacity(8),
            cf_scale: 1.0,
            mass_decay: 0.02,
            mass_collapse: 0.30,
            activation_threshold: 0.65,
            deactivation_threshold: 0.20,
            ema_alpha: 2.0 / 6.0, // 5-bar EMA
        }
    }

    /// Ingest a new close price. Returns `true` if activation state changed.
    pub fn update(&mut self, price: f64) -> bool {
        let prev_price = self.prices.back().copied().unwrap_or(price);
        self.prices.push_back(price);
        if self.prices.len() > 6 {
            self.prices.pop_front();
        }

        let effective_cf = self.cf * self.cf_scale;

        // Minkowski metric: beta = |dp| / (p * cf).
        let dp = (price - prev_price).abs();
        let beta = dp / (prev_price * effective_cf + 1e-12);

        // ds² = 1 - β²
        let ds2 = 1.0 - beta * beta;
        let timelike = ds2 > 0.0;

        let prev_active = self.active;

        if timelike {
            // Mass grows: EMA toward 1.0.
            self.mass = self.mass + self.ema_alpha * (1.0 - self.mass) - self.mass_decay;
            self.mass = self.mass.clamp(0.0, 1.0);

            // Update direction from price momentum.
            if price > prev_price {
                self.bh_dir = 1;
            } else if price < prev_price {
                self.bh_dir = -1;
            }
        } else {
            // Spacelike event: mass collapses.
            self.mass *= self.mass_collapse;

            // Potentially flip direction on spacelike break.
            if price > prev_price {
                self.bh_dir = 1;
            } else {
                self.bh_dir = -1;
            }
        }

        // Update CTL as an EMA of price.
        if self.ctl == 0.0 {
            self.ctl = price;
        } else {
            self.ctl = self.ctl + self.ema_alpha * 0.5 * (price - self.ctl);
        }

        // State transitions.
        if !self.active && self.mass >= self.activation_threshold {
            self.active = true;
        } else if self.active && self.mass <= self.deactivation_threshold {
            self.active = false;
            self.bh_dir = 0;
        }

        self.active != prev_active
    }

    /// Reset state (e.g., after a forced exit).
    pub fn reset(&mut self) {
        self.mass = 0.0;
        self.active = false;
        self.bh_dir = 0;
        self.prices.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_garch_variance_positive() {
        let mut g = GARCHState::default_crypto();
        for ret in [-0.02, 0.01, -0.015, 0.03, -0.005] {
            g.update(ret);
            assert!(g.variance > 0.0);
        }
    }

    #[test]
    fn test_garch_vol_scale_reasonable() {
        let mut g = GARCHState::default_crypto();
        // Feed some typical crypto returns.
        for ret in [0.01, -0.02, 0.005, -0.008, 0.015, -0.003, 0.02, -0.01] {
            g.update(ret);
        }
        let vol = g.vol_scale();
        // Annualised vol should be between 10% and 300% for typical crypto.
        assert!(vol > 0.10, "vol too low: {}", vol);
        assert!(vol < 3.00, "vol too high: {}", vol);
    }

    #[test]
    fn test_ou_zscore_returns_value() {
        let mut ou = OUState::new(50);
        // Prices around 100 with small noise.
        let base = 100.0_f64;
        for i in 0..40 {
            ou.push(base + (i as f64 * 0.01).sin() * 0.5);
        }
        let z = ou.z_score(105.0);
        // z should be positive (price above mean).
        assert!(z.is_some());
        assert!(z.unwrap() > 0.0, "expected positive z-score for price above mean");
    }

    #[test]
    fn test_bh_activation_on_sustained_moves() {
        let mut bh = BHState::new(0.005);
        let mut activated = false;
        // Feed many small upward ticks (timelike) to grow mass.
        let mut price = 50000.0_f64;
        for _ in 0..100 {
            price *= 1.0002; // tiny step, well within cf=0.005 horizon.
            if bh.update(price) && bh.active {
                activated = true;
            }
        }
        assert!(activated, "BH should have activated after sustained timelike moves");
        assert_eq!(bh.bh_dir, 1);
    }

    #[test]
    fn test_bh_deactivation_on_spacelike_shock() {
        let mut bh = BHState::new(0.005);
        let mut price = 50000.0_f64;
        // Grow mass first.
        for _ in 0..100 {
            price *= 1.0002;
            bh.update(price);
        }
        // Now deliver a large spacelike shock to collapse mass.
        for _ in 0..10 {
            price *= 0.95; // -5% each bar >> cf threshold.
            bh.update(price);
        }
        assert!(!bh.active || bh.mass < 0.5, "mass should have collapsed after spacelike shocks");
    }

    #[test]
    fn test_bh_reset() {
        let mut bh = BHState::new(0.005);
        let mut price = 50000.0_f64;
        for _ in 0..100 {
            price *= 1.0003;
            bh.update(price);
        }
        bh.reset();
        assert!(!bh.active);
        assert_eq!(bh.mass, 0.0);
        assert_eq!(bh.bh_dir, 0);
    }

    #[test]
    fn test_garch_inv_vol_scale() {
        let mut g = GARCHState::default_crypto();
        for ret in [0.02, -0.01, 0.015] {
            g.update(ret);
        }
        let scale = g.inv_vol_scale(0.40);
        assert!(scale > 0.0);
        assert!(scale <= 5.0);
    }
}
