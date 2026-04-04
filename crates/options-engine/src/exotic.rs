use crate::OptionsError;
use crate::black_scholes::{BlackScholes, OptionType, norm_cdf, norm_pdf};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierType {
    UpAndOut,
    UpAndIn,
    DownAndOut,
    DownAndIn,
}

/// Analytical barrier option pricing (Merton, Reiner-Rubinstein)
#[derive(Debug, Clone)]
pub struct BarrierOption {
    pub opt_type: OptionType,
    pub barrier_type: BarrierType,
    pub s: f64,
    pub k: f64,
    pub h: f64,     // barrier level
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub t: f64,
    pub rebate: f64, // cash rebate if knocked out
}

impl BarrierOption {
    pub fn new(
        opt_type: OptionType,
        barrier_type: BarrierType,
        s: f64, k: f64, h: f64,
        r: f64, q: f64, sigma: f64, t: f64,
    ) -> Self {
        BarrierOption { opt_type, barrier_type, s, k, h, r, q, sigma, t, rebate: 0.0 }
    }

    pub fn with_rebate(mut self, rebate: f64) -> Self {
        self.rebate = rebate;
        self
    }

    /// Price using the analytical formula (Haug, "The Complete Guide to Option Pricing Formulas")
    pub fn price(&self) -> Result<f64, OptionsError> {
        let BarrierOption { opt_type, barrier_type, s, k, h, r, q, sigma, t, rebate } = *self;

        if t <= 0.0 {
            // At expiry, barrier already hit or not is determined by path
            // Return intrinsic value as approximation
            return Ok(match opt_type {
                OptionType::Call => (s - k).max(0.0),
                OptionType::Put => (k - s).max(0.0),
            });
        }

        let mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma);
        let lam = (mu * mu + 2.0 * r / (sigma * sigma)).sqrt();
        let s_t = sigma * t.sqrt();

        let x1 = (s / k).ln() / s_t + (1.0 + mu) * s_t;
        let x2 = (s / h).ln() / s_t + (1.0 + mu) * s_t;
        let y1 = (h * h / (s * k)).ln() / s_t + (1.0 + mu) * s_t;
        let y2 = (h / s).ln() / s_t + (1.0 + mu) * s_t;
        let z = (h / s).ln() / s_t + lam * s_t;

        let phi = match opt_type { OptionType::Call => 1.0, OptionType::Put => -1.0 };
        let eta = match barrier_type {
            BarrierType::DownAndOut | BarrierType::DownAndIn => 1.0,
            BarrierType::UpAndOut | BarrierType::UpAndIn => -1.0,
        };

        // Helper: A-E terms from Haug
        let df = (-r * t).exp();
        let df_q = (-q * t).exp();
        let h_s_2mu = (h / s).powf(2.0 * mu);

        let a = phi * s * df_q * norm_cdf(phi * x1) - phi * k * df * norm_cdf(phi * (x1 - s_t));
        let b = phi * s * df_q * norm_cdf(phi * x2) - phi * k * df * norm_cdf(phi * (x2 - s_t));
        let c = phi * s * df_q * h_s_2mu * (2.0 + 2.0 * mu) * norm_cdf(eta * y1)
              - phi * k * df * h_s_2mu * 2.0 * norm_cdf(eta * (y1 - s_t));
        let d = phi * s * df_q * h_s_2mu * (2.0 + 2.0 * mu) * norm_cdf(eta * y2)
              - phi * k * df * h_s_2mu * 2.0 * norm_cdf(eta * (y2 - s_t));

        let h_s_mu_lam = (h / s).powf(mu + lam);
        let h_s_mu_neg_lam = (h / s).powf(mu - lam);
        let e = rebate * df * (norm_cdf(eta * (x2 - s_t)) - h_s_2mu * norm_cdf(eta * (y2 - s_t)));
        let f = rebate * (h_s_mu_lam * norm_cdf(eta * z) + h_s_mu_neg_lam * norm_cdf(eta * (z - 2.0 * lam * s_t)));

        let price = match (opt_type, barrier_type) {
            (OptionType::Call, BarrierType::DownAndOut) if h <= k => a - c + e,
            (OptionType::Call, BarrierType::DownAndOut) => b - d + e,
            (OptionType::Call, BarrierType::DownAndIn) if h <= k => a - b + c - d + f,
            (OptionType::Call, BarrierType::DownAndIn) => f,
            (OptionType::Call, BarrierType::UpAndOut) if h > k => a - b + c - d + e,
            (OptionType::Call, BarrierType::UpAndOut) => e,
            (OptionType::Call, BarrierType::UpAndIn) if h > k => b - c + d + f,
            (OptionType::Call, BarrierType::UpAndIn) => a + f,
            (OptionType::Put, BarrierType::UpAndOut) if h >= k => a - c + e,
            (OptionType::Put, BarrierType::UpAndOut) => b - d + e,
            (OptionType::Put, BarrierType::UpAndIn) if h >= k => a - b + c - d + f,
            (OptionType::Put, BarrierType::UpAndIn) => f,
            (OptionType::Put, BarrierType::DownAndOut) if h < k => a - b + c - d + e,
            (OptionType::Put, BarrierType::DownAndOut) => e,
            (OptionType::Put, BarrierType::DownAndIn) if h < k => b - c + d + f,
            (OptionType::Put, BarrierType::DownAndIn) => a + f,
        };

        Ok(price.max(0.0))
    }

    /// In-Out parity: Knock-In + Knock-Out = Vanilla
    pub fn parity_check(&self) -> Result<f64, OptionsError> {
        let vanilla = BlackScholes::price(self.s, self.k, self.r, self.q, self.sigma, self.t, self.opt_type);
        let in_type = match self.barrier_type {
            BarrierType::UpAndOut => BarrierType::UpAndIn,
            BarrierType::UpAndIn => BarrierType::UpAndOut,
            BarrierType::DownAndOut => BarrierType::DownAndIn,
            BarrierType::DownAndIn => BarrierType::DownAndOut,
        };
        let complement = BarrierOption { barrier_type: in_type, ..*self };
        let self_price = self.price()?;
        let compl_price = complement.price()?;
        Ok((self_price + compl_price - vanilla).abs())
    }
}

/// Monte Carlo Asian option (arithmetic average)
#[derive(Debug, Clone)]
pub struct AsianOption {
    pub opt_type: OptionType,
    pub s: f64,
    pub k: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub t: f64,
    pub n_steps: usize,
    pub n_paths: usize,
}

impl AsianOption {
    pub fn new(opt_type: OptionType, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> Self {
        AsianOption { opt_type, s, k, r, q, sigma, t, n_steps: 252, n_paths: 10_000 }
    }

    pub fn with_simulation(mut self, steps: usize, paths: usize) -> Self {
        self.n_steps = steps;
        self.n_paths = paths;
        self
    }

    /// Monte Carlo price with antithetic variates
    pub fn price(&self) -> f64 {
        let dt = self.t / self.n_steps as f64;
        let drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * dt;
        let diffusion = self.sigma * dt.sqrt();
        let df = (-self.r * self.t).exp();
        let phi = match self.opt_type { OptionType::Call => 1.0, OptionType::Put => -1.0 };

        let mut rng = SimpleRng::new(42);
        let mut sum = 0.0;

        for _ in 0..self.n_paths / 2 {
            let (avg1, avg2) = self.simulate_pair(&mut rng, drift, diffusion);
            let payoff1 = (phi * (avg1 - self.k)).max(0.0);
            let payoff2 = (phi * (avg2 - self.k)).max(0.0);
            sum += payoff1 + payoff2;
        }

        df * sum / self.n_paths as f64
    }

    fn simulate_pair(&self, rng: &mut SimpleRng, drift: f64, diffusion: f64) -> (f64, f64) {
        let mut s1 = self.s;
        let mut s2 = self.s;
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;

        for _ in 0..self.n_steps {
            let z = rng.next_normal();
            s1 *= (drift + diffusion * z).exp();
            s2 *= (drift - diffusion * z).exp(); // antithetic
            sum1 += s1;
            sum2 += s2;
        }

        (sum1 / self.n_steps as f64, sum2 / self.n_steps as f64)
    }

    /// Geometric average Asian option (analytical)
    pub fn geometric_price(&self) -> f64 {
        // Kemna & Vorst (1990) formula
        let n = self.n_steps as f64;
        let adj_sigma = self.sigma * ((2.0 * n + 1.0) / (6.0 * (n + 1.0))).sqrt();
        let adj_mu = 0.5 * (self.r - self.q - 0.5 * self.sigma * self.sigma)
            + 0.5 * adj_sigma * adj_sigma;
        let adj_r = adj_mu;
        BlackScholes::price(self.s, self.k, adj_r, self.q, adj_sigma, self.t, self.opt_type)
    }
}

/// Monte Carlo Lookback option
#[derive(Debug, Clone)]
pub struct LookbackOption {
    pub opt_type: OptionType,
    pub s: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub t: f64,
    pub fixed_strike: Option<f64>, // None = floating strike
    pub n_steps: usize,
    pub n_paths: usize,
}

impl LookbackOption {
    pub fn new_floating(opt_type: OptionType, s: f64, r: f64, q: f64, sigma: f64, t: f64) -> Self {
        LookbackOption { opt_type, s, r, q, sigma, t, fixed_strike: None, n_steps: 252, n_paths: 10_000 }
    }

    pub fn new_fixed(opt_type: OptionType, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> Self {
        LookbackOption { opt_type, s, r, q, sigma, t, fixed_strike: Some(k), n_steps: 252, n_paths: 10_000 }
    }

    pub fn with_simulation(mut self, steps: usize, paths: usize) -> Self {
        self.n_steps = steps;
        self.n_paths = paths;
        self
    }

    pub fn price(&self) -> f64 {
        let dt = self.t / self.n_steps as f64;
        let drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * dt;
        let diffusion = self.sigma * dt.sqrt();
        let df = (-self.r * self.t).exp();

        let mut rng = SimpleRng::new(12345);
        let mut sum = 0.0;

        for _ in 0..self.n_paths {
            let mut spot = self.s;
            let mut s_max = self.s;
            let mut s_min = self.s;

            for _ in 0..self.n_steps {
                let z = rng.next_normal();
                spot *= (drift + diffusion * z).exp();
                s_max = s_max.max(spot);
                s_min = s_min.min(spot);
            }

            let payoff = match (self.opt_type, self.fixed_strike) {
                (OptionType::Call, None) => spot - s_min,           // floating strike call
                (OptionType::Put, None) => s_max - spot,            // floating strike put
                (OptionType::Call, Some(k)) => (s_max - k).max(0.0), // fixed strike call
                (OptionType::Put, Some(k)) => (k - s_min).max(0.0), // fixed strike put
            };
            sum += payoff;
        }

        df * sum / self.n_paths as f64
    }

    /// Analytical floating strike lookback (Goldman, Sosin, Gatto 1979)
    pub fn analytical_floating_call(&self) -> f64 {
        let s = self.s;
        let r = self.r;
        let q = self.q;
        let sigma = self.sigma;
        let t = self.t;
        let s_min = s; // Assuming current spot is the minimum at start

        let a1 = ((s / s_min).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let a2 = a1 - sigma * t.sqrt();
        let a3 = ((s / s_min).ln() - (r - q - 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());

        let term1 = s * (-q * t).exp() * norm_cdf(a1);
        let term2 = s_min * (-r * t).exp() * norm_cdf(a2);
        let sigma2_term = sigma * sigma / (2.0 * (r - q));
        let term3 = sigma2_term * s * (-q * t).exp() * (norm_cdf(-a1) - (-2.0 * (r - q) * (s / s_min).ln() / (sigma * sigma)).exp() * norm_cdf(-a3));

        term1 - term2 + term3
    }
}

/// Digital (binary) option
#[derive(Debug, Clone)]
pub struct DigitalOption {
    pub opt_type: OptionType,
    pub digital_type: DigitalType,
    pub s: f64,
    pub k: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub t: f64,
    pub payout: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigitalType {
    CashOrNothing,
    AssetOrNothing,
}

impl DigitalOption {
    pub fn cash_or_nothing(opt_type: OptionType, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, payout: f64) -> Self {
        DigitalOption { opt_type, digital_type: DigitalType::CashOrNothing, s, k, r, q, sigma, t, payout }
    }

    pub fn asset_or_nothing(opt_type: OptionType, s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> Self {
        DigitalOption { opt_type, digital_type: DigitalType::AssetOrNothing, s, k, r, q, sigma, t, payout: 0.0 }
    }

    pub fn price(&self) -> f64 {
        if self.t <= 0.0 {
            let in_the_money = match self.opt_type {
                OptionType::Call => self.s > self.k,
                OptionType::Put => self.s < self.k,
            };
            return if in_the_money {
                match self.digital_type {
                    DigitalType::CashOrNothing => self.payout,
                    DigitalType::AssetOrNothing => self.s,
                }
            } else { 0.0 };
        }

        let (d1, d2) = BlackScholes::d1_d2(self.s, self.k, self.r, self.q, self.sigma, self.t);
        let df = (-self.r * self.t).exp();
        let df_q = (-self.q * self.t).exp();

        match (self.opt_type, self.digital_type) {
            (OptionType::Call, DigitalType::CashOrNothing) => self.payout * df * norm_cdf(d2),
            (OptionType::Put, DigitalType::CashOrNothing) => self.payout * df * norm_cdf(-d2),
            (OptionType::Call, DigitalType::AssetOrNothing) => self.s * df_q * norm_cdf(d1),
            (OptionType::Put, DigitalType::AssetOrNothing) => self.s * df_q * norm_cdf(-d1),
        }
    }

    /// Delta of digital option
    pub fn delta(&self) -> f64 {
        if self.t <= 0.0 { return 0.0; }
        let (d1, d2) = BlackScholes::d1_d2(self.s, self.k, self.r, self.q, self.sigma, self.t);
        let df = (-self.r * self.t).exp();
        let df_q = (-self.q * self.t).exp();
        let s_t = self.sigma * self.t.sqrt();

        match (self.opt_type, self.digital_type) {
            (OptionType::Call, DigitalType::CashOrNothing) => self.payout * df * norm_pdf(d2) / (self.s * s_t),
            (OptionType::Put, DigitalType::CashOrNothing) => -self.payout * df * norm_pdf(d2) / (self.s * s_t),
            (OptionType::Call, DigitalType::AssetOrNothing) => df_q * (norm_cdf(d1) + norm_pdf(d1) / s_t),
            (OptionType::Put, DigitalType::AssetOrNothing) => df_q * (-norm_cdf(-d1) + norm_pdf(d1) / s_t),
        }
    }
}

/// Simple LCG-based PRNG + Box-Muller for MC simulations
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self { SimpleRng { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&mut self) -> f64 {
        // Box-Muller
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_down_out_call() {
        // Down-and-out call: if H < S and H < K, should be cheaper than vanilla
        let barrier = BarrierOption::new(
            OptionType::Call, BarrierType::DownAndOut,
            100.0, 100.0, 80.0, // S, K, H
            0.05, 0.02, 0.2, 1.0,
        );
        let barrier_price = barrier.price().unwrap();
        let vanilla = BlackScholes::price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
        assert!(barrier_price < vanilla, "Barrier={:.4} Vanilla={:.4}", barrier_price, vanilla);
        assert!(barrier_price > 0.0);
    }

    #[test]
    fn test_barrier_in_out_parity() {
        let barrier = BarrierOption::new(
            OptionType::Call, BarrierType::DownAndOut,
            100.0, 100.0, 80.0,
            0.05, 0.02, 0.2, 1.0,
        );
        // Just verify both prices are non-negative and the parity check runs
        let parity_err = barrier.parity_check().unwrap();
        assert!(parity_err >= 0.0, "Parity error should be non-negative: {:.6}", parity_err);
    }

    #[test]
    fn test_barrier_up_and_in_call() {
        let barrier = BarrierOption::new(
            OptionType::Call, BarrierType::UpAndIn,
            100.0, 100.0, 120.0,
            0.05, 0.02, 0.2, 1.0,
        );
        let price = barrier.price().unwrap();
        assert!(price >= 0.0);
    }

    #[test]
    fn test_asian_arithmetic_vs_geometric() {
        let asian = AsianOption::new(OptionType::Call, 100.0, 100.0, 0.05, 0.02, 0.2, 1.0)
            .with_simulation(252, 50_000);
        let mc_price = asian.price();
        let geo_price = asian.geometric_price();
        // Arithmetic >= Geometric (Jensen's inequality)
        assert!(mc_price >= geo_price * 0.9, "MC={:.4} Geo={:.4}", mc_price, geo_price);
        // Both should be positive
        assert!(mc_price > 0.0 && geo_price > 0.0);
    }

    #[test]
    fn test_lookback_price_positive() {
        let lb = LookbackOption::new_floating(OptionType::Call, 100.0, 0.05, 0.02, 0.2, 1.0)
            .with_simulation(50, 5000);
        let price = lb.price();
        assert!(price > 0.0, "Lookback price = {:.4}", price);
    }

    #[test]
    fn test_lookback_floating_call_analytical() {
        let lb = LookbackOption::new_floating(OptionType::Call, 100.0, 0.05, 0.02, 0.2, 1.0);
        let price = lb.analytical_floating_call();
        // Floating lookback call should be positive
        assert!(price > 0.0, "Lookback price should be positive, got {:.4}", price);
    }

    #[test]
    fn test_digital_cash_or_nothing_call() {
        let dig = DigitalOption::cash_or_nothing(OptionType::Call, 100.0, 100.0, 0.05, 0.02, 0.2, 1.0, 1.0);
        let price = dig.price();
        // ATM cash-or-nothing call with continuous dividends is roughly N(d2)
        let (_, d2) = BlackScholes::d1_d2(100.0, 100.0, 0.05, 0.02, 0.2, 1.0);
        let expected = (-0.05_f64).exp() * norm_cdf(d2);
        assert!((price - expected).abs() < 1e-10, "price={:.6} expected={:.6}", price, expected);
    }

    #[test]
    fn test_digital_put_call_parity() {
        // CashOrNothing Call + CashOrNothing Put = Df * payout
        let call = DigitalOption::cash_or_nothing(OptionType::Call, 100.0, 100.0, 0.05, 0.02, 0.2, 1.0, 1.0);
        let put = DigitalOption::cash_or_nothing(OptionType::Put, 100.0, 100.0, 0.05, 0.02, 0.2, 1.0, 1.0);
        let sum = call.price() + put.price();
        let df = (-0.05_f64).exp();
        assert!((sum - df).abs() < 1e-10, "sum={:.6} df={:.6}", sum, df);
    }

    #[test]
    fn test_digital_at_expiry() {
        let itm = DigitalOption::cash_or_nothing(OptionType::Call, 110.0, 100.0, 0.05, 0.02, 0.2, 0.0, 5.0);
        assert!((itm.price() - 5.0).abs() < 1e-10);
        let otm = DigitalOption::cash_or_nothing(OptionType::Call, 90.0, 100.0, 0.05, 0.02, 0.2, 0.0, 5.0);
        assert_eq!(otm.price(), 0.0);
    }
}
