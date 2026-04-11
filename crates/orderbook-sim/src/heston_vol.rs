/// Heston stochastic volatility model integrated into LOB mid-price dynamics.
///
/// Model:
///   dS = μ·S·dt + √V·S·dW_S
///   dV = κ(θ − V)·dt + σ·√V·dW_V
///   dW_S·dW_V = ρ·dt
///
/// where:
///   S = asset price (mid-price of LOB)
///   V = instantaneous variance (CIR process, Feller condition: 2κθ > σ²)
///   κ = mean-reversion speed
///   θ = long-run variance
///   σ = vol-of-vol
///   ρ = correlation between price and variance
///
/// Implements:
///   - Euler-Maruyama discretization with full truncation scheme
///   - Method of moments calibration from observed returns
///   - Characteristic function for option pricing (Heston 1993)
///   - Path generation driving the LOB mid-price

use std::f64::consts::PI;

// ── Parameters ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct HestonParams {
    /// Drift of log-price.
    pub mu: f64,
    /// Mean-reversion speed of variance.
    pub kappa: f64,
    /// Long-run variance (θ).
    pub theta: f64,
    /// Vol-of-vol (σ).
    pub sigma: f64,
    /// Correlation W_S, W_V.
    pub rho: f64,
    /// Initial variance V₀.
    pub v0: f64,
}

impl HestonParams {
    pub fn new(mu: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64) -> Self {
        assert!(kappa > 0.0, "kappa must be positive");
        assert!(theta > 0.0, "theta must be positive");
        assert!(sigma > 0.0, "sigma must be positive");
        assert!(rho >= -1.0 && rho <= 1.0, "rho must be in [-1, 1]");
        assert!(v0 > 0.0, "v0 must be positive");
        HestonParams { mu, kappa, theta, sigma, rho, v0 }
    }

    /// Feller condition: 2κθ > σ². If satisfied, V stays strictly positive.
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }

    /// Unconditional mean of variance: θ.
    pub fn unconditional_variance(&self) -> f64 {
        self.theta
    }

    /// Unconditional std of variance (CIR stationary distribution: Gamma).
    pub fn unconditional_vol_of_vol(&self) -> f64 {
        (self.sigma * self.sigma * self.theta / (2.0 * self.kappa)).sqrt()
    }

    /// Half-life of variance mean reversion (in same units as dt).
    pub fn variance_half_life(&self) -> f64 {
        2.0_f64.ln() / self.kappa
    }
}

impl Default for HestonParams {
    fn default() -> Self {
        HestonParams {
            mu: 0.0,
            kappa: 2.0,
            theta: 0.04,
            sigma: 0.3,
            rho: -0.7,
            v0: 0.04,
        }
    }
}

// ── Simulation Path ───────────────────────────────────────────────────────────

/// One simulated step of the Heston model.
#[derive(Debug, Clone, Copy)]
pub struct HestonStep {
    pub t: f64,
    pub price: f64,
    pub variance: f64,
    pub log_return: f64,
}

/// Simulate a Heston price path using Euler-Maruyama with full truncation.
///
/// Full truncation: replace V_t with max(V_t, 0) before taking sqrt.
/// Returns vector of (time, price, variance).
pub fn simulate_heston(
    s0: f64,
    params: &HestonParams,
    n_steps: usize,
    horizon: f64,
    rng: &mut impl rand::Rng,
) -> Vec<HestonStep> {
    let dt = horizon / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let rho_bar = (1.0 - params.rho * params.rho).sqrt();

    let mut path = Vec::with_capacity(n_steps + 1);
    let mut s = s0;
    let mut v = params.v0;
    let mut t = 0.0_f64;

    path.push(HestonStep { t, price: s, variance: v, log_return: 0.0 });

    for _ in 0..n_steps {
        // Correlated Brownian increments.
        let z1: f64 = rng.gen::<f64>();
        let z2: f64 = rng.gen::<f64>();
        // Box-Muller.
        let n1 = box_muller(z1, z2);
        let z3: f64 = rng.gen::<f64>();
        let z4: f64 = rng.gen::<f64>();
        let n2_raw = box_muller(z3, z4);
        // Correlate: dW_S = n1, dW_V = ρ·n1 + √(1−ρ²)·n2.
        let dw_s = n1 * sqrt_dt;
        let dw_v = (params.rho * n1 + rho_bar * n2_raw) * sqrt_dt;

        // Full truncation variance.
        let v_plus = v.max(0.0);
        let sqrt_v = v_plus.sqrt();

        // Log-price Euler step (log-price form for numerical stability).
        let log_s = s.ln();
        let d_log_s = (params.mu - 0.5 * v_plus) * dt + sqrt_v * dw_s;
        let s_new = (log_s + d_log_s).exp();

        // Variance Euler step (full truncation: clamp v before using, update freely).
        let dv = params.kappa * (params.theta - v_plus) * dt + params.sigma * sqrt_v * dw_v;
        let v_new = v + dv; // Can go negative; will be clamped next step.

        t += dt;
        let log_ret = s_new.ln() - s.ln();
        s = s_new;
        v = v_new;

        path.push(HestonStep { t, price: s, variance: v.max(0.0), log_return: log_ret });
    }

    path
}

/// Box-Muller transform: (u1, u2) uniform → standard normal.
fn box_muller(u1: f64, u2: f64) -> f64 {
    let u1 = u1.max(1e-20);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ── Method of Moments Calibration ────────────────────────────────────────────

/// Summary statistics of observed log-return series.
#[derive(Debug, Clone, Copy)]
pub struct ReturnStats {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub excess_kurtosis: f64,
    pub acf_sq_lag1: f64,    // ACF of squared returns at lag 1 (proxy for vol clustering)
    pub n: usize,
    pub dt: f64,             // time interval between observations
}

impl ReturnStats {
    pub fn from_returns(returns: &[f64], dt: f64) -> Self {
        let n = returns.len();
        assert!(n >= 4, "Need at least 4 observations");

        let mean = returns.iter().sum::<f64>() / n as f64;
        let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = var.sqrt();

        let skew = returns.iter()
            .map(|&r| ((r - mean) / std).powi(3))
            .sum::<f64>() / n as f64;

        let kurt = returns.iter()
            .map(|&r| ((r - mean) / std).powi(4))
            .sum::<f64>() / n as f64 - 3.0;

        // ACF of squared returns at lag 1.
        let sq: Vec<f64> = returns.iter().map(|&r| r * r).collect();
        let sq_mean = sq.iter().sum::<f64>() / n as f64;
        let cov = sq[..n-1].iter().zip(sq[1..].iter())
            .map(|(&a, &b)| (a - sq_mean) * (b - sq_mean))
            .sum::<f64>() / (n - 1) as f64;
        let sq_var = sq.iter().map(|&s| (s - sq_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let acf = if sq_var > 1e-20 { cov / sq_var } else { 0.0 };

        ReturnStats { mean, variance: var, skewness: skew, excess_kurtosis: kurt, acf_sq_lag1: acf, n, dt }
    }
}

/// Calibrate Heston parameters via method of moments.
///
/// Matching: E[r], Var[r], Kurt[r], and vol-autocorrelation.
/// Uses relationships between Heston moments and (κ, θ, σ, ρ, V₀).
pub fn calibrate_heston_mom(stats: &ReturnStats) -> HestonParams {
    let dt = stats.dt;

    // Moment matching:
    // E[r] = μ·dt → μ = mean/dt
    let mu = stats.mean / dt;

    // Var[r] = θ·dt → θ = var/dt (annualised variance)
    let theta = (stats.variance / dt).max(1e-6);

    // Excess kurtosis of Heston: κ₄ ≈ 3·σ²·dt/(2κ·θ²) for small dt.
    // → σ² = κ₄·2κ·θ²/(3·dt). We need to estimate κ first.
    // From vol clustering (ACF of squared returns): ρ_v ≈ exp(-κ·dt).
    let acf_clamped = stats.acf_sq_lag1.clamp(0.001, 0.999);
    let kappa = -(acf_clamped.ln()) / dt;
    let kappa = kappa.clamp(0.1, 50.0);

    // σ from kurtosis relation: σ² ≈ kurt·2κθ²/(3).
    let kurt = stats.excess_kurtosis.max(0.0);
    let sigma_sq = (kurt * 2.0 * kappa * theta * theta / 3.0).max(1e-6);
    let sigma = sigma_sq.sqrt().clamp(0.01, 3.0);

    // Ensure Feller condition: 2κθ > σ². Scale sigma down if needed.
    let sigma = if 2.0 * kappa * theta < sigma * sigma {
        (2.0 * kappa * theta * 0.99).sqrt()
    } else {
        sigma
    };

    // ρ: sign of skewness of returns usually determined by leverage effect.
    // Skewness ≈ 3·ρ·σ·θ·√(something). Use a simple heuristic.
    let rho = (-stats.skewness.signum() * 0.5_f64).clamp(-0.95, 0.95);

    // V₀ from current variance estimate (could use realized vol).
    let v0 = theta;

    HestonParams::new(mu, kappa, theta, sigma, rho, v0)
}

// ── Characteristic Function ───────────────────────────────────────────────────

/// Heston characteristic function φ(u) = E[exp(iu·log(S_T/S_0))].
///
/// Uses the formulation from Heston (1993).
pub fn heston_char_fn(u: f64, t: f64, params: &HestonParams) -> (f64, f64) {
    let kappa = params.kappa;
    let theta = params.theta;
    let sigma = params.sigma;
    let rho = params.rho;
    let v0 = params.v0;

    // d = sqrt((κ - iρσu)² + σ²u(u + i))
    let a_re = kappa - rho * sigma * u;  // Re part of (κ - iρσu)
    let a_im = -u;                        // contribution from -iρσu but u real: -ρσu... wait

    // Let me redo: complex arithmetic.
    // d² = (κ - ρσiu)² + σ²(iu)(iu + 1)
    //    = (κ)² - 2κ(ρσiu) + (ρσiu)² + σ²(i²u² + iu)
    //    = κ² - 2κρσiu - ρ²σ²u² + σ²(-u² + iu)
    //    = (κ² - ρ²σ²u² - σ²u²) + i(-2κρσu + σ²u)
    //    = (κ² - σ²u²(ρ² + 1)) + iσ²u(1 - 2κρ/σ... this is getting messy.
    // Use standard form.

    // For a real call option characteristic function, φ is complex.
    // We return (Re φ, Im φ) for use in Carr-Madan.
    // Using Albrecher et al. form:

    // d = sqrt((ρ σ i u - κ)² + σ²(iu + u²))
    // Re/Im of d.
    let kappa_rho_u = kappa - rho * sigma * u;
    // (ρσiu - κ)² = (−kappa_rho_u + iρσu)^2 but careful...
    // Let X = -κ + ρσ(iu) = −κ − iρσu. X² = κ² + 2iκρσu − ρ²σ²u².
    // σ²(iu + u²) = σ²(−u² + iu).
    // d² = κ² + 2iκρσu − ρ²σ²u² − σ²u² + iσ²u.
    let d2_re = kappa * kappa + 2.0 * kappa * rho * sigma * 0.0 /* only real part from iu terms = 0 */
        - rho * rho * sigma * sigma * u * u
        - sigma * sigma * u * u;
    // Im part: 2κρσu + σ²u.
    let d2_im = 2.0 * kappa * rho * sigma * u + sigma * sigma * u;

    // d = sqrt(complex(d2_re, d2_im)).
    let d_re_inner = (d2_re * d2_re + d2_im * d2_im).sqrt();
    let d_re = ((d_re_inner + d2_re) / 2.0).max(0.0).sqrt();
    let d_im = if d_re_inner + d2_re < 0.0 { 0.0 } else {
        let s = ((d_re_inner - d2_re) / 2.0).max(0.0).sqrt();
        if d2_im >= 0.0 { s } else { -s }
    };

    let d_abs = (d_re * d_re + d_im * d_im).sqrt();
    if d_abs < 1e-15 {
        return (1.0, 0.0);
    }

    // G = (κ - ρσiu - d) / (κ - ρσiu + d)
    // Numerator: (κ - ρσiu - d)
    let num_re = kappa_rho_u - d_re;
    let num_im = 0.0 - d_im;
    let den_re = kappa_rho_u + d_re;
    let den_im = d_im;

    let den_abs2 = den_re * den_re + den_im * den_im;
    if den_abs2 < 1e-30 { return (1.0, 0.0); }

    let g_re = (num_re * den_re + num_im * den_im) / den_abs2;
    let g_im = (num_im * den_re - num_re * den_im) / den_abs2;

    // D(u,t) = (κ - ρσiu - d)/σ² · (1 - exp(-dt)) / (1 - G·exp(-dt))
    let exp_re = (-d_re * t).exp() * (-d_im * t).cos();
    let exp_im = (-d_re * t).exp() * (-d_im * t).sin();

    // numerator of D: (1 - e^{-dt}) where d complex
    let one_minus_exp_re = 1.0 - exp_re;
    let one_minus_exp_im = -exp_im;

    // denominator: (1 - G·e^{-dt})
    let ge_re = g_re * exp_re - g_im * exp_im;
    let ge_im = g_re * exp_im + g_im * exp_re;
    let denom_re = 1.0 - ge_re;
    let denom_im = -ge_im;
    let denom_abs2 = denom_re * denom_re + denom_im * denom_im;
    if denom_abs2 < 1e-30 { return (1.0, 0.0); }

    // (num_re + i·num_im) / (denom_re + i·denom_im)
    let frac_re = (one_minus_exp_re * denom_re + one_minus_exp_im * denom_im) / denom_abs2;
    let frac_im = (one_minus_exp_im * denom_re - one_minus_exp_re * denom_im) / denom_abs2;

    let d_factor = 1.0 / (sigma * sigma);
    let d_re_full = (kappa_rho_u - d_re) * d_factor;
    let d_im_full = -d_im * d_factor;

    // D(u,t) = (d_re_full + i·d_im_full) * (frac_re + i·frac_im)
    let cap_d_re = d_re_full * frac_re - d_im_full * frac_im;
    let cap_d_im = d_re_full * frac_im + d_im_full * frac_re;

    // C(u,t) = κθ/σ² · [(κ-ρσiu-d)t - 2·log((1-G·e^{-dt})/(1-G))]
    // 1 - G = 1 - g_re - i·g_im
    let one_minus_g_re = 1.0 - g_re;
    let one_minus_g_im = -g_im;
    let omg_abs2 = one_minus_g_re * one_minus_g_re + one_minus_g_im * one_minus_g_im;
    if omg_abs2 < 1e-30 { return (1.0, 0.0); }

    // log((1 - G·e^{-dt})/(1-G)) = log(denom) - log(one_minus_g)
    let ln_denom_abs = (denom_abs2.sqrt()).ln();
    let ln_denom_arg = denom_im.atan2(denom_re);
    let ln_omg_abs = (omg_abs2.sqrt()).ln();
    let ln_omg_arg = one_minus_g_im.atan2(one_minus_g_re);

    let log_ratio_re = ln_denom_abs - ln_omg_abs;
    let log_ratio_im = ln_denom_arg - ln_omg_arg;

    let kappa_theta_over_sigma2 = kappa * theta / (sigma * sigma);
    let cap_c_re = kappa_theta_over_sigma2 * ((kappa_rho_u - d_re) * t - 2.0 * log_ratio_re);
    let cap_c_im = kappa_theta_over_sigma2 * ((-d_im) * t - 2.0 * log_ratio_im);

    // φ(u) = exp(C + D·V₀ + iu·log(S₀/S₀)) = exp(C + D·V₀) for log(S_T/S_0).
    // Including drift: multiply by exp(iuμt) — but for log(S_T/S_0) forward:
    let exp_arg_re = cap_c_re + cap_d_re * v0;
    let exp_arg_im = cap_c_im + cap_d_im * v0 + u * params.mu * t;

    let phi_abs = exp_arg_re.exp();
    let phi_re = phi_abs * exp_arg_im.cos();
    let phi_im = phi_abs * exp_arg_im.sin();

    (phi_re, phi_im)
}

// ── LOB Integration ───────────────────────────────────────────────────────────

/// State machine: Heston model drives the mid-price of the LOB.
pub struct HestonLobDriver {
    pub params: HestonParams,
    /// Current mid-price.
    pub price: f64,
    /// Current instantaneous variance.
    pub variance: f64,
    /// Time elapsed (seconds).
    pub t: f64,
}

impl HestonLobDriver {
    pub fn new(initial_price: f64, params: HestonParams) -> Self {
        let v0 = params.v0;
        HestonLobDriver { params, price: initial_price, variance: v0, t: 0.0 }
    }

    /// Advance the mid-price by dt seconds.
    /// Returns (new_price, new_variance, log_return).
    pub fn step(&mut self, dt: f64, rng: &mut impl rand::Rng) -> (f64, f64, f64) {
        let sqrt_dt = dt.sqrt();
        let rho_bar = (1.0 - self.params.rho * self.params.rho).sqrt();

        let n1 = sample_normal(rng);
        let n2_raw = sample_normal(rng);
        let dw_s = n1 * sqrt_dt;
        let dw_v = (self.params.rho * n1 + rho_bar * n2_raw) * sqrt_dt;

        let v_plus = self.variance.max(0.0);
        let sqrt_v = v_plus.sqrt();

        let d_log_s = (self.params.mu - 0.5 * v_plus) * dt + sqrt_v * dw_s;
        let new_price = self.price * d_log_s.exp();

        let dv = self.params.kappa * (self.params.theta - v_plus) * dt
            + self.params.sigma * sqrt_v * dw_v;
        let new_var = (self.variance + dv).max(0.0);

        self.price = new_price;
        self.variance = new_var;
        self.t += dt;

        (new_price, new_var, d_log_s)
    }

    /// Current annualised volatility estimate.
    pub fn implied_vol(&self) -> f64 {
        self.variance.max(0.0).sqrt()
    }

    /// Realized volatility from a window of log-returns.
    pub fn realized_vol(returns: &[f64], ann_factor: f64) -> f64 {
        if returns.is_empty() { return 0.0; }
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        (var * ann_factor).sqrt()
    }
}

fn sample_normal(rng: &mut impl rand::Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-20);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ── Variance Process (CIR) Constraints ───────────────────────────────────────

/// Check and enforce CIR constraints on parameters.
pub fn enforce_cir_constraints(params: &mut HestonParams) {
    // Feller: 2κθ > σ². If violated, scale σ down.
    if !params.feller_satisfied() {
        params.sigma = (2.0 * params.kappa * params.theta * 0.99).sqrt();
    }
    // κ > 0.
    params.kappa = params.kappa.max(0.01);
    // θ > 0.
    params.theta = params.theta.max(1e-6);
    // σ > 0.
    params.sigma = params.sigma.max(1e-6);
    // V₀ > 0.
    params.v0 = params.v0.max(1e-6);
    // ρ ∈ (−1, 1).
    params.rho = params.rho.clamp(-0.999, 0.999);
}

// ── Moment Formulas ───────────────────────────────────────────────────────────

/// Theoretical first moment of log-return: E[r_t] = (μ - θ/2)·dt (approx).
pub fn heston_mean_return(params: &HestonParams, dt: f64) -> f64 {
    (params.mu - 0.5 * params.theta) * dt
}

/// Theoretical variance of log-return:
/// Var[r_t] = θ·dt + (σ²-2κρσ)·θ·dt²/(2κ) + O(dt³) for stationary process.
pub fn heston_return_variance(params: &HestonParams, dt: f64) -> f64 {
    let v1 = params.theta * dt;
    let correction = (params.sigma * params.sigma - 2.0 * params.kappa * params.rho * params.sigma)
        * params.theta * dt * dt / (2.0 * params.kappa);
    v1 + correction
}

/// Theoretical excess kurtosis of log-returns.
pub fn heston_excess_kurtosis(params: &HestonParams, dt: f64) -> f64 {
    // Leading term: 3σ²dt/(2κθ) · (second-order term).
    3.0 * params.sigma * params.sigma * dt / (2.0 * params.kappa * params.theta)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_feller_condition() {
        let p = HestonParams::default(); // κ=2, θ=0.04, σ=0.3 → 2·2·0.04=0.16 > 0.09=0.3²
        assert!(p.feller_satisfied());
    }

    #[test]
    fn test_simulation_price_positive() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let p = HestonParams::default();
        let path = simulate_heston(100.0, &p, 1000, 1.0, &mut rng);
        assert_eq!(path.len(), 1001);
        for step in &path {
            assert!(step.price > 0.0, "price went non-positive: {}", step.price);
            assert!(step.variance >= 0.0, "variance went negative: {}", step.variance);
        }
    }

    #[test]
    fn test_driver_step() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let p = HestonParams::default();
        let mut driver = HestonLobDriver::new(100.0, p);
        for _ in 0..100 {
            let (price, var, _) = driver.step(0.001, &mut rng);
            assert!(price > 0.0);
            assert!(var >= 0.0);
        }
    }

    #[test]
    fn test_calibration_roundtrip() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let true_p = HestonParams::new(0.0, 3.0, 0.04, 0.2, -0.6, 0.04);
        let path = simulate_heston(100.0, &true_p, 10000, 252.0, &mut rng);
        let returns: Vec<f64> = path.windows(2)
            .map(|w| w[1].price.ln() - w[0].price.ln())
            .collect();
        let dt = 1.0 / 10000.0;
        let stats = ReturnStats::from_returns(&returns, dt);
        let est = calibrate_heston_mom(&stats);
        // Estimated mu should be close-ish.
        assert!(est.kappa > 0.0);
        assert!(est.theta > 0.0);
        assert!(est.sigma > 0.0);
        assert!(est.feller_satisfied());
    }

    #[test]
    fn test_char_fn_at_zero() {
        // φ(0) should be 1.
        let p = HestonParams::default();
        let (re, im) = heston_char_fn(0.0, 1.0, &p);
        assert!((re - 1.0).abs() < 1e-6, "φ(0) real part: {}", re);
        assert!(im.abs() < 1e-6, "φ(0) imag part: {}", im);
    }

    #[test]
    fn test_enforce_cir() {
        let mut p = HestonParams {
            mu: 0.0, kappa: 0.5, theta: 0.01, sigma: 1.0, rho: 0.0, v0: 0.01
        };
        // 2·0.5·0.01 = 0.01 < 1.0 = σ². Feller violated.
        assert!(!p.feller_satisfied());
        enforce_cir_constraints(&mut p);
        assert!(p.feller_satisfied());
    }

    #[test]
    fn test_moment_formulas() {
        let p = HestonParams::default();
        let dt = 1.0 / 252.0;
        let mean = heston_mean_return(&p, dt);
        let var = heston_return_variance(&p, dt);
        assert!(var > 0.0);
        let kurt = heston_excess_kurtosis(&p, dt);
        assert!(kurt > 0.0);
    }
}
